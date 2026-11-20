// esp32_emg_streamer: sample N analog EMG channels at 1 kHz and stream them
// over WiFi UDP using the myobridge wire protocol (docs/protocol.md).
//
// Target: any ESP32 dev board (original ESP32 / WROOM-32), Arduino-ESP32 core 3.x.
// Note: core 3.x changed the timer API. timerBegin() now takes a tick frequency
// in Hz (it used to take num, divider, countUp), and timerAlarmWrite/Enable were
// replaced by a single timerAlarm(). This sketch uses the 3.x calls only.
//
// ---------------------------------------------------------------------------
// WIRING (MyoWare 2 or 2.0 muscle sensor, one per channel; docs/hardware.md)
//
//   MyoWare +  ->  ESP32 3V3        (power the sensor at 3.3 V so its output
//   MyoWare -  ->  ESP32 GND         can never exceed the ESP32's ADC range)
//   MyoWare RAW -> ESP32 GPIO 32 / 33 / 34 / 35   (channel 0..3)
//   Reference electrode cable on a bony spot (elbow), see MyoWare guide.
//
//   Use the RAW output, not ENV. ENV is the sensor's on-board rectified and
//   smoothed envelope: handy for "is the muscle on?", but it throws away the
//   frequency content our features need (zero crossings, slope changes) and
//   the PC can't undo its filtering. RAW is the amplified, bandpassed EMG
//   centred at Vcc/2, so we do all the DSP ourselves, in software, where it's
//   easy to change.
//
//   GPIO 32-35 are on ADC1. ADC2 pins can't be read while WiFi is running.
//   (34 and 35 are input-only pins, which is fine for an ADC.)
//
// SAFETY: run the ESP32 + sensors from a battery (LiPo or USB power bank) ONLY.
// Never have electrodes on your skin while the circuit is connected to
// anything mains-powered, e.g. a laptop that is charging over the same USB
// cable. A fault in that supply could put current through your body. Flash the
// board, unplug USB, then put the electrodes on. WiFi is the only link we need.
// ---------------------------------------------------------------------------

#include <WiFi.h>
#include <WiFiUdp.h>

#include "config.h"  // copy config.example.h -> config.h first
#include "mb_decimator.h"
#include "mb_protocol.h"

// ---- Anti-aliasing ---------------------------------------------------------
// The MyoWare's RAW output is band-limited only by a gentle analog roll-off
// above ~500 Hz, which is NOT a proper anti-aliasing filter for 1 kHz
// sampling: wideband electronic noise between 500 Hz and many kHz folds into
// the EMG band. Real EMG itself has almost no power up there, so gestures are
// unaffected; the cost is a higher noise floor (docs/aliasing.md models it at
// about +3 dB for a first-order roll-off). OVERSAMPLE 2 samples at 2 kHz and
// decimates with a halfband FIR (mb_decimator.h), cutting that to about +1 dB.
// The PC still receives 1 kHz. OVERSAMPLE 2 is untested on hardware: measure
// the rest-noise floor both ways during bring-up (docs/hardware-bringup.md).
#ifndef OVERSAMPLE
#define OVERSAMPLE 1
#endif
#if OVERSAMPLE != 1 && OVERSAMPLE != 2
#error "OVERSAMPLE must be 1 or 2"
#endif

// ---- Acquisition settings (keep in sync with what the PC expects) ----------
static const uint8_t NUM_CHANNELS = 4;
static const uint8_t CHANNEL_PINS[NUM_CHANNELS] = {32, 33, 34, 35};
static const uint16_t SAMPLE_RATE_HZ = 1000;
static const uint16_t FRAMES_PER_PACKET = 20;  // 20 ms of data per packet
static const int ADC_MIDPOINT = 2048;           // 12-bit ADC, RAW idles at Vcc/2

static const uint32_t ADC_RATE_HZ = (uint32_t)SAMPLE_RATE_HZ * OVERSAMPLE;
static const size_t PACKET_SIZE = mb::HEADER_SIZE + (size_t)NUM_CHANNELS * FRAMES_PER_PACKET * 2 +
                                  mb::CRC_SIZE;  // wire format: docs/protocol.md

struct Packet {
  uint8_t bytes[PACKET_SIZE];
};

// ---- Globals ---------------------------------------------------------------
static hw_timer_t *sampleTimer = nullptr;
static TaskHandle_t samplerTask = nullptr;
static QueueHandle_t packetQueue = nullptr;  // sampler task -> loop() (sender)
static WiFiUDP udp;

static volatile uint32_t missedTicks = 0;  // sampler fell behind the timer
static volatile uint32_t queueFull = 0;    // sender fell behind the sampler
static uint32_t sendFailed = 0;            // udp.endPacket() failed (loop() only)

// ---- Timer interrupt: runs every 1/ADC_RATE_HZ s -------------------------
// Keep ISRs tiny. analogRead() is not safe/fast enough to call here, so the
// ISR only wakes the sampler task, which does the actual reading.
void ARDUINO_ISR_ATTR onSampleTimer() {
  BaseType_t higherPriorityWoken = pdFALSE;
  vTaskNotifyGiveFromISR(samplerTask, &higherPriorityWoken);
  if (higherPriorityWoken) portYIELD_FROM_ISR();
}

// ---- Sampler task: one frame per timer tick, one packet per 20 frames ------
static void samplerLoop(void *) {
  Packet packet;
  uint32_t seq = 0;
  uint16_t frame = 0;
  uint64_t firstSampleUs = 0;
#if OVERSAMPLE == 2
  static mb::HalfbandDecimator<NUM_CHANNELS> decimator;
  int16_t pending[NUM_CHANNELS];
  bool havePending = false;
#endif

  for (;;) {
    // Sleep until the ISR "gives". The return value counts gives since we
    // last woke: more than 1 means we missed ticks (should never happen).
    uint32_t ticks = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (ticks > 1) missedTicks += ticks - 1;
    uint64_t nowUs = (uint64_t)esp_timer_get_time();

    int16_t raw[NUM_CHANNELS];
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
      // Centre around zero so the samples are signed, like the EMG itself.
      raw[ch] = (int16_t)(analogRead(CHANNEL_PINS[ch]) - ADC_MIDPOINT);
    }

    int16_t out[NUM_CHANNELS];
#if OVERSAMPLE == 2
    if (!havePending) {  // first of a pair: wait for the second
      for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) pending[ch] = raw[ch];
      havePending = true;
      continue;
    }
    havePending = false;
    decimator.push(pending, raw, out);
#else
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) out[ch] = raw[ch];
#endif

    // With OVERSAMPLE 2 the timestamp is when the newest input sample of this
    // output was read; the filter's delay is 23 input samples (11.5 ms).
    if (frame == 0) firstSampleUs = nowUs;
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
      mb::putSample(packet.bytes, NUM_CHANNELS, frame, ch, out[ch]);
    }

    if (++frame == FRAMES_PER_PACKET) {
      mb::finishPacket(packet.bytes, NUM_CHANNELS, SAMPLE_RATE_HZ, seq, firstSampleUs,
                       FRAMES_PER_PACKET);
      // Don't wait if the queue is full: dropping a packet is better than
      // stalling the sampler. seq still advances, so the PC sees the gap.
      if (xQueueSend(packetQueue, &packet, 0) != pdTRUE) queueFull++;
      seq++;
      frame = 0;
    }
  }
}

// ---- WiFi ------------------------------------------------------------------
static void connectWiFi() {
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // modem sleep adds 100+ ms latency spikes
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.printf("\nConnected, IP %s -> sending to %s:%d\n",
                WiFi.localIP().toString().c_str(), HOST_IP, HOST_PORT);
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);        // 0..4095
  analogSetAttenuation(ADC_11db);  // full ~0-3.3 V input range
  connectWiFi();

  packetQueue = xQueueCreate(16, sizeof(Packet));  // ~320 ms of slack

  // Higher priority than loop() so sampling always wins over sending.
  // Pinned to core 1; the WiFi stack lives on core 0.
  xTaskCreatePinnedToCore(samplerLoop, "sampler", 4096, nullptr,
                          configMAX_PRIORITIES - 2, &samplerTask, 1);

  // Core 3.x timer API: 1 MHz tick (1 us resolution), alarm every
  // 1e6 / ADC_RATE_HZ ticks, auto-reload forever (reload_count 0).
  sampleTimer = timerBegin(1000000);
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, 1000000 / ADC_RATE_HZ, true, 0);
}

void loop() {
  static uint32_t lastReportMs = 0;
  Packet packet;

  // Wait (up to 100 ms) for the sampler to hand over a finished packet.
  if (xQueueReceive(packetQueue, &packet, pdMS_TO_TICKS(100)) == pdTRUE) {
    udp.beginPacket(HOST_IP, HOST_PORT);
    udp.write(packet.bytes, PACKET_SIZE);
    if (!udp.endPacket()) sendFailed++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();  // packets pile up (then drop) meanwhile; the PC sees the gap
  }

  if (millis() - lastReportMs > 5000) {
    lastReportMs = millis();
    Serial.printf("missed ticks %lu, queue-full drops %lu, send failures %lu\n",
                  (unsigned long)missedTicks, (unsigned long)queueFull,
                  (unsigned long)sendFailed);
  }
}
