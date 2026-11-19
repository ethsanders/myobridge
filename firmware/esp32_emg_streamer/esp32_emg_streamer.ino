// esp32_emg_streamer: sample N analog EMG channels at 1 kHz and stream them
// over WiFi UDP using the myobridge wire protocol (docs/PROTOCOL.md).
//
// Target: any ESP32 dev board (original ESP32 / WROOM-32), Arduino-ESP32 core 3.x.
// Note: core 3.x changed the timer API. timerBegin() now takes a tick frequency
// in Hz (it used to take num, divider, countUp), and timerAlarmWrite/Enable were
// replaced by a single timerAlarm(). This sketch uses the 3.x calls only.
//
// ---------------------------------------------------------------------------
// WIRING (MyoWare 2.0 muscle sensor, one per channel)
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

// ---- Acquisition settings (keep in sync with what the PC expects) ----------
static const uint8_t NUM_CHANNELS = 4;
static const uint8_t CHANNEL_PINS[NUM_CHANNELS] = {32, 33, 34, 35};
static const uint16_t SAMPLE_RATE_HZ = 1000;
static const uint16_t FRAMES_PER_PACKET = 20;  // 20 ms of data per packet
static const int ADC_MIDPOINT = 2048;           // 12-bit ADC, RAW idles at Vcc/2

// ---- Wire protocol (see docs/PROTOCOL.md) ----------------------------------
static const uint8_t MAGIC_0 = 'M';
static const uint8_t MAGIC_1 = 'B';
static const uint8_t PROTOCOL_VERSION = 1;
static const size_t HEADER_SIZE = 20;
static const size_t PAYLOAD_SIZE = (size_t)NUM_CHANNELS * FRAMES_PER_PACKET * 2;
static const size_t PACKET_SIZE = HEADER_SIZE + PAYLOAD_SIZE + 2;  // +2 = CRC

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

// ---- Little-endian helpers (write the low byte first) ----------------------
static void putU16(uint8_t *buf, size_t at, uint16_t v) {
  buf[at] = v & 0xFF;
  buf[at + 1] = v >> 8;
}

static void putU32(uint8_t *buf, size_t at, uint32_t v) {
  for (int i = 0; i < 4; i++) buf[at + i] = (v >> (8 * i)) & 0xFF;
}

static void putU64(uint8_t *buf, size_t at, uint64_t v) {
  for (int i = 0; i < 8; i++) buf[at + i] = (v >> (8 * i)) & 0xFF;
}

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, MSB first, no final XOR.
// Must match protocol.crc16_ccitt_false() in Python ("123456789" -> 0x29B1).
static uint16_t crc16CcittFalse(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void writeHeader(uint8_t *buf, uint32_t seq, uint64_t timestampUs) {
  buf[0] = MAGIC_0;
  buf[1] = MAGIC_1;
  buf[2] = PROTOCOL_VERSION;
  buf[3] = NUM_CHANNELS;
  putU16(buf, 4, SAMPLE_RATE_HZ);
  putU32(buf, 6, seq);
  putU64(buf, 10, timestampUs);
  putU16(buf, 18, FRAMES_PER_PACKET);
}

// ---- Timer interrupt: runs every 1 ms --------------------------------------
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

  for (;;) {
    // Sleep until the ISR "gives". The return value counts gives since we
    // last woke: more than 1 means we missed ticks (should never happen).
    uint32_t ticks = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (ticks > 1) missedTicks += ticks - 1;

    if (frame == 0) firstSampleUs = (uint64_t)esp_timer_get_time();

    size_t at = HEADER_SIZE + (size_t)frame * NUM_CHANNELS * 2;
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
      // Centre around zero so the samples are signed, like the EMG itself.
      int16_t sample = (int16_t)(analogRead(CHANNEL_PINS[ch]) - ADC_MIDPOINT);
      putU16(packet.bytes, at, (uint16_t)sample);
      at += 2;
    }

    if (++frame == FRAMES_PER_PACKET) {
      writeHeader(packet.bytes, seq, firstSampleUs);
      uint16_t crc = crc16CcittFalse(packet.bytes, PACKET_SIZE - 2);
      putU16(packet.bytes, PACKET_SIZE - 2, crc);
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
  // 1e6 / SAMPLE_RATE_HZ ticks, auto-reload forever (reload_count 0).
  sampleTimer = timerBegin(1000000);
  timerAttachInterrupt(sampleTimer, &onSampleTimer);
  timerAlarm(sampleTimer, 1000000 / SAMPLE_RATE_HZ, true, 0);
}

void loop() {
  static uint32_t lastReportMs = 0;
  Packet packet;

  // Wait (up to 100 ms) for the sampler to hand over a finished packet.
  if (xQueueReceive(packetQueue, &packet, pdMS_TO_TICKS(100)) == pdTRUE) {
    udp.beginPacket(HOST_IP, HOST_PORT);
    udp.write(packet.bytes, PACKET_SIZE);
    udp.endPacket();
  }

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();  // packets pile up (then drop) meanwhile; the PC sees the gap
  }

  if (millis() - lastReportMs > 5000) {
    lastReportMs = millis();
    Serial.printf("missed ticks %lu, queue-full drops %lu\n",
                  (unsigned long)missedTicks, (unsigned long)queueFull);
  }
}
