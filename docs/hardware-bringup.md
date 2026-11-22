# Hardware bring-up

This is a checklist for the first sessions on real hardware. Each step says
what to measure, what the result should look like, and what it means if the
result is different. Up to this point the code was tested on public datasets
and the simulator; these steps check the hardware itself.

Write the results, with dates, into the table at the bottom of this file as
you go. A dated measurement on this hardware is stronger evidence than any
of the model numbers.

## 0. Safety

- Power the ESP32 and sensors **only from a battery** (LiPo or USB power
  bank) whenever electrodes touch skin.
- Flash the board, **unplug the USB cable**, then put electrodes on. Data goes
  over WiFi, so nothing else needs to be connected.
- Never wear electrodes while the circuit touches anything mains-powered,
  including a laptop that is charging. Don't place electrodes across the
  chest.

## 1. Flash and check the link (no electrodes)

```bash
cp firmware/esp32_emg_streamer/config.example.h firmware/esp32_emg_streamer/config.h
# set WIFI_SSID, WIFI_PASSWORD, HOST_IP (your PC), keep OVERSAMPLE 1
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/esp32_emg_streamer
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/esp32_emg_streamer
myobridge listen
```

**Expect:** packets arriving, loss under 1%, `reordered` and `restarts` at 0.
The serial monitor (115200 baud, while still on USB) prints
`missed ticks 0, queue-full drops 0` every 5 s.

**If not:** `missed ticks` above 0 means the sampler can't keep up, which
should not happen at 4 channels and 1 kHz. `queue-full drops` mean WiFi
stalled for more than ~320 ms. If loss is high on a busy network, move
closer to the router or use a phone hotspot.

## 2. The ADC on its own

Wire each ADC pin (GPIO 32-35) to a fixed mid-supply voltage instead of a
sensor: two equal resistors (say 10 kΩ) from 3V3 and from GND to the pin,
with a 100 nF capacitor from the pin to GND. Then:

```bash
myobridge record --seconds 10 --out adc_os1.npz
# set OVERSAMPLE 2 in config.h, flash again, then:
myobridge record --seconds 10 --out adc_os2.npz
```

`record` prints each channel's 20-450 Hz RMS in ADC counts.

**Expect:** a small RMS (a few counts) that is lower with `OVERSAMPLE 2`.
The ADC's own noise is added after sampling, so at 2 kHz the same noise is
spread over twice the bandwidth and the halfband filter removes half of it.
That gives up to about 3 dB less, or RMS times 0.71. This is the baseline
for the next step.

## 3. MyoWare at rest

Put the sensors back on, with electrodes on a relaxed forearm and the arm
resting on a table.

```bash
myobridge record --seconds 30 --out rest_os1.npz   # OVERSAMPLE 1
myobridge record --seconds 30 --out rest_os2.npz   # OVERSAMPLE 2
```

**What the model predicts** ([aliasing.md](aliasing.md)): if the MyoWare's
upper roll-off is first order and its wideband noise dominates the rest
signal, the folded noise floor drops from +2.9 dB to +1.0 dB. The rest RMS
should then fall by about 20% with `OVERSAMPLE 2`, *beyond* what step 2
showed for the ADC alone.

- **About 20% lower, more than the ADC alone:** the aliasing model holds.
  Keep `OVERSAMPLE 2`.
- **Same drop as the ADC alone:** the MyoWare's roll-off is steeper than first
  order, or its noise isn't white that far up. `OVERSAMPLE 2` still reduces
  the ADC noise. Note the measured numbers in aliasing.md.
- **No change:** something other than broadband noise sets the floor, such
  as mains hum or a loose electrode. Look at `mains share` in the printout
  and at step 5.

Also check the timing. `missed ticks` must stay at 0 with `OVERSAMPLE 2`,
because the ESP32 now reads 4 channels every 500 µs.

## 4. Check the MyoWare's filter directly (optional)

The model assumes a first-order upper edge at 498 Hz. To measure it, feed a
sine from a signal generator (or a second ESP32's DAC) through a resistor
divider into the MyoWare's electrode inputs at a level that doesn't clip,
and record RAW with an oscilloscope or an ADC sampling well above 2 kHz. The
gain should fall 3 dB at the corner. Beyond it, a first-order filter falls a
further 6 dB per octave (at 1 and 2 kHz), and a second-order one 12 dB. Put
the measured slope into `myobridge.aliasing.folded_noise_ratio` to get the
folded noise figure for this hardware.

## 5. Signal quality on a person

```bash
myobridge --mains 50 listen     # in Europe; 60 elsewhere
```

Relax for the first 3 s, which sets each channel's noise floor, then make a
fist and open the hand. Each channel line ends with `ok` or `CHECK: ...`
([signal-quality.md](signal-quality.md)):

| Message | Usual cause | Fix |
|---|---|---|
| `flatline` | Electrode off or lead unplugged | Press the electrodes on, check the snap cable |
| `clipping` | Gain too high, or the signal hits the ADC rails | Turn the MyoWare gain down |
| `line_noise` | Mains hum: long leads, a charger nearby, poor reference contact | Battery only, reference electrode on bone, shorter leads |
| `motion` | Cable or skin moving | Tape the cables down |

**Expect:** every channel `ok` at rest and during contractions.

## 6. First gestures

```bash
myobridge calibrate --seconds 5 --out model.npz
myobridge forward --model model.npz --verbose
```

**Expect:** with 4 sensors placed on different muscles, 5 gestures should
mostly come out right. On the public 3DC data, 4 well-spread channels give
about 81% for 11 gestures within one sitting ([results.md](results.md)),
and fewer gestures are easier to separate.

## 7. The next day

Put the electrodes back on as close to the same spots as you can (mark the
skin), and run `forward` with yesterday's model before recalibrating. The
MultiDay results predict a drop of a few points after one day and much more
after weeks, and a short recalibration should recover it. This experiment
checks the public-data results on my own hardware.
[The first recorded session](first-session.md) is the full protocol for
these days, and `myobridge report` turns them into a report.

## Results log

| Date | Step | OVERSAMPLE | Channel RMS (counts) | Notes |
|---|---|---|---|---|
| | | | | |
