# Build the armband

The armband is four MyoWare muscle sensors read by an ESP32, which streams
the samples over WiFi. Everything here is off the shelf.

!!! danger "Battery power only"
    Power the ESP32 and the sensors **only from a battery** (a LiPo or a USB
    power bank) whenever electrodes are on skin. Flash the board, unplug the
    USB cable, then put the electrodes on. Never wear electrodes while the
    circuit touches anything mains-powered, including a charging laptop.
    Don't place electrodes across the chest. This is a hobby project, not a
    medical device.

## Parts

| Part | Qty | Notes |
|---|---|---|
| ESP32 dev board (WROOM-32) | 1 | Any board with GPIO 32-35 broken out |
| MyoWare 2 muscle sensor (SparkFun DEV-27924) | 4 | It replaced the MyoWare 2.0 (DEV-21265), which works the same here. About $43 each at SparkFun in November 2026 |
| Ag/AgCl snap electrodes | 12 per session | Three per sensor; they are single-use |
| LiPo or USB power bank | 1 | See the warning above |
| Jumper wires | | |

## Wiring

| MyoWare | ESP32 | Why |
|---|---|---|
| `+` | `3V3` | At 3.3 V the sensor's output can't exceed the ADC's input range |
| `-` | `GND` | |
| `RAW` | GPIO 32, 33, 34, 35 (channels 0-3) | These are on ADC1, which still works while WiFi is on |

Use the **RAW** output, not ENV. ENV is the sensor's own rectified and
smoothed envelope: fine for "is the muscle on?", but it throws away the
frequency content the features use, and the PC can't undo it. RAW is the
amplified, band-passed signal, centred at half the supply voltage. Check the
pin labels on your board; the output modes are printed next to the pads.

## Placement

Put the four sensors around the thickest part of the forearm, roughly evenly
spaced, each with its two electrodes along the muscle and the reference
electrode on a bony spot such as the elbow. The benchmarks use the same
"four spread evenly" layout on the 3DC armband; on 11 gestures it costs about
10 points of accuracy against 10 electrodes (see [Results](results.md)).

## Flash the firmware

```bash
cp firmware/esp32_emg_streamer/config.example.h firmware/esp32_emg_streamer/config.h
# edit config.h: WiFi name and password, your computer's IP, OVERSAMPLE 1 or 2
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/esp32_emg_streamer
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/esp32_emg_streamer
```

The sketch targets Arduino-ESP32 core 3.x. `config.h` is gitignored, so
your WiFi password stays on your machine. `OVERSAMPLE 2` samples at 2 kHz
and filters down to 1 kHz on the ESP32, which should lower the aliased noise
floor ([Does 1 kHz alias?](aliasing.md)); the computer sees 1 kHz either way.

## Check it works

Unplug USB, power the board from the battery, and run:

```bash
myobridge listen --port 5005
```

You should see four RMS bars that jump when you clench, a healthy status
for each channel, and packet loss near zero.
[First sessions on hardware](hardware-bringup.md) is the full checklist,
including how to measure the noise floor and set the quality thresholds for
your sensors.
