# Aliasing: is 1 kHz behind a MyoWare enough?

Short answer: for the EMG itself, yes. For the noise floor, not quite, and
the firmware has an optional fix that the model says removes about three quarters
of the folded noise power.

Every number here is in `results/aliasing.json`, made by
`myobridge bench aliasing`.

## The problem

Sampling at `fs` can only represent frequencies up to `fs/2`. Anything above
that which reaches the ADC doesn't disappear: a component at `f` shows up
at `fs - f`, `f - fs`, and so on ("folds back"). A proper system puts a
steep analog low-pass (an anti-aliasing filter) in front of the ADC.

The MyoWare 2.0's RAW output is band-passed at about 20-498 Hz, but its upper
edge is a gentle analog roll-off, not a steep filter. At 1 kHz sampling the
fold-over point is 500 Hz, right where that roll-off starts. So the question
is what lives above 500 Hz and how much of it folds into the 20-450 Hz band
the classifier uses.

There are two separate things up there, and they behave differently.

## 1. The EMG itself barely aliases

The MultiDay recordings were made at 2048 Hz behind a proper anti-aliasing
filter, so they show what real forearm EMG looks like up to 1 kHz.

| | Median share of power above 512 Hz, per channel |
|---|---|
| Gestures (1010 recordings) | 0.04% to 0.10% |
| Rest (101 recordings) | 0.11% to 0.17% |

Under a fifth of a percent. Skin and fat between muscle and electrode act as
a low-pass filter, so surface EMG has almost nothing up there. (Rest has a
*larger* share only because there is less EMG below 512 Hz to compare to.)

To check this end to end, the bench turns every MultiDay recording into a
1024 Hz one in four ways and re-runs the whole pipeline (filters, features,
LDA) on days 31-121:

| How 2048 Hz becomes 1024 Hz | Same-day accuracy | Next-day accuracy |
|---|---|---|
| `ideal`: steep anti-alias FIR, then drop every 2nd sample | 86.8% | 83.7% |
| `none`: just drop every 2nd sample, no filter at all | 86.8% | 83.7% |
| `rc`: one-pole low-pass at 498 Hz (MyoWare-like), then drop | 86.5% | 83.6% |
| `rc_ideal`: the same one-pole, then the steep FIR | 86.5% | 83.5% |

Even with no filter at all, accuracy doesn't move. The signal itself
confirms it: the difference between `none` and `ideal` inside 20-450 Hz holds
about 0.01% of the signal's energy (-40 dB). The one-pole filter costs a few
tenths of a point, and since `rc` and `rc_ideal` match, that is its in-band
droop and phase lag, not aliasing.

## 2. Wideband noise does alias

What MultiDay can't show is noise *above* 1 kHz: its recorder's own
anti-alias filter already removed it. A MyoWare plus an ESP32 has no such
filter, and amplifier and ADC noise is roughly white out to tens of kHz.
Behind a first-order roll-off, a lot of it survives above 500 Hz and folds
into the band.

`myobridge.aliasing.folded_noise_ratio` computes this for white noise through
an n-th order low-pass at 498 Hz, summing the folded copies from every
multiple of `fs` (5000 of them; the tail is negligible). The result is how
much the in-band noise floor grows:

| Analog filter | ADC at 1 kHz | ADC at 2 kHz | ADC at 4 kHz |
|---|---|---|---|
| 1st order (MyoWare-like) | **+2.9 dB** | +1.0 dB | +0.3 dB |
| 2nd order | +0.9 dB | +0.05 dB | +0.003 dB |
| 4th order | +0.3 dB | ~0 dB | ~0 dB |

+2.9 dB means the folded noise has almost the same power as the noise that
was there to begin with: the noise floor nearly doubles. That makes weak
contractions harder to tell from rest and adds variance to the features that
count zero crossings and slope changes.

The 2 kHz and 4 kHz columns assume a perfect digital filter before dropping
back to 1 kHz. The firmware's real one is next.

## The fix in the firmware: `OVERSAMPLE 2`

With `#define OVERSAMPLE 2` in `config.h`, the ESP32 samples at 2 kHz, runs
a 47-tap halfband FIR, and keeps every second output, so the PC still
receives 1 kHz and nothing else changes.

- **Halfband**: every second coefficient is exactly zero and the centre is
  exactly 1/2, so it costs 24 multiplies per output sample, in integer maths
  (`firmware/esp32_emg_streamer/mb_decimator.h`).
- **Response** (tested in `tests/test_decimator.py`): flat to 0.1 dB from 20
  to 400 Hz, -0.7 dB at 450 Hz, and at least 40 dB down above 600 Hz, which
  is what would otherwise fold onto 0-400 Hz.
- **Bit-exact**: `myobridge.decimator` is a Python copy with the same Q15
  coefficients, rounding and saturation. A test compiles the C++ header with
  g++ and checks both give identical int16 output on the same input.
- **Delay**: 23 input samples, 11.5 ms.

`folded_noise_ratio_decimated` repeats the noise calculation with this exact
filter instead of a perfect one, including its droop and finite stopband.
It gives **+0.97 dB** for a first-order front end, the same as the ideal
2 kHz case to within 0.01 dB. So the filter is not the limit; the 2 kHz rate
is.

## What I haven't checked

- **The real filter order.** I've modelled the MyoWare's upper edge as first
  order from its published 20-498 Hz band. If it is steeper, the problem is
  smaller (+0.9 dB at second order).
- **Real noise isn't white forever.** Op-amp bandwidth and the ADC's own
  input bandwidth roll the noise off somewhere, which makes the model an
  upper bound for a given filter order.
- **`OVERSAMPLE 2` hasn't run on hardware.** It compiles in CI and the maths
  is tested bit for bit, but the ESP32 has to read 4 channels every 500 µs.
  The plan for checking it, by recording the rest-noise floor both ways, is
  in [HARDWARE_BRINGUP.md](HARDWARE_BRINGUP.md).

## Reproduce

```bash
myobridge fetch multiday
myobridge bench aliasing        # ~20 s after the first load
pytest tests/test_aliasing.py tests/test_decimator.py tests/test_firmware.py
```
