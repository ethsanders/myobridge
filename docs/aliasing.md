# Aliasing at 1 kHz behind a MyoWare

Sampling at 1 kHz behind a MyoWare is enough for the EMG signal itself. It
is marginal for the noise floor. The firmware has an optional fix that,
according to the noise model below, removes about three quarters of the
folded noise power.

The numbers on this page are in `results/aliasing.json`, which
`myobridge bench aliasing` writes.

## Background

Sampling at `fs` can only represent frequencies up to `fs/2`. A component
above that which reaches the ADC does not disappear. A component at `f`
appears at `fs - f`, `f - fs`, and so on; this is called folding. A
standard design puts a steep analog low-pass (an anti-aliasing filter) in
front of the ADC.

The MyoWare 2.0's RAW output is band-passed at about 20-498 Hz, but its upper
edge is a gentle analog roll-off rather than a steep filter. At 1 kHz
sampling the fold-over point is 500 Hz, where that roll-off starts. This
page asks what is present above 500 Hz and how much of it folds into the
20-450 Hz band that the classifier uses.

Two sources contribute above 500 Hz, the EMG itself and wideband electronic
noise, and they behave differently.

## 1. Aliasing of the EMG signal

The MultiDay recordings were made at 2048 Hz behind a proper anti-aliasing
filter, so they show the spectrum of real forearm EMG up to 1 kHz.

| | Median share of power above 512 Hz, per channel |
|---|---|
| Gestures (1010 recordings) | 0.04% to 0.10% |
| Rest (101 recordings) | 0.11% to 0.17% |

The share is under a fifth of a percent on every channel. Skin and fat
between muscle and electrode act as a low-pass filter, so surface EMG has
very little power above 512 Hz. Rest has a *larger* share only because it
has less EMG power below 512 Hz to compare against.

To check this end to end, the bench converts every MultiDay recording to
1024 Hz in four ways and re-runs the pipeline (filters, features, LDA) on
days 31-121:

| How 2048 Hz becomes 1024 Hz | Same-day accuracy | Next-day accuracy |
|---|---|---|
| `ideal`: steep anti-alias FIR, then drop every 2nd sample | 88.1% | 86.6% |
| `none`: just drop every 2nd sample, no filter at all | 88.1% | 86.5% |
| `rc`: one-pole low-pass at 498 Hz (MyoWare-like), then drop | 88.1% | 86.4% |
| `rc_ideal`: the same one-pole, then the steep FIR | 88.0% | 86.4% |

Accuracy is unchanged even with no filter. The signal agrees: inside
20-450 Hz, the difference between `none` and `ideal` holds about 0.01% of
the signal's energy (-40 dB). The one-pole filter changes about 1% of the
in-band energy. Because `rc` and `rc_ideal` match, that change is the
filter's droop and phase lag, not aliasing. Accuracy moves by less than 0.2
points either way.

## 2. Aliasing of wideband noise

MultiDay cannot show noise above 1 kHz, because its recorder's own
anti-alias filter removed it. A MyoWare plus an ESP32 has no such filter,
and amplifier and ADC noise is roughly white out to tens of kHz. Behind a
first-order roll-off, much of that noise survives above 500 Hz and folds
into the band.

`myobridge.aliasing.folded_noise_ratio` computes this for white noise through
an n-th order low-pass at 498 Hz. It sums the folded copies from 5000
multiples of `fs`; the tail beyond them is negligible.
The result is the increase in the in-band noise floor:

| Analog filter | ADC at 1 kHz | ADC at 2 kHz | ADC at 4 kHz |
|---|---|---|---|
| 1st order (MyoWare-like) | **+2.9 dB** | +1.0 dB | +0.3 dB |
| 2nd order | +0.9 dB | +0.05 dB | +0.003 dB |
| 4th order | +0.3 dB | ~0 dB | ~0 dB |

For a first-order filter at 1 kHz, the folded noise has almost the same
power as the original in-band noise, so the noise floor nearly doubles.
This makes weak contractions harder to distinguish from rest, and it adds
variance to the features that count zero crossings and slope changes.

The 2 kHz and 4 kHz columns assume a perfect digital filter before
decimating to 1 kHz. The next section covers the firmware's actual filter.

## Firmware option: `OVERSAMPLE 2`

With `#define OVERSAMPLE 2` in `config.h`, the ESP32 samples at 2 kHz, runs
a 47-tap halfband FIR, and keeps every second output. The PC still receives
1 kHz data, and nothing downstream changes.

- **Halfband**: every second coefficient is exactly zero and the centre is
  exactly 1/2, so the filter costs 24 multiplies per output sample, in
  integer arithmetic (`firmware/esp32_emg_streamer/mb_decimator.h`).
- **Response** (tested in `tests/test_decimator.py`): flat to 0.1 dB from 20
  to 400 Hz, -0.7 dB at 450 Hz, and at least 40 dB down above 600 Hz. That
  stopband covers the range that would otherwise fold onto 0-400 Hz.
- **Bit-exact**: `myobridge.decimator` is a Python copy with the same Q15
  coefficients, rounding and saturation. A test compiles the C++ header with
  g++ and checks that both give identical int16 output on the same input.
- **Delay**: 23 input samples, 11.5 ms.

`folded_noise_ratio_decimated` repeats the noise calculation with this
filter in place of a perfect one, including its droop and finite stopband.
For a first-order front end it gives +0.97 dB, within 0.01 dB of the ideal
2 kHz case. The remaining folded noise is therefore set by the 2 kHz rate
and not by the filter.

## Limitations

- **Filter order.** I modelled the MyoWare's upper edge as first order from
  its published 20-498 Hz band. If the edge is steeper, the problem is
  smaller (+0.9 dB at second order).
- **Noise bandwidth.** Real noise is not white at all frequencies. Op-amp
  bandwidth and the ADC's own input bandwidth roll it off somewhere, so the
  model is an upper bound for a given filter order.
- **Hardware.** `OVERSAMPLE 2` has not run on hardware. It compiles in CI
  and the arithmetic is tested bit for bit, but the ESP32 has to read 4
  channels every 500 µs. [hardware-bringup.md](hardware-bringup.md)
  describes the plan for checking it, which records the rest-noise floor
  both ways.

## Reproduce

```bash
myobridge fetch multiday
myobridge bench aliasing        # ~20 s after the first load
pytest tests/test_aliasing.py tests/test_decimator.py tests/test_firmware.py
```
