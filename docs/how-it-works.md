# How myobridge works

This page describes what each part of myobridge does and why it is designed
that way. Measured results on public datasets are in [results.md](results.md).

## Overview

When a muscle contracts, many motor units fire. Each firing sends a small
electrical wave along the muscle fibres, and electrodes on the skin record the
sum of these waves. The result is **surface EMG**, a noisy, zero-mean signal
of a few hundred microvolts to a few millivolts. Its *amplitude* rises with
effort, and its *spectrum* lies roughly between 20 and 450 Hz.

The MyoWare 2.0 amplifies that to volts. The ESP32 samples it, packs it into
UDP packets and sends it over WiFi. The PC cleans the signal, summarises each
200 ms window as 60 numbers (features), and a linear classifier maps those
numbers to a gesture. The gesture goes out as JSON to a game engine.

```
skin -> MyoWare (amp) -> ESP32 ADC (1 kHz) -> UDP -> receiver -> bandpass -> notch
     -> 200 ms windows every 50 ms -> features -> LDA -> majority vote -> JSON -> Unity
```

## Modules

| File | Job | Why it's built like this |
|---|---|---|
| `firmware/.../esp32_emg_streamer.ino` | Hardware timer fires every 1 ms; the ISR only wakes a sampler task, which reads 4 ADC pins. Every 20 frames it builds a packet and hands it to `loop()` through a FreeRTOS queue; `loop()` sends it over UDP. | ISRs must be short, and WiFi sends can block for milliseconds. Putting a queue between sampling and sending means slow WiFi never causes a missed sample. If the queue is full the packet is dropped but `seq` is still incremented, so the PC can count the loss. |
| `protocol.py` | Encode/decode packets, CRC, `SequenceTracker`. | The byte layout is defined in one place. The tracker turns sequence numbers into drop, reorder and restart counts. |
| `receiver.py` | Blocking UDP socket with a 0.5 s timeout, yields packets. | It uses no threads or asyncio, which keeps it simple to reason about. The timeout keeps Ctrl-C and the display responsive. |
| `simulator.py` | Fake armband producing the same int16 packets. | Lets the whole pipeline run and be tested without hardware; deterministic with a seed. |
| `dsp.py` | Streaming bandpass, notch, rectify, RMS envelope. | Filters keep state between packets, so chunked output equals one-shot output. |
| `features.py` | 15 features per channel: the Hudgins time-domain set, autoregressive coefficients and spectral moments. | These features are cheap and well studied, and the set was chosen on validation people and days (see below). |
| `classifier.py` | LDA, calibration, majority vote. | Fitting is closed-form and needs little data; prediction is fast. |
| `pipeline.py` | Ties filter + windower + features + LDA + vote together. | The demo, `forward` and the plot script use the same code path. |
| `quality.py` | Per-channel health checks: flatline, clipping, mains hum, motion. | A dead or saturated electrode degrades a classifier without any visible error. Flagging it lets the decoder fall back to the healthy channels ([signal-quality.md](signal-quality.md)). |
| `decimator.py` + `firmware/.../mb_decimator.h` | Optional 2x oversampling with a 47-tap integer halfband filter. | Lowers the aliased noise floor. The Python copy is the bit-exact reference the C++ is tested against ([aliasing.md](aliasing.md)). |
| `aliasing.py` | Model of how much wideband noise folds into the band. | Gives a numerical answer to whether 1 kHz is a high enough sample rate. |
| `datasets/`, `evaluation.py`, `bench/` | Download public datasets at pinned commits, split them into training and test sets, run the experiments. | Every number in the README comes from `myobridge bench` ([datasets.md](datasets.md)). |
| `cli.py` | `simulate`, `listen`, `calibrate`, `forward`, `demo`, `fetch`, `bench`. | Thin wrappers around the modules above. |

## Signal chain

**Sampling.** By the Nyquist criterion, capturing frequencies up to *f*
requires sampling faster than 2*f*. Surface EMG has useful content up to
~450-500 Hz, so 1 kHz is the minimum sensible rate (research systems often use
2 kHz). Conversely, anything above 500 Hz that reaches the ADC folds back into
0-500 Hz. The MyoWare's analog filter rolls off only gently there, so it does
not act as a true anti-aliasing filter. EMG itself has almost no power above
500 Hz (about 0.1% in the 2 kHz MultiDay recordings), so the gesture signal is
largely unaffected, but amplifier noise folds in and raises the noise floor.
[aliasing.md](aliasing.md) quantifies this and describes the firmware's
remedy, which samples at 2 kHz and filters digitally down to 1 kHz
(`OVERSAMPLE 2`). The ESP32's ADC is 12-bit (0-4095); subtracting 2048 makes
the samples signed.

**Bandpass 20-450 Hz (4th-order Butterworth, so 8th order overall).** Content
below 20 Hz is mostly *motion artefact* (cables and skin moving) and *baseline
drift* (the electrode-skin potential changing slowly), not muscle activity.
Above 450 Hz little EMG power remains, and the signal is mostly amplifier and
ADC noise. 450 Hz is also 0.9× the 500 Hz Nyquist limit, about as high as the
cutoff can go. A Butterworth filter has a maximally flat passband, so
amplitudes inside the band are not distorted. The filter is implemented as
second-order sections (biquads) for numerical stability.

**Notch at 60 Hz (Q = 30, ~2 Hz wide).** Mains wiring radiates 60 Hz (50 Hz
in Europe), and the body acts as an antenna. The hum falls inside the EMG
band, where the bandpass cannot remove it. A narrow notch removes it with
little effect on neighbouring frequencies. The tests check for >20 dB
attenuation at 60 Hz and <0.5 dB change at 30/100/150/250/400 Hz.

**Rectify + RMS envelope (100 ms).** Because EMG is zero-mean, its average
carries no information about effort; its magnitude does. Full-wave
rectification (absolute value) followed by a moving RMS gives a smooth
estimate of how hard the muscle is working. The `listen` bars display this
envelope.

**Windows.** Each window is 200 ms long, which is long enough for stable
features and short enough for responsive control. Windows overlap, so a new
decision is made every 50 ms.

**Streaming filter state.** An IIR filter's output depends on its past inputs
and outputs. Filtering each 20-sample packet from a zero state would produce a
startup transient every 20 ms. `scipy.signal.sosfilt(..., zi=state)` returns
the final state, which is passed into the next call. A test filters the same
signal in blocks of 1, 7, 20 and 333 samples and checks that the result
matches the one-shot result to 1e-9.

## Features (per channel, per window)

| Feature | Formula | Captures |
|---|---|---|
| MAV | mean(\|x\|) | amplitude |
| RMS | sqrt(mean(x²)) | amplitude / power |
| Waveform length | Σ\|x[i+1] − x[i]\| | amplitude × frequency |
| Zero crossings | sign changes with a jump ≥ threshold | frequency |
| Slope sign changes | peaks/valleys above a threshold | frequency |

The amplitude features are `log1p`-scaled because EMG amplitude spans orders
of magnitude between rest and a full fist. On a log scale the class clusters
have similar spread, which suits LDA's shared-covariance assumption.

These five features form the `td` feature set. The default set, `td-ar-psd`,
adds two groups that describe the *shape* of the signal rather than its size:

| Group | Features | Captures |
|---|---|---|
| Autoregressive (AR) | 4 coefficients of a 4th-order AR model, fitted by Yule-Walker | how each sample depends on the last few: the spectrum's shape, independent of amplitude |
| TD-PSD (Al-Timemy et al., 2016) | 6 values built from the zeroth, second and fourth spectral moments, computed in the time domain | overall power, sharpness of the spectrum, irregularity, and how those change with the waveform's derivative |

4 channels × 15 features = 60 numbers per window. The TD-PSD code follows
LibEMG's implementation, and a test checks it against a transcription of
theirs. The AR coefficients do not change when the signal's gain changes,
which should help when an electrode's contact changes between days; the
effect is measured in [results.md](results.md). `feature_set="td"` selects the
old five features per channel.

## Linear discriminant analysis

LDA assumes that each gesture's feature vectors form a Gaussian cloud with its
own centre (mean) and a common shape (one covariance matrix shared by all
classes). Fitting computes each class mean and the pooled covariance, with no
iterative optimisation. Because the covariance is shared, comparing class
log-likelihoods reduces to a linear function of the features:
`score = x·W + b`. Prediction is one matrix multiply (60 features × the number
of gestures) plus a softmax for probabilities. A small ridge is added to the
covariance so that it is always invertible, for example when a channel is
dead. Each feature gets 0.1% of its own variance, so the result does not
depend on the features' units.

Because the model consists only of class means and a covariance, dropping a
channel is exact and immediate. `LDAClassifier.subset` keeps the rows and
columns that belong to the remaining channels' features. The pipeline uses
this to continue when the quality monitor flags an electrode, without keeping
the training data. The method is Zhang and Huang's sensor-fault-tolerant LDA
(*Journal of NeuroEngineering and Rehabilitation* 12:18, 2015).

**Majority vote (k = 5).** The output is the most common label of the last 5
windows (ties go to the newest). This removes single-window flickers at the
cost of ~100 ms extra lag when the gesture changes. The lag is why smoothed
accuracy in the demo is slightly *lower* than raw accuracy: it costs a few
windows at every switch, but the output a user sees is far steadier.

## Loss detection

Every packet carries a 32-bit counter. The receiver stores the next number it
expects and computes `ahead = (seq − expected) mod 2³²`:

- `0`: on time.
- small positive: a gap; that many packets were lost.
- "negative" (≥ 2³¹ after the modulo) by a little (up to 32): a late packet.
  It was already counted as dropped, so the tracker reverses that, counts it
  as reordered, and discards it (the filters cannot rewind). A seq that was
  already seen is counted as a duplicate instead and does not change the drop
  count.
- negative by a lot: the device rebooted and restarted at 0, so the tracker
  resyncs.

The device timestamp resolves the cases the counter cannot. A reboot resets
the clock too, so a repeated seq with a new timestamp, or a "late" packet that
is out of place in time, is treated as a reboot even when seq only went back a
little.

The modulo handles wraparound without a special case. After 4294967295 comes
0, and `(0 − 4294967295) mod 2³² = 1`, which is the next expected number. At
50 packets/s a u32 wraps after ~2.7 years, but the code does not rely on that.

The CRC-16 catches corruption before the tracker sees the packet, so a
corrupted packet is counted both as `corrupt` and as a seq gap.

## Latency budget

The table covers the time from a gesture change to the JSON message leaving
the PC.

| Stage | Typical |
|---|---|
| Fill a 20-sample packet | 0-20 ms (avg 10) |
| WiFi UDP to PC | 2-10 ms (power save off) |
| DSP + features + LDA per packet | ~0.1 ms (demo measures ~100 µs) |
| Wait for next 50 ms decision step | 0-50 ms (avg 25) |
| Window must be mostly the new gesture | ~100 ms (half the 200 ms window) |
| Majority vote (k = 5) | ~100-150 ms at a switch |
| JSON to Quest over WiFi | 2-10 ms |

The total is roughly 150-300 ms end to end. Most of it comes from windowing
and smoothing; the network and the CPU contribute little. The usual guideline
for myoelectric control is to stay under ~300 ms so that control feels
responsive. The simplest ways to reduce latency are a smaller vote k, a
shorter step or a shorter window, and each trades stability for speed.

## Limitations

- Simulated data is not real EMG. The simulator is band-limited noise scaled
  by made-up per-channel activation patterns. Real signals have motion
  artefacts, cross-talk between neighbouring muscles, fatigue (the spectrum
  shifts lower over minutes), sweat changing impedance, and cable noise.
- The demo's accuracy says nothing about real EMG. On public data the same
  pipeline gets 92.4% within one sitting and 73.5% on later days without
  recalibration ([results.md](results.md)). One second per gesture at the
  start of each session, passed to `adapt()`, raises it to 89.0%, and the
  adaptation takes well under a millisecond. `adapt()` is the covariate-shift
  adaptation of Vidovic et al. (*IEEE TNSRE* 24(9):961-970, 2016).
- Electrode placement matters more than the algorithm. Moving the band by
  1 cm changes the patterns. Mark the positions on the skin, put the
  electrodes along the muscle fibres over the belly, and put the reference on
  bone.
- The ESP32 ADC is nonlinear near the rails, noisy, and has per-chip offsets.
  It is adequate for a prototype; a dedicated ADC (ADS1299, ADS131) is on the
  roadmap.
- Only the mains fundamental (50 or 60 Hz) is notched; its harmonics pass
  through. Battery power and short leads keep them small.
- Lost packets leave a gap in time that the filters don't know about. At ~1%
  loss this has little effect; a link with heavier loss would need
  zero-filling or interpolation.

## Design decisions

### UDP rather than TCP

For real-time control, old data has no value. TCP retransmits a lost packet
and holds newer data behind it (head-of-line blocking), so one loss becomes a
freeze. With UDP the receiver drops the packet, counts it from the sequence
numbers, and continues. `listen` shows the live loss rate.

### Timer, task and queue in the firmware

Sampling in `loop()` with `delay()` drifts, and WiFi sends can block for
milliseconds. A hardware timer gives exact 1 ms ticks. The ISR only notifies a
high-priority task, because ISRs must be short and `analogRead` is not
ISR-safe. The queue between sampling and sending means slow WiFi drops whole
packets, which the PC can count, instead of distorting the sample timing.

### ADC1 pins and the RAW output

ADC2 is shared with the WiFi radio and cannot be read while WiFi is on, so
the sensors use ADC1 pins. The MyoWare's RAW output keeps the full EMG
waveform. Its ENV output is already rectified and smoothed by the sensor,
which removes the information the frequency features (zero crossings, slope
sign changes) depend on.

### A CRC on top of UDP's checksum

The UDP checksum is optional in IPv4 and is only a 16-bit ones'-complement
sum. The packet's own CRC-16 is end to end, under our control and testable,
and it catches all burst errors up to 16 bits. It costs 2 bytes per 182-byte
packet.

### LDA rather than a neural network

LDA is trained in closed form and predicts with one matrix multiply. With
about a minute of calibration data per session, a deep model would overfit,
and LDA is the standard baseline in the field. It is also small enough to run
on the ESP32. A CNN becomes worth trying once there are hours of real
recordings across sessions.

### Sending gestures to a headset

`myobridge forward` sends one small JSON message per decision over UDP. A
Unity script on a Quest listens on a port, parses each message with
`JsonUtility` and keeps the latest gesture ([protocol](protocol.md)). Later
the classifier could run on the ESP32 and send only the gesture, or the Quest
could receive raw packets directly.
