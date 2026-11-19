# How myobridge works (interview prep)

Plain-language notes on every piece: what it does, why it's built that way,
and the numbers behind it. If you can explain this page, you can explain the repo.

## The big picture

When a muscle contracts, many motor units fire. Each firing sends a tiny
electrical wave along the muscle fibres. Electrodes on the skin pick up the sum
of all of them: **surface EMG**, a noisy, zero-mean signal of a few hundred
microvolts to a few millivolts whose *amplitude* rises with effort and whose
*spectrum* sits roughly between 20 and 450 Hz.

The MyoWare 2.0 amplifies that to volts. The ESP32 samples it, packs it into
UDP packets and sends it over WiFi. The PC cleans the signal, summarises each
200 ms window as 20 numbers (features), and a linear classifier maps those
numbers to a gesture. The gesture goes out as JSON to a game engine.

```
skin -> MyoWare (amp) -> ESP32 ADC (1 kHz) -> UDP -> receiver -> bandpass -> notch
     -> 200 ms windows every 50 ms -> features -> LDA -> majority vote -> JSON -> Unity
```

## Module by module

| File | Job | Why it's built like this |
|---|---|---|
| `firmware/.../esp32_emg_streamer.ino` | Hardware timer fires every 1 ms; the ISR only wakes a sampler task, which reads 4 ADC pins. Every 20 frames it builds a packet and hands it to `loop()` through a FreeRTOS queue; `loop()` sends it over UDP. | ISRs must be short, and WiFi sends can block for milliseconds. Splitting "sample" and "send" across a queue means slow WiFi never makes us miss a sample. If the queue is full we drop the packet but still increment `seq`, so the PC *sees* the loss. |
| `protocol.py` | Encode/decode packets, CRC, `SequenceTracker`. | One place defines the byte layout. The tracker turns sequence numbers into drop / reorder / restart counts. |
| `receiver.py` | Blocking UDP socket with a 0.5 s timeout, yields packets. | No threads or asyncio: easiest to reason about, and the timeout keeps Ctrl-C and the display responsive. |
| `simulator.py` | Fake armband producing the same int16 packets. | Lets everything run and be tested without hardware; deterministic with a seed. |
| `dsp.py` | Streaming bandpass, notch, rectify, RMS envelope. | Filters keep state between packets so chunked == one-shot. |
| `features.py` | MAV, RMS, waveform length, zero crossings, slope sign changes. | The classic Hudgins time-domain set: cheap, robust, well studied. |
| `classifier.py` | LDA, calibration, majority vote. | Closed-form, fast, small-data friendly. |
| `pipeline.py` | Ties filter + windower + features + LDA + vote together. | Same code path for the demo, `forward`, and the plot script. |
| `cli.py` | `simulate`, `listen`, `calibrate`, `forward`, `demo`. | Thin wrappers around the above. |

## The signal chain, with numbers

**Sampling at 1 kHz.** Nyquist: to capture frequencies up to *f* you must
sample faster than 2*f*. Surface EMG has useful content up to ~450-500 Hz, so
1 kHz is the minimum sensible rate (research systems often use 2 kHz). The
MyoWare's own analog filtering rolls off the high end, which acts as a gentle
anti-aliasing filter (a steeper one is a job for a custom front end).
The ESP32's ADC is 12-bit (0-4095); we subtract 2048
so the samples are signed.

**Bandpass 20-450 Hz (4th-order Butterworth, so 8th order overall).**
- Below 20 Hz is mostly *motion artefact* (cables and skin moving) and
  *baseline drift* (electrode-skin potential slowly changing). Not muscle.
- Above 450 Hz there's little EMG power left, mostly amplifier/ADC noise.
  450 Hz is also 0.9× the 500 Hz Nyquist limit, about as high as we can go.
- Butterworth = maximally flat passband, so amplitudes inside the band aren't
  distorted. Implemented as second-order sections (biquads) for numerical stability.

**Notch at 60 Hz (Q = 30, ~2 Hz wide).** Mains wiring radiates 60 Hz (50 Hz
in Europe) and your body acts as an antenna, so hum sits right in the middle
of the EMG band where the bandpass can't remove it. A narrow notch removes it
while barely touching neighbouring frequencies. Tests check >20 dB attenuation
at 60 Hz and <0.5 dB change at 30/100/150/250/400 Hz.

**Rectify + RMS envelope (100 ms).** EMG is zero-mean, so its average is
useless; its *size* is what tracks effort. Full-wave rectification (absolute
value) followed by a moving RMS gives a smooth "how hard is this muscle
working" curve. That's what the `listen` bars show.

**Windows: 200 ms long, every 50 ms.** Long enough for stable features, short
enough for responsive control. Overlap gives a fresh decision every 50 ms.

**Why streaming state matters.** An IIR filter's output depends on its past
inputs/outputs. If you filtered each 20-sample packet from a zero state you'd
get a startup transient every 20 ms. `scipy.signal.sosfilt(..., zi=state)`
returns the final state, which we feed into the next call. A test filters the
same signal in blocks of 1, 7, 20 and 333 samples and checks it matches the
one-shot result to 1e-9.

## Features (per channel, per window)

| Feature | Formula | Captures |
|---|---|---|
| MAV | mean(\|x\|) | amplitude |
| RMS | sqrt(mean(x²)) | amplitude / power |
| Waveform length | Σ\|x[i+1] − x[i]\| | amplitude × frequency |
| Zero crossings | sign changes with a jump ≥ threshold | frequency |
| Slope sign changes | peaks/valleys above a threshold | frequency |

4 channels × 5 features = 20 numbers per window. The amplitude features are
`log1p`-scaled because EMG amplitude spans orders of magnitude between rest and
a full fist; on a log scale the class clusters have similar spread, which is
what LDA's shared-covariance assumption wants.

## LDA in five sentences

LDA assumes each gesture's feature vectors form a Gaussian cloud with its own
centre (mean) but the same shape (one covariance matrix shared by all classes).
Fitting is just computing each class mean and the pooled covariance, no
iterative optimisation. Because the covariance is shared, comparing class
log-likelihoods reduces to a linear function of the features: `score = x·W + b`.
Prediction is one 20×5 matrix multiply plus a softmax for probabilities.
We add a small ridge (a bit of the identity) to the covariance so it is always
invertible, e.g. if a channel is dead.

**Majority vote (k = 5).** The output is the most common label of the last 5
windows (ties go to the newest). It removes single-window flickers at the cost
of ~100 ms extra lag when you switch gesture. In the demo that's why smoothed
accuracy is slightly *lower* than raw: the lag costs a few windows at every
switch, but the output a user sees is far steadier.

## How loss detection works

Every packet carries a 32-bit counter. The receiver remembers the next number
it expects and computes `ahead = (seq − expected) mod 2³²`:
- `0`: on time.
- small positive: a gap; that many packets were lost.
- "negative" (≥ 2³¹ after the modulo) by a little: a late or duplicate packet.
  We had counted it as dropped, so undo that, count it as reordered, and throw
  it away (filters can't rewind).
- negative by a lot (>1000): the device rebooted and restarted at 0, so resync.

The modulo makes wraparound free: after 4294967295 comes 0, and
`(0 − 4294967295) mod 2³² = 1`, i.e. "the next one". (At 50 packets/s a u32
wraps after ~2.7 years, but correct code shouldn't depend on that.)

Corruption is caught by the CRC-16 before the tracker sees the packet, so a
corrupted packet shows up as both `corrupt` and a seq gap.

## Latency budget (gesture change → JSON leaves the PC)

| Stage | Typical |
|---|---|
| Fill a 20-sample packet | 0-20 ms (avg 10) |
| WiFi UDP to PC | 2-10 ms (power save off) |
| DSP + features + LDA per packet | ~0.1 ms (demo measures ~100 µs) |
| Wait for next 50 ms decision step | 0-50 ms (avg 25) |
| Window must be mostly the new gesture | ~100 ms (half the 200 ms window) |
| Majority vote (k = 5) | ~100-150 ms at a switch |
| JSON to Quest over WiFi | 2-10 ms |

Roughly 150-300 ms end to end, dominated by *windowing and smoothing*, not by
the network or the CPU. The usual guideline for myoelectric control is to stay
under ~300 ms so it feels responsive. Easiest knobs: smaller vote k, shorter
step, shorter window (each trades stability for speed).

## Honest limitations

- **Simulated data is not real EMG.** The simulator is band-limited noise
  scaled by made-up per-channel activation patterns. Real signals have motion
  artefacts, cross-talk between neighbouring muscles, fatigue (spectrum shifts
  lower over minutes), sweat changing impedance, and cable noise.
- **Real accuracy will be lower** than the demo's ~95%, especially across
  sessions. Recalibrating at the start of each session (a minute of data)
  is normal practice; LDA trains in milliseconds so that's cheap.
- **Electrode placement matters more than the algorithm.** Shift the band 1 cm
  and the patterns change. Mark positions on the skin; put electrodes along
  the muscle fibres over the belly; reference on bone.
- **The ESP32 ADC is mediocre:** nonlinear near the rails, noisy, with
  per-chip offsets. Fine for a prototype; a dedicated ADC (ADS1299, ADS131) is
  on the roadmap.
- Only the 60 Hz fundamental is notched; harmonics (120, 180 Hz) pass through.
  Battery power and short leads keep them small.
- Lost packets leave a gap in time that the filters don't know about.
  At ~1% loss this barely matters; a heavier-loss link would want
  zero-filling or interpolation.

## Likely interview questions

1. **Why UDP and not TCP?**
   For real-time control, old data is useless. TCP retransmits and blocks
   newer data behind a lost packet (head-of-line blocking), so one loss becomes
   a freeze. With UDP I drop it, count it with sequence numbers, and keep going.

2. **How do you know if you're losing packets?**
   Every packet has a u32 sequence number. Gaps mean drops, numbers behind the
   expected one mean reordering or duplicates, a big backwards jump means the
   device rebooted. All math is mod 2³² so wraparound isn't a special case. The
   `listen` command shows the live loss %.

3. **Why sample at 1 kHz?**
   Surface EMG has content up to ~450-500 Hz; Nyquist says sample at more than
   twice the highest frequency, so 1 kHz is the minimum. The MyoWare's analog
   filtering rolls off the high end and acts as a gentle anti-alias filter; a
   steeper one belongs on the custom front-end PCB.

4. **Why a 20-450 Hz bandpass and a 60 Hz notch?**
   Below 20 Hz is motion artefact and drift, above 450 Hz is mostly noise.
   Mains hum at 60 Hz is inside the EMG band so the bandpass can't remove it;
   a narrow notch (Q = 30, ~2 Hz wide) does, and tests show >20 dB rejection
   with the passband unchanged.

5. **What does "streaming filter" mean and how did you test it?**
   The filter carries its internal state from one packet to the next, so
   processing in chunks gives the same output as processing all at once. The
   test filters random data in chunks of 1, 7, 20 and 333 samples and compares
   to the one-shot output.

6. **Explain LDA. Why not a neural network?**
   Gaussian classes with a shared covariance give linear decision boundaries;
   training is closed-form, prediction is a matrix multiply. With a minute of
   calibration data per session, a deep model would overfit, and LDA is the
   standard baseline in the field. It's also small enough to run on the ESP32.
   I'd try a CNN once I have hours of real data across sessions.

7. **What is the end-to-end latency, and where does it go?**
   About 150-300 ms, mostly from the 200 ms window and the majority vote, not
   from the network (~5 ms) or computation (~0.1 ms per packet, measured).
   To reduce it I'd shrink the vote or the window step.

8. **Why ISR + task + queue in the firmware instead of reading in `loop()`?**
   Timing. `loop()` with `delay()` drifts and WiFi sends can block. A hardware
   timer gives exact 1 ms ticks; the ISR only notifies a high-priority task
   (ISRs must be short and `analogRead` isn't ISR-safe); a queue decouples
   sampling from sending, so slow WiFi drops packets visibly instead of
   distorting timing.

9. **Why ADC1 pins? Why the RAW output?**
   ADC2 is shared with the WiFi radio and can't be read while WiFi is on.
   RAW keeps the full EMG waveform; ENV is already rectified and smoothed by
   the sensor, which kills the frequency features (zero crossings, slope
   changes) and can't be undone.

10. **Why a CRC when UDP has a checksum?**
    The UDP checksum is optional in IPv4 and only 16-bit ones'-complement; our
    CRC is end-to-end, defined by us, testable, and catches all burst errors
    up to 16 bits. It costs 2 bytes per 182-byte packet.

11. **What's the biggest risk when moving to real hardware?**
    Signal quality: electrode placement, skin prep, motion artefacts, and the
    ESP32 ADC's noise. And safety: electrodes on skin must never share a
    circuit with mains power, so everything runs from a battery.

12. **How would this get to a Quest?**
    `myobridge forward` sends a small JSON message per decision over UDP. A
    Unity script listens on a port, parses it with `JsonUtility`, and keeps the
    latest gesture. Later the classifier could run on the ESP32 and send just
    the gesture, or the Quest could receive raw packets directly.
