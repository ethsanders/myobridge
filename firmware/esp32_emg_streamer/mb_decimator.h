// mb_decimator.h: halfband low-pass + keep every 2nd sample, in integer maths.
//
// Used when the sketch samples at 2x the output rate (OVERSAMPLE 2 in
// config.h). Bit-exact with src/myobridge/decimator.py, which documents the
// design; firmware/tests/ checks the two agree on random input.
//
// 47 taps, Q15. Every odd offset from the centre is zero and the centre is
// exactly 0.5, so one output costs 12 symmetric pairs + the centre tap.
// Worst case |acc| = 32768 * sum|h| = 1.76e9, which fits in int32.
#pragma once

#include <stdint.h>

namespace mb {

static const int HALFBAND_TAPS = 47;
static const int HALFBAND_CENTRE = 23;
// The non-zero taps on one side, from the outermost (index 0) inwards to
// index 22; the mirror half and the zero taps are implied.
static const int16_t HALFBAND_Q15_HALF[12] = {-7,   23,  -55,  107,   -189,  311,
                                              -489, 750, -1150, 1830, -3318, 10377};

template <int NCH>
class HalfbandDecimator {
 public:
  HalfbandDecimator() { reset(); }

  void reset() {
    for (int i = 0; i < HALFBAND_TAPS; i++)
      for (int c = 0; c < NCH; c++) hist_[i][c] = 0;
    head_ = 0;
  }

  // Feed two consecutive input frames; writes one output frame.
  void push(const int16_t *first, const int16_t *second, int16_t *out) {
    store(first);
    store(second);
    for (int c = 0; c < NCH; c++) {
      // at(k) = x[n - k], n = the newest sample
      int32_t acc = (int32_t)16384 * at(HALFBAND_CENTRE, c);
      for (int j = 0; j < 12; j++) {
        int k = 2 * j;  // non-zero taps sit at even indices 0, 2, ..., 22
        acc += (int32_t)HALFBAND_Q15_HALF[j] * ((int32_t)at(k, c) + at(HALFBAND_TAPS - 1 - k, c));
      }
      int32_t y = (acc + (1 << 14)) >> 15;  // arithmetic shift: rounds like Python
      if (y > 32767) y = 32767;
      if (y < -32768) y = -32768;
      out[c] = (int16_t)y;
    }
  }

 private:
  void store(const int16_t *frame) {
    head_ = (head_ + 1) % HALFBAND_TAPS;
    for (int c = 0; c < NCH; c++) hist_[head_][c] = frame[c];
  }
  int16_t at(int k, int c) const { return hist_[(head_ - k + HALFBAND_TAPS) % HALFBAND_TAPS][c]; }

  int16_t hist_[HALFBAND_TAPS][NCH];
  int head_;
};

}  // namespace mb
