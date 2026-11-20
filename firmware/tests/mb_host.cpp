// Host-side driver for the firmware's pure C++ headers, so pytest can check
// them against the Python implementation (tests/test_firmware.py).
//
//   mb_host selftest
//   mb_host packet <channels> <frames> <rate> <seq> <timestamp_us>   < samples
//   mb_host decimate <frames>                                         < samples
//
// Samples are whitespace-separated integers, frame by frame. `packet` prints
// the packet as hex; `decimate` (4 channels) prints the output samples.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../esp32_emg_streamer/mb_decimator.h"
#include "../esp32_emg_streamer/mb_protocol.h"

static int fail(const char *what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  return 1;
}

static int selftest() {
  const char *check = "123456789";
  if (mb::crc16CcittFalse((const uint8_t *)check, 9) != 0x29B1) return fail("crc check value");
  if (mb::packetSize(4, 20) != 182) return fail("packet size");
  uint8_t buf[8];
  mb::putU64(buf, 0, 0x0102030405060708ULL);
  if (buf[0] != 0x08 || buf[7] != 0x01) return fail("little endian");
  // A constant input settles to the same constant: the taps sum to 1.0 (+/- rounding).
  mb::HalfbandDecimator<1> d;
  int16_t a = 1000, out = 0;
  for (int i = 0; i < 40; i++) d.push(&a, &a, &out);
  if (out < 999 || out > 1001) return fail("dc gain");
  std::puts("ok");
  return 0;
}

static std::vector<long> readInts(size_t n) {
  std::vector<long> v(n);
  for (size_t i = 0; i < n; i++) {
    if (std::scanf("%ld", &v[i]) != 1) {
      std::fprintf(stderr, "expected %zu integers, got %zu\n", n, i);
      std::exit(2);
    }
  }
  return v;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "selftest")) return selftest();
  if (argc == 7 && !std::strcmp(argv[1], "packet")) {
    uint8_t nch = (uint8_t)std::atoi(argv[2]);
    uint16_t frames = (uint16_t)std::atoi(argv[3]);
    uint16_t rate = (uint16_t)std::atoi(argv[4]);
    uint32_t seq = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    uint64_t ts = std::strtoull(argv[6], nullptr, 10);
    std::vector<uint8_t> buf(mb::packetSize(nch, frames));
    std::vector<long> s = readInts((size_t)nch * frames);
    for (uint16_t f = 0; f < frames; f++)
      for (uint8_t c = 0; c < nch; c++) mb::putSample(buf.data(), nch, f, c, (int16_t)s[f * nch + c]);
    size_t n = mb::finishPacket(buf.data(), nch, rate, seq, ts, frames);
    for (size_t i = 0; i < n; i++) std::printf("%02x", buf[i]);
    std::puts("");
    return 0;
  }
  if (argc == 3 && !std::strcmp(argv[1], "decimate")) {
    const int NCH = 4;
    long frames = std::atol(argv[2]);
    std::vector<long> s = readInts((size_t)frames * NCH);
    mb::HalfbandDecimator<NCH> d;
    for (long f = 0; f + 1 < frames; f += 2) {
      int16_t a[NCH], b[NCH], out[NCH];
      for (int c = 0; c < NCH; c++) {
        a[c] = (int16_t)s[f * NCH + c];
        b[c] = (int16_t)s[(f + 1) * NCH + c];
      }
      d.push(a, b, out);
      for (int c = 0; c < NCH; c++) std::printf("%d%c", out[c], c + 1 < NCH ? ' ' : '\n');
    }
    return 0;
  }
  std::fprintf(stderr, "usage: see the comment at the top of mb_host.cpp\n");
  return 2;
}
