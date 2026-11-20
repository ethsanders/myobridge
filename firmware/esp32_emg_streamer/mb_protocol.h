// mb_protocol.h: the myobridge wire format, as plain C++ with no Arduino code.
//
// Kept separate from the sketch so the exact same code can be compiled on a
// PC and tested against the Python encoder/decoder (firmware/tests/). The
// format is documented in docs/protocol.md and implemented in Python in
// src/myobridge/protocol.py.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace mb {

static const uint8_t MAGIC_0 = 'M';
static const uint8_t MAGIC_1 = 'B';
static const uint8_t PROTOCOL_VERSION = 1;
static const size_t HEADER_SIZE = 20;
static const size_t CRC_SIZE = 2;

inline size_t packetSize(uint8_t nChannels, uint16_t nFrames) {
  return HEADER_SIZE + (size_t)nChannels * nFrames * 2 + CRC_SIZE;
}

// ---- Little-endian helpers (write the low byte first) ----------------------
inline void putU16(uint8_t *buf, size_t at, uint16_t v) {
  buf[at] = (uint8_t)(v & 0xFF);
  buf[at + 1] = (uint8_t)(v >> 8);
}

inline void putU32(uint8_t *buf, size_t at, uint32_t v) {
  for (int i = 0; i < 4; i++) buf[at + i] = (uint8_t)((v >> (8 * i)) & 0xFF);
}

inline void putU64(uint8_t *buf, size_t at, uint64_t v) {
  for (int i = 0; i < 8; i++) buf[at + i] = (uint8_t)((v >> (8 * i)) & 0xFF);
}

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, MSB first, no final XOR.
// Must match protocol.crc16_ccitt_false() in Python ("123456789" -> 0x29B1).
inline uint16_t crc16CcittFalse(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

inline void writeHeader(uint8_t *buf, uint8_t nChannels, uint16_t sampleRateHz, uint32_t seq,
                        uint64_t timestampUs, uint16_t nFrames) {
  buf[0] = MAGIC_0;
  buf[1] = MAGIC_1;
  buf[2] = PROTOCOL_VERSION;
  buf[3] = nChannels;
  putU16(buf, 4, sampleRateHz);
  putU32(buf, 6, seq);
  putU64(buf, 10, timestampUs);
  putU16(buf, 18, nFrames);
}

// Store one sample (frame f, channel c) in an already-sized packet buffer.
inline void putSample(uint8_t *buf, uint8_t nChannels, uint16_t frame, uint8_t channel,
                      int16_t value) {
  putU16(buf, HEADER_SIZE + ((size_t)frame * nChannels + channel) * 2, (uint16_t)value);
}

// Header + CRC around samples already written with putSample(). Returns the size.
inline size_t finishPacket(uint8_t *buf, uint8_t nChannels, uint16_t sampleRateHz, uint32_t seq,
                           uint64_t timestampUs, uint16_t nFrames) {
  writeHeader(buf, nChannels, sampleRateHz, seq, timestampUs, nFrames);
  size_t size = packetSize(nChannels, nFrames);
  putU16(buf, size - CRC_SIZE, crc16CcittFalse(buf, size - CRC_SIZE));
  return size;
}

}  // namespace mb
