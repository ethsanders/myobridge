# myobridge wire protocol (version 1)

One UDP datagram carries one packet. A packet holds `n_frames` consecutive
samples from every channel. All multi-byte fields are **little-endian** (the
native byte order of both the ESP32 and x86/ARM PCs, written out explicitly so
the format never depends on the CPU).

## Layout

| Offset | Size | Type   | Field          | Notes |
|-------:|-----:|--------|----------------|-------|
| 0      | 2    | bytes  | `magic`        | ASCII `"MB"` (`0x4D 0x42`). Rejects stray traffic on the port. |
| 2      | 1    | u8     | `version`      | `1`. Bump on any incompatible change. |
| 3      | 1    | u8     | `n_channels`   | e.g. `4` |
| 4      | 2    | u16    | `sample_rate`  | Hz, e.g. `1000` |
| 6      | 4    | u32    | `seq`          | Packet counter, +1 per packet, wraps 4294967295 → 0 |
| 10     | 8    | u64    | `timestamp_us` | Device clock (`esp_timer_get_time()`) at the first sample |
| 18     | 2    | u16    | `n_frames`     | Samples per channel in this packet, e.g. `20` |
| 20     | 2·C·N| i16[]  | `samples`      | Interleaved by frame: `f0c0 f0c1 … f0cC-1 f1c0 …` |
| 20+2CN | 2    | u16    | `crc`          | CRC-16/CCITT-FALSE over bytes `0 .. 20+2CN-1` |

Header = 20 bytes, trailer = 2 bytes. With the default 4 channels × 20 frames
a packet is `20 + 160 + 2 = 182` bytes, sent 50 times per second
(~73 kbit/s including UDP/IP headers, a tiny fraction of WiFi bandwidth).

Samples are ADC counts minus the ADC midpoint (2048 for 12 bits), so the
signal is centred near zero. 12-bit values fit comfortably in int16.

## CRC

CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, input and
output not reflected, final XOR `0x0000`. Check value: `"123456789"` → `0x29B1`.

UDP already has a 16-bit checksum, but it is optional over IPv4 and some
stacks/NICs/bridges zero it out; our own CRC makes corruption detection
end-to-end and testable.

## Receiver rules

A datagram is rejected (counted as `corrupt`) if:

1. it is shorter than header + CRC (`truncated`),
2. `magic` is not `"MB"` (`magic`),
3. `version` is unknown (`version`),
4. its length is not exactly `20 + 2·n_channels·n_frames + 2` (`truncated`),
5. the CRC doesn't match (`crc`).

Valid packets go through the sequence tracker. With `expected` = last
accepted `seq` + 1, and all arithmetic modulo 2³²:

| `ahead = (seq - expected) mod 2³²` | Meaning | Action |
|---|---|---|
| `0` | in order | accept |
| `1 … 2³¹-1` | gap: `ahead` packets missing | `dropped += ahead`, accept |
| `2³²-1000 … 2³²-1` (up to 1000 behind) | late or duplicate | `reordered += 1`, `dropped -= 1`, **discard** |
| further behind | device restarted at 0 | `restarts += 1`, accept, resync |

Late packets are discarded because the streaming filters can't go back in
time; inserting an old block would corrupt their state.

## Gesture output (`myobridge forward`)

The PC sends one small JSON object per classification step (every 50 ms by
default) as a UDP datagram, UTF-8, to the game engine (default
`127.0.0.1:5006`):

```json
{"type": "gesture", "gesture": "fist", "confidence": 0.973, "seq": 1842, "t": 1763520000.123}
```

| Field | Meaning |
|---|---|
| `type` | Always `"gesture"` (room for `"imu"`, `"status"` later) |
| `gesture` | Majority-vote-smoothed label, one of the calibrated gestures |
| `confidence` | LDA probability of that label for the latest window (0..1) |
| `seq` | Message counter, so the app can ignore stale/out-of-order messages |
| `t` | Sender's Unix time in seconds |

A Unity script only needs a `UdpClient` on port 5006 and `JsonUtility.FromJson`
on each datagram; keep the latest message and act on it in `Update()`.
