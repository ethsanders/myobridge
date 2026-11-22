# myobridge wire protocol (version 1)

One UDP datagram carries one packet. A packet holds `n_frames` consecutive
samples from every channel. All multi-byte fields are **little-endian**. That is
the native byte order of both the ESP32 and x86/ARM PCs, but it is stated
here explicitly so that the format does not depend on the CPU.

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

The header is 20 bytes and the trailer 2 bytes. With the default 4 channels
× 20 frames a packet is `20 + 160 + 2 = 182` bytes, sent 50 times per second.
That is ~73 kbit/s including UDP/IP headers, a small fraction of WiFi
bandwidth.

Samples are ADC counts minus the ADC midpoint (2048 for 12 bits), so the
signal is centred near zero. 12-bit values fit in int16 with room to spare.

## CRC

CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, input and
output not reflected, final XOR `0x0000`. Check value: `"123456789"` → `0x29B1`.

UDP already has a 16-bit checksum, but it is optional over IPv4, and some
network stacks, NICs and bridges zero it out. A CRC in the packet itself
detects corruption end to end and can be tested.

## Receiver rules

A datagram is rejected (counted as `corrupt`) if:

1. it is shorter than header + CRC (`truncated`),
2. `magic` is not `"MB"` (`magic`),
3. `version` is unknown (`version`),
4. the header asks for more than fits in one datagram (65507 bytes) (`format`),
5. its length is not exactly `20 + 2·n_channels·n_frames + 2` (`truncated`),
6. the CRC doesn't match (`crc`),
7. `n_channels` or `sample_rate` is 0 (`format`).

`n_frames = 0` is valid. Such a packet carries no samples.

Valid packets go through the sequence tracker. With `expected` = last
accepted `seq` + 1, and all arithmetic modulo 2³²:

| `ahead = (seq - expected) mod 2³²` | Meaning | Action |
|---|---|---|
| `0` | in order | accept |
| `1 … 2³¹-1` | gap: `ahead` packets missing | `dropped += ahead`, accept |
| `2³²-32 … 2³²-1` (up to 32 behind), seq already seen | duplicate | `duplicates += 1`, **discard** |
| `2³²-32 … 2³²-1`, seq not seen | late | `reordered += 1`, `dropped -= 1`, **discard** |
| further behind | device restarted | `restarts += 1`, accept, resync |

A reboot resets the device clock along with `seq`, so `timestamp_us` overrides
the table: a packet is also a restart when its `seq` was already seen with a
different timestamp, when a "late" packet's timestamp doesn't lie between
those of the packets either side of it, or when a newer `seq` carries an
older timestamp than the last packet.

Late packets are discarded because the streaming filters cannot go back in
time. Inserting an old block would corrupt their state.

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

A Unity script needs a `UdpClient` on port 5006 and `JsonUtility.FromJson`
on each datagram. It should keep the latest message and act on it in
`Update()`.

## Lab Streaming Layer (`myobridge forward --lsl [NAME]`)

The same decisions can also be sent on Lab Streaming Layer, alongside the
UDP JSON. `NAME` defaults to `myobridge`. Two streams share the device's
source id, so a recording of both lines up:

| Stream | LSL type | Channels | Rate | Content |
|---|---|---|---|---|
| `NAME-emg` | `EMG` | one per electrode, float32 | the device's sample rate | raw ADC counts as they arrive |
| `NAME-gestures` | `Markers` | 1, string | irregular | `{"gesture": "fist", "confidence": 0.9734, "sample_index": 12000}` |

`sample_index` counts samples since the decoder started, so a marker can be
placed exactly on the EMG timeline. The streams are made by
`myobridge.LslOutlets`, which other programs can use directly.
