// Pieces of the demo that don't touch the page, so node can test them (demo.test.mjs).

// The 3DC mirror at the commit myobridge pins (src/myobridge/datasets/threedc.py).
export const DATA_BASE =
  "https://raw.githubusercontent.com/libemg/3DCDataset/1af9058237a5d79a883feb2ce61f4d59090288ac";
export const N_PEOPLE = 22;
export const N_GESTURES = 11;
export const TRAIN_REPS = [0, 1]; // first two cycles of the first half of the session
export const TEST_REP = 0; // first cycle of the second half
export const EVEN_4 = [0, 2, 5, 7]; // the same four channels as the benchmarks

export function recordingUrl(person, session, rep, label) {
  const folder = session === 0 ? "train" : "test";
  return `${DATA_BASE}/Participant${person}/${folder}/EMG/3dc_EMG_gesture_${rep}_${label}.txt`;
}

// Every recording the demo needs, in play order for the test part.
export function plan(person) {
  const train = [];
  for (const rep of TRAIN_REPS) {
    for (let label = 0; label < N_GESTURES; label++) {
      train.push({ url: recordingUrl(person, 0, rep, label), label, session: 0, rep });
    }
  }
  const test = [];
  for (let label = 0; label < N_GESTURES; label++) {
    test.push({ url: recordingUrl(person, 1, TEST_REP, label), label, session: 1, rep: TEST_REP });
  }
  return { train, test };
}

// A 3DC file is comma-separated integers, one row per sample.
export function parseRecording(text, nChannels = 10) {
  const parts = text.trim().split(/[\s,]+/);
  if (parts.length === 0 || parts.length % nChannels !== 0) {
    throw new Error(`expected rows of ${nChannels} numbers, got ${parts.length} values`);
  }
  const out = new Float64Array(parts.length);
  for (let i = 0; i < parts.length; i++) {
    const v = Number(parts[i]);
    if (!Number.isFinite(v)) throw new Error(`not a number: ${parts[i]}`);
    out[i] = v;
  }
  return out;
}

// Run `worker` over `items` with at most `limit` at a time, keeping order.
export async function mapLimited(items, limit, worker, onDone = () => {}) {
  const results = new Array(items.length);
  let next = 0;
  let done = 0;
  async function lane() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
      onDone(++done, items.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, lane));
  return results;
}

// Fixed-length history of one channel's samples for drawing, oldest first.
export class Ring {
  constructor(size) {
    this.data = new Float32Array(size);
    this.start = 0;
    this.filled = 0;
  }
  push(value) {
    const i = (this.start + this.filled) % this.data.length;
    this.data[i] = value;
    if (this.filled < this.data.length) this.filled++;
    else this.start = (this.start + 1) % this.data.length;
  }
  toArray() {
    const out = new Float32Array(this.filled);
    for (let i = 0; i < this.filled; i++) out[i] = this.data[(this.start + i) % this.data.length];
    return out;
  }
}

// Robust vertical scale for a trace: the 99th percentile of |x - median|.
export function traceScale(values) {
  if (values.length === 0) return { median: 0, scale: 1 };
  const sorted = Float32Array.from(values).sort();
  const median = sorted[sorted.length >> 1];
  const dev = Float32Array.from(values, (v) => Math.abs(v - median)).sort();
  const p99 = dev[Math.min(dev.length - 1, Math.floor(dev.length * 0.99))];
  return { median, scale: Math.max(p99, 1e-6) };
}

export function prettyGesture(name) {
  return name ? name.replaceAll("_", " ") : "-";
}

// The state a channel's dot shows: ok, warn (flagged, kept) or bad (dropped).
export function channelState(ch) {
  if (ch.dropped) return "bad";
  return ch.issues.length ? "warn" : "ok";
}

export const ISSUE_TEXT = {
  flatline: "no signal",
  clipping: "clipping",
  line_noise: "mains hum",
  motion: "motion",
};
