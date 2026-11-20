// node --test docs/demo/demo.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DATA_BASE,
  Ring,
  channelState,
  mapLimited,
  parseRecording,
  plan,
  prettyGesture,
  recordingUrl,
  traceScale,
} from "./core.js";

test("urls follow the 3DC layout at the pinned commit", () => {
  assert.equal(
    recordingUrl(7, 1, 0, 10),
    `${DATA_BASE}/Participant7/test/EMG/3dc_EMG_gesture_0_10.txt`,
  );
  const { train, test: replay } = plan(3);
  assert.equal(train.length, 22);
  assert.equal(replay.length, 11);
  assert.deepEqual(
    replay.map((r) => r.label),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  );
  assert.ok(train.every((r) => r.session === 0 && r.url.includes("/train/")));
});

test("parses comma-separated rows and rejects ragged files", () => {
  const values = parseRecording("1,2,3,4,5,6,7,8,9,10\n-1,-2,-3,-4,-5,-6,-7,-8,-9,-10\n");
  assert.equal(values.length, 20);
  assert.equal(values[10], -1);
  assert.ok(values instanceof Float64Array);
  assert.throws(() => parseRecording("1,2,3"), /rows of 10/);
  assert.throws(() => parseRecording("1,2,3,4,5,6,7,8,9,x"), /not a number/);
});

test("mapLimited keeps order and limits concurrency", async () => {
  let running = 0;
  let peak = 0;
  const seen = [];
  const out = await mapLimited(
    [5, 1, 4, 2, 3],
    2,
    async (x) => {
      running++;
      peak = Math.max(peak, running);
      await new Promise((r) => setTimeout(r, x));
      running--;
      return x * 10;
    },
    (done, total) => seen.push(`${done}/${total}`),
  );
  assert.deepEqual(out, [50, 10, 40, 20, 30]);
  assert.equal(peak, 2);
  assert.equal(seen.at(-1), "5/5");
});

test("ring keeps the newest samples, oldest first", () => {
  const r = new Ring(3);
  [1, 2].forEach((v) => r.push(v));
  assert.deepEqual([...r.toArray()], [1, 2]);
  [3, 4, 5].forEach((v) => r.push(v));
  assert.deepEqual([...r.toArray()], [3, 4, 5]);
});

test("trace scale ignores a few spikes", () => {
  const values = Array.from({ length: 1000 }, (_, i) => (i % 2 ? 10 : -10));
  values[0] = 1e6;
  const { median, scale } = traceScale(values);
  assert.equal(Math.abs(median), 10);
  assert.ok(scale <= 20);
  assert.deepEqual(traceScale([]), { median: 0, scale: 1 });
});

test("labels and channel states", () => {
  assert.equal(prettyGesture("wrist_flexion"), "wrist flexion");
  assert.equal(prettyGesture(null), "-");
  assert.equal(channelState({ issues: [], dropped: false }), "ok");
  assert.equal(channelState({ issues: ["motion"], dropped: false }), "warn");
  assert.equal(channelState({ issues: ["flatline"], dropped: true }), "bad");
});
