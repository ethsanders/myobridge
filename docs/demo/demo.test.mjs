// node --test docs/demo/demo.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, readFileSync } from "node:fs";
import {
  DATA_BASE,
  Ring,
  channelState,
  countUpTo,
  mapLimited,
  orderedSet,
  pct,
  segmentAt,
  setMembers,
  tally,
  variantKey,
  windowKind,
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

test("set bitmasks decode and order with the decoder's pick first", () => {
  assert.deepEqual(setMembers(0b1010, 4), [1, 3]);
  assert.deepEqual(setMembers(0, 11), []);
  assert.deepEqual(orderedSet(0b1011, 3, 11), [3, 0, 1]);
  assert.deepEqual(orderedSet(0b0110, 0, 11), [1, 2]); // pick not in the set
  assert.equal(variantKey(true, "myobridge", "ok"), "adapt-myobridge-ok");
  assert.equal(variantKey(false, "textbook", "off_monitor"), "static-textbook-off_monitor");
});

test("window kinds and running counts", () => {
  assert.equal(windowKind(1 << 4, 4), "sure");
  assert.equal(windowKind(1 << 4, 2), "wrong");
  assert.equal(windowKind((1 << 4) | (1 << 2), 2), "unsure");
  assert.equal(windowKind((1 << 4) | (1 << 5), 2), "missed");
  assert.equal(windowKind(0, 2), "missed");
  const truth = [0, 0, 1, 1];
  const sets = [1, 0b11, 0b01, 0b10]; // sure, unsure, wrong, sure
  const all = tally(truth, sets);
  assert.deepEqual(
    [all.windows, all.covered, all.singletons, all.singletonsRight],
    [4, 3, 3, 2],
  );
  assert.equal(all.coverage, 0.75);
  assert.equal(tally(truth, sets, 0).coverage, null);
  assert.equal(tally(truth, sets, 2).accuracyWhenSingleton, 1);
});

test("playback helpers", () => {
  assert.equal(countUpTo([0.2, 0.25, 0.3], 0.1), 0);
  assert.equal(countUpTo([0.2, 0.25, 0.3], 0.25), 2);
  assert.equal(countUpTo([0.2, 0.25, 0.3], 9), 3);
  const segs = [
    { gesture: 0, start: 0, length: 3.5 },
    { gesture: 1, start: 3.5, length: 3.4 },
  ];
  assert.equal(segmentAt(segs, 0).gesture, 0);
  assert.equal(segmentAt(segs, 3.6).gesture, 1);
  assert.equal(segmentAt(segs, 99).gesture, 1);
  assert.equal(pct(0.8912), "89.1%");
  assert.equal(pct(0.61, 0), "61%");
  assert.equal(pct(null), "-");
});

// The page counts from days.json; the benchmark that wrote it counted too.
const DAYS = new URL("./days.json", import.meta.url);
test("the page's counts match the benchmark's on every day and setting", { skip: !existsSync(DAYS) }, () => {
  const data = JSON.parse(readFileSync(DAYS, "utf8"));
  assert.equal(data.days.length, 4);
  for (const day of data.days) {
    for (const [key, v] of Object.entries(day.variants)) {
      const s = tally(day.truth, v.sets);
      assert.deepEqual(
        { covered: s.covered, singletons: s.singletons, singletons_right: s.singletonsRight },
        v.check,
        `day ${day.day}, ${key}`,
      );
    }
  }
});
