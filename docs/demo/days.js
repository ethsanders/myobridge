// The first part of the demo: decoded gesture sets on four MultiDay days, played back.
// No Python here, so it starts at once; every number it shows is counted from days.json.
import {
  countUpTo,
  orderedSet,
  pct,
  prettyGesture,
  segmentAt,
  tally,
  variantKey,
  windowKind,
} from "./core.js";

const $ = (id) => document.getElementById(id);

const state = {
  data: null,
  day: 0, // index into data.days
  method: "myobridge",
  recalibrate: true,
  electrode: "ok",
  t: 0, // seconds into the day's replay
  playing: true,
  speed: 2,
  last: 0,
};

function current() {
  const day = state.data.days[state.day];
  const key = variantKey(state.recalibrate, state.method, state.electrode);
  return { day, key, variant: day.variants[key] };
}

function duration(day) {
  const last = day.segments.at(-1);
  return last.start + last.length;
}

function chip(text, kind) {
  const el = document.createElement("span");
  el.className = `chip ${kind}`;
  el.textContent = text;
  return el;
}

function renderSet(day, variant, i) {
  const box = $("d-set");
  if (i < 0) {
    box.replaceChildren(chip("-", "idle"));
    return;
  }
  const k = state.data.gestures.length;
  const members = orderedSet(variant.sets[i], variant.top[i], k);
  const kind = windowKind(variant.sets[i], day.truth[i]);
  const names = members.map((g) => prettyGesture(state.data.gestures[g]));
  if (members.length === 1) {
    box.replaceChildren(chip(names[0], kind === "sure" ? "sure" : "wrong"));
  } else if (members.length === 0) {
    box.replaceChildren(chip("no gesture fits", "missed"));
  } else {
    const lead = document.createElement("span");
    lead.className = "chips-lead";
    lead.textContent = "not sure:";
    const truth = day.truth[i];
    box.replaceChildren(
      lead,
      ...members.map((g, j) => chip(names[j], g === truth ? "unsure hit" : "unsure")),
    );
  }
}

function renderScore(day, variant, n) {
  const s = tally(day.truth, variant.sets, n);
  const promised = state.data.headline.promised;
  $("scoreboard").replaceChildren(
    ...[
      ["Promised", pct(promised, 0)],
      ["delivered so far", pct(s.coverage)],
      ["answered with one gesture", pct(s.singletonRate, 0)],
      ["right when it did", pct(s.accuracyWhenSingleton, 0)],
    ].map(([label, value], j) => {
      const span = document.createElement("span");
      span.className = j === 1 ? "score main" : "score";
      span.innerHTML = `<span class="score-label"></span> <b></b>`;
      span.firstElementChild.textContent = label;
      span.lastElementChild.textContent = value;
      if (j === 1 && s.coverage != null) {
        span.classList.toggle("short", s.coverage < promised - 0.05);
      }
      return span;
    }),
  );
}

function renderBenchmark() {
  const h = state.data.headline;
  const key = `${state.recalibrate ? "adapt" : "static"}-${state.method}`;
  const c = h.multiday[key];
  let text =
    `Over all ${h.test_days} test days, this setting delivers ${pct(c.mean)} ` +
    `[${pct(c.low)}, ${pct(c.high)}] against the ${pct(h.promised, 0)} promise ` +
    `(from myobridge bench conformal).`;
  if (state.electrode !== "ok") {
    const f = h.threedc_lead_off;
    text +=
      ` In the benchmark's electrode test (3DC, ${f.people} people), the sets deliver ` +
      `${pct(f.no_monitor)} without the monitor and ${pct(f.monitor_shared_threshold)} with it.`;
  }
  $("benchmark").textContent = text;
}

function drawStrip(day, variant, n) {
  const canvas = $("strip");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const style = getComputedStyle(canvas);
  const color = (name) => style.getPropertyValue(`--${name}`).trim();
  const total = duration(day);
  const x = (t) => (t / total) * width;
  const barTop = 14;
  const barH = height - barTop - 2;
  // gesture boundaries
  ctx.fillStyle = color("line");
  for (const s of day.segments) ctx.fillRect(Math.round(x(s.start)), 0, 1, height);
  if (state.electrode !== "ok") {
    ctx.fillStyle = color("fault");
    ctx.fillRect(x(day.fault_from), 0, width - x(day.fault_from), 4);
  }
  const w = Math.max(1, width / day.t.length);
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = color(windowKind(variant.sets[i], day.truth[i]));
    ctx.fillRect(x(day.t[i]) - w, barTop, Math.ceil(w), barH);
  }
  // playhead
  ctx.fillStyle = color("ink");
  ctx.fillRect(Math.min(width - 2, x(state.t)), barTop - 4, 2, barH + 4);
}

function render() {
  const { day, variant } = current();
  const n = countUpTo(day.t, state.t);
  $("d-truth").textContent = prettyGesture(state.data.gestures[segmentAt(day.segments, state.t).gesture]);
  renderSet(day, variant, n - 1);
  renderScore(day, variant, n);
  drawStrip(day, variant, n);
}

function tick(now) {
  const dt = state.last ? Math.min((now - state.last) / 1000, 0.25) : 0;
  state.last = now;
  if (state.playing && state.data) {
    const day = current().day;
    state.t = Math.min(duration(day), state.t + dt * state.speed);
    if (state.t >= duration(day)) {
      state.playing = false;
      $("d-play").textContent = "Play";
    }
    render();
  }
  requestAnimationFrame(tick);
}

function pickDay(i) {
  state.day = i;
  state.t = 0;
  state.playing = true;
  $("d-play").textContent = "Pause";
  for (const b of $("day-buttons").children) {
    b.setAttribute("aria-pressed", String(Number(b.dataset.index) === i));
  }
  render();
}

function wire() {
  state.data.days.forEach((d, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = String(d.day);
    b.dataset.index = String(i);
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => pickDay(i));
    $("day-buttons").append(b);
  });
  // Each method comes with its own recalibration setting (the textbook recipe
  // doesn't recalibrate); the checkbox can still change it afterwards.
  for (const b of document.querySelectorAll("[data-method]")) {
    b.addEventListener("click", () => {
      state.method = b.dataset.method;
      state.recalibrate = state.method === "myobridge";
      $("recal").checked = state.recalibrate;
      for (const o of document.querySelectorAll("[data-method]")) {
        o.setAttribute("aria-pressed", String(o === b));
      }
      renderBenchmark();
      if (!state.playing) render();
    });
  }
  $("recal").addEventListener("change", (e) => {
    state.recalibrate = e.target.checked;
    renderBenchmark();
    if (!state.playing) render();
  });
  $("electrode").addEventListener("change", (e) => {
    state.electrode = e.target.value;
    renderBenchmark();
    if (!state.playing) render();
  });
  $("d-play").addEventListener("click", () => {
    if (!state.playing && state.t >= duration(current().day)) state.t = 0;
    state.playing = !state.playing;
    $("d-play").textContent = state.playing ? "Pause" : "Play";
  });
  $("d-speed").addEventListener("change", (e) => {
    state.speed = Number(e.target.value);
  });
  $("d-restart").addEventListener("click", () => pickDay(state.day));
  window.addEventListener("resize", () => render());
}

// Settings from the address, for sharing a view: ?day=91&method=textbook&recal=0&electrode=off
function fromQuery() {
  const q = new URLSearchParams(location.search);
  state.day = Math.max(0, state.data.days.findIndex((d) => d.day === state.data.typical_day));
  const day = state.data.days.findIndex((d) => String(d.day) === q.get("day"));
  if (day >= 0) state.day = day;
  if (q.get("method") === "textbook") {
    state.method = "textbook";
    state.recalibrate = false;
  }
  if (q.get("recal") === "0") state.recalibrate = false;
  if (["off", "off_monitor"].includes(q.get("electrode"))) state.electrode = q.get("electrode");
  if (q.has("speed")) state.speed = Number(q.get("speed")) || state.speed;
  $("recal").checked = state.recalibrate;
  $("electrode").value = state.electrode;
  $("d-speed").value = String(state.speed);
  for (const b of document.querySelectorAll("[data-method]")) {
    b.setAttribute("aria-pressed", String(b.dataset.method === state.method));
  }
}

async function main() {
  try {
    const response = await fetch("days.json");
    if (!response.ok) throw new Error(`${response.status} loading days.json`);
    state.data = await response.json();
  } catch (err) {
    console.error(err);
    $("scoreboard").textContent = `Couldn't load the decoded days: ${err.message}`;
    return;
  }
  wire();
  fromQuery();
  pickDay(state.day);
  renderBenchmark();
  requestAnimationFrame(tick);
}

main();
