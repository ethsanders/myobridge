// The demo page: load Pyodide and myobridge, fetch one person's recordings, replay them.
import {
  EVEN_4,
  ISSUE_TEXT,
  N_PEOPLE,
  Ring,
  channelState,
  mapLimited,
  parseRecording,
  plan,
  prettyGesture,
  traceScale,
} from "./core.js";

const $ = (id) => document.getElementById(id);
const STEP_S = 0.1; // seconds of data per animation tick at 1x
const HISTORY_S = 4; // seconds of trace shown

let py = null; // the Pyodide instance, loaded once
let bridge = null;
let session = null; // { info, rings, scales, playing, speed, person }

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function setProgress(fraction) {
  const bar = $("progress");
  bar.hidden = fraction === null;
  if (fraction !== null) bar.firstElementChild.style.width = `${Math.round(fraction * 100)}%`;
}

async function loadPython() {
  if (bridge) return bridge;
  setStatus("Starting Python...");
  py = await loadPyodide();
  setStatus("Loading NumPy and SciPy...");
  await py.loadPackage(["numpy", "scipy", "micropip"]);
  setStatus("Installing myobridge...");
  const { file } = await (await fetch("wheel.json")).json();
  py.globals.set("WHEEL_URL", new URL(file, location.href).href);
  await py.runPythonAsync("import micropip\nawait micropip.install(WHEEL_URL, deps=False)");
  const source = await (await fetch("bridge.py")).text();
  py.FS.writeFile("/home/pyodide/bridge.py", source);
  py.runPython("import sys\nsys.path.insert(0, '/home/pyodide')");
  bridge = py.pyimport("bridge");
  return bridge;
}

async function fetchRecordings(items, done, total) {
  return mapLimited(
    items,
    6,
    async (item) => {
      const response = await fetch(item.url);
      if (!response.ok) throw new Error(`${response.status} for ${item.url}`);
      return { ...item, values: parseRecording(await response.text()) };
    },
    (n) => {
      setStatus(`Downloading recordings: ${done + n} of ${total}`);
      setProgress((done + n) / total);
    },
  );
}

function queue(part, list) {
  for (const r of list) bridge.add(part, r.values, r.label, r.session, r.rep);
}

// Call into Python and get plain JavaScript back, freeing the proxy.
function call(fn, ...args) {
  const proxy = fn(...args);
  const value = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return value;
}

async function start() {
  $("start").disabled = true;
  stop();
  try {
    await loadPython();
    const person = Number($("person").value);
    const { train, test } = plan(person);
    const total = train.length + test.length;
    const trainData = await fetchRecordings(train, 0, total);
    const testData = await fetchRecordings(test, train.length, total);
    setProgress(null);
    setStatus("Calibrating on the first 2 cycles...");
    await new Promise((r) => setTimeout(r, 20)); // let the status paint before Python blocks
    const channels = $("channels").value === "4" ? EVEN_4 : null;
    queue("train", trainData);
    queue("test", testData);
    const info = call(bridge.start, channels);
    buildLanes(info, channels);
    session = {
      info,
      rings: Array.from({ length: info.n_channels }, () => new Ring(HISTORY_S * info.display_fs)),
      // Fixed per-channel scales from the calibration gestures, so a fault looks
      // like one (an electrode coming off goes flat instead of being zoomed back up).
      scales: info.trace_scale.map((scale, c) => ({ median: info.trace_centre[c], scale })),
      playing: true,
      speed: Number($("speed").value),
      person,
    };
    $("model").textContent = `LDA, ${info.features} features`;
    $("live").hidden = false;
    setStatus(`Person ${person}: replaying ${info.duration_s.toFixed(0)} s they did after calibrating.`);
    $("play").textContent = "Pause";
    lastTime = 0;
    const id = ++loop;
    requestAnimationFrame((t) => tick(t, id));
  } catch (err) {
    console.error(err);
    setProgress(null);
    setStatus(`Couldn't start the demo: ${err.message}`, true);
  } finally {
    $("start").disabled = false;
  }
}

function stop() {
  session = null;
}

function buildLanes(info, channels) {
  const lanes = $("lanes");
  lanes.replaceChildren();
  const template = $("lane-template");
  for (let c = 0; c < info.n_channels; c++) {
    const lane = template.content.firstElementChild.cloneNode(true);
    const physical = channels ? channels[c] : c;
    lane.querySelector(".name").textContent = `ch ${physical + 1}`;
    const select = lane.querySelector(".fault");
    select.setAttribute("aria-label", `Fault on channel ${physical + 1}`);
    select.addEventListener("change", () => bridge.inject(c, select.value || null));
    lanes.append(lane);
  }
}

let lastTime = 0;
let loop = 0; // bumps on every start, so an old animation loop stops
function tick(now, id) {
  if (!session || id !== loop) return;
  const dt = lastTime ? Math.min((now - lastTime) / 1000, 0.25) : 0;
  lastTime = now;
  if (session.playing && dt > 0) {
    const frame = call(bridge.step, dt * session.speed);
    absorb(frame);
    render(frame);
    if (frame.done) {
      session.playing = false;
      $("play").textContent = "Play";
      setStatus(
        `Done: ${(100 * frame.accuracy).toFixed(1)}% of decisions matched the gesture being held.`,
      );
    }
  }
  requestAnimationFrame((t) => tick(t, id));
}

function absorb(frame) {
  const n = session.info.n_channels;
  for (let i = 0; i < frame.n_raw; i++) {
    for (let c = 0; c < n; c++) session.rings[c].push(frame.raw[i * n + c]);
  }
}

function render(frame) {
  $("truth").textContent = prettyGesture(frame.truth);
  $("prediction").textContent = prettyGesture(frame.prediction);
  const match = frame.prediction && frame.prediction === frame.truth;
  $("prediction").classList.toggle("match", Boolean(match));
  $("prediction").classList.toggle("miss", Boolean(frame.prediction) && !match);
  $("confidence").style.width = `${Math.round(frame.confidence * 100)}%`;
  $("confidence-text").textContent = frame.prediction ? `${Math.round(frame.confidence * 100)}%` : "-";
  $("accuracy").textContent = frame.accuracy == null ? "-" : `${(100 * frame.accuracy).toFixed(1)}%`;
  $("clock").textContent = `${frame.t.toFixed(1)} s`;

  const lanes = $("lanes").children;
  frame.channels.forEach((ch, c) => {
    const lane = lanes[c];
    const dot = lane.querySelector(".dot");
    dot.className = `dot ${channelState(ch)}`;
    const issues = ch.issues.map((i) => ISSUE_TEXT[i] ?? i).join(", ");
    lane.querySelector(".issue").textContent = ch.dropped ? `${issues}: dropped` : issues;
    const values = session.rings[c].toArray();
    drawTrace(lane.querySelector("canvas"), values, channelState(ch), session.scales[c]);
  });
}

function drawTrace(canvas, values, state, fixed) {
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
  if (values.length < 2) return;
  const { median, scale } = fixed ?? traceScale(values);
  const style = getComputedStyle(canvas);
  ctx.strokeStyle = style.getPropertyValue(state === "ok" ? "--trace" : `--trace-${state}`);
  ctx.lineWidth = 1;
  ctx.beginPath();
  const capacity = HISTORY_S * session.info.display_fs;
  const x0 = width * (1 - values.length / capacity);
  for (let i = 0; i < values.length; i++) {
    const x = x0 + (i / capacity) * width;
    const y = height / 2 - ((values[i] - median) / scale) * (height * 0.45);
    const yc = Math.max(1, Math.min(height - 1, y));
    if (i === 0) ctx.moveTo(x, yc);
    else ctx.lineTo(x, yc);
  }
  ctx.stroke();
}

function init() {
  const person = $("person");
  for (let p = 1; p <= N_PEOPLE; p++) {
    const option = new Option(`Person ${p}`, String(p));
    person.append(option);
  }
  person.value = "7"; // the first person the benchmarks report (1-6 were used for tuning)
  $("start").addEventListener("click", start);
  $("play").addEventListener("click", () => {
    if (!session) return;
    session.playing = !session.playing;
    $("play").textContent = session.playing ? "Pause" : "Play";
  });
  $("speed").addEventListener("change", () => {
    if (session) session.speed = Number($("speed").value);
  });
  $("restart").addEventListener("click", start);
}

init();
