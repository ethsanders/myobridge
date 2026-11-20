"""Live gesture sets from the armband, with an optional prompted, scored run.

`myobridge sets` uses this. Each window of the live stream gets its
conformal set: one gesture when the decoder is sure, two or more ("not
sure: fist or open") when it isn't. In prompted mode the app asks for one
gesture at a time and counts, window by window, how often the set held the
asked-for gesture, next to the promise. That count is the measured number.

The view is a line in the terminal and, with `serve()`, a page in the
browser at http://127.0.0.1:<port>/ in large type, for a screen or a camera.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .classifier import LDAClassifier
from .conformal import ConformalGestures
from .pipeline import GesturePipeline, Prediction
from .quality import QualityMonitor

GRACE_S = 1.5  # after each prompt, time to move into the gesture before windows count


class Prompter:
    """Asks for each gesture in turn, `hold_s` seconds each, in a fresh order every round.

    Windows in the first `grace_s` seconds of a prompt aren't scored: the
    person is still moving into the gesture.
    """

    def __init__(
        self, gestures: Sequence[str], hold_s: float = 4.0, grace_s: float = GRACE_S, seed: int = 0
    ):
        if hold_s <= grace_s:
            raise ValueError("each hold must be longer than the grace period")
        if len(gestures) < 2:
            raise ValueError("need at least two gestures to prompt")
        self.gestures, self.hold_s, self.grace_s = list(gestures), hold_s, grace_s
        self._rng = np.random.default_rng(seed)
        self._order: list[str] = []

    def at(self, t: float) -> tuple[str, bool, float]:
        """The gesture asked for at `t` seconds, whether windows count, seconds left."""
        slot = int(t // self.hold_s)
        while len(self._order) <= slot:
            round_ = [str(g) for g in self._rng.permutation(self.gestures)]
            # never ask for the same gesture twice in a row across rounds
            if self._order and round_[0] == self._order[-1]:
                round_[0], round_[-1] = round_[-1], round_[0]
            self._order += round_
        into = t - slot * self.hold_s
        return self._order[slot], into >= self.grace_s, self.hold_s - into


@dataclass
class Score:
    """Running counts of the scored windows."""

    windows: int = 0
    covered: int = 0  # sets that held the asked-for gesture
    singletons: int = 0
    singletons_right: int = 0

    def add(self, gesture_set: Sequence[str], truth: str) -> None:
        self.windows += 1
        self.covered += truth in gesture_set
        if len(gesture_set) == 1:
            self.singletons += 1
            self.singletons_right += gesture_set[0] == truth

    def as_dict(self) -> dict[str, Any]:
        def share(a: int, b: int) -> float | None:
            return a / b if b else None

        return {
            "windows": self.windows,
            "coverage": share(self.covered, self.windows),
            "singleton_rate": share(self.singletons, self.windows),
            "accuracy_when_singleton": share(self.singletons_right, self.singletons),
        }


class LiveSets:
    """Raw blocks in; the newest set, the prompt and the score out, as a dict.

    Args:
        sets: A calibration from `sessions.fit_sets` (after `recalibrate`, for
            a new day). Its model must record how it was trained.
        n_channels: Channels in the stream.
        prompter: Ask for gestures and score the sets, or None to just show them.
        monitor: Drop broken electrodes, as the live pipeline does.
    """

    def __init__(
        self,
        sets: ConformalGestures,
        n_channels: int,
        prompter: Prompter | None = None,
        monitor: QualityMonitor | None = None,
        info: dict[str, Any] | None = None,
    ):
        model = sets.model
        if not isinstance(model, LDAClassifier) or model.settings is None:
            raise ValueError("the sets' model must record how it was trained (use `fit_sets`)")
        self.pipeline = GesturePipeline.from_model(
            model, n_channels, conformal=sets, monitor=monitor
        )
        self.fs = float(model.settings.fs)
        self.prompter = prompter
        self.score = Score()
        self.promised = sets.coverage
        self.info = info or {}
        self.last: Prediction | None = None
        self._state: dict[str, Any] = self._snapshot(0.0)

    @property
    def t(self) -> float:
        return self.pipeline.samples_seen / self.fs

    def prompt(self) -> str | None:
        """The gesture being asked for right now (the next block's), if prompting."""
        return None if self.prompter is None else self.prompter.at(self.t)[0]

    def push(self, block: Any) -> dict[str, Any]:
        """Feed one packet's samples; returns the state after it."""
        for pred in self.pipeline.process_block(block):
            self.last = pred
            if self.prompter is not None and pred.gesture_set is not None:
                # score a window only if all of it lies after the grace period
                # of one prompt (prompts in a row always differ)
                end = pred.sample_index / self.fs
                start = (pred.sample_index - self.pipeline.windower.size) / self.fs
                truth, scored, _ = self.prompter.at(max(start, 0.0))
                if scored and self.prompter.at(end - 1e-9)[0] == truth:
                    self.score.add(pred.gesture_set, truth)
        self._state = self._snapshot(self.t)
        return self._state

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def _snapshot(self, t: float) -> dict[str, Any]:
        state: dict[str, Any] = {"t": round(t, 3), "promised": self.promised, **self.info}
        pred = self.last
        if pred is not None and pred.gesture_set is not None:
            # the decoder's own pick first, then the rest of the set
            members = sorted(pred.gesture_set, key=lambda g: g != pred.raw_gesture)
            state |= {"set": members, "top": pred.raw_gesture}
            state["dropped"] = list(pred.dropped_channels)
        if self.prompter is not None:
            asked, scored, left = self.prompter.at(t)
            state |= {"prompt": asked, "scoring": scored, "left": round(left, 2)}
            state["score"] = self.score.as_dict()
        return state


def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def describe(state: dict[str, Any]) -> str:
    """The state as one line of text, for the terminal."""
    parts = []
    if "prompt" in state:
        parts.append(f"hold {state['prompt']:<10} {state['left']:4.1f} s")
    members = state.get("set")
    if members is None:
        parts.append("waiting for the first window")
    elif len(members) == 1:
        parts.append(f"[ {members[0]} ]")
    elif not members:
        parts.append("no gesture fits")
    else:
        parts.append("not sure: " + " or ".join(members))
    if "score" in state:
        s = state["score"]
        parts.append(
            f"promised {pct(state['promised'], 0)}, delivered {pct(s['coverage'])} "
            f"of {s['windows']} windows, one gesture {pct(s['singleton_rate'], 0)}"
        )
    if state.get("dropped"):
        parts.append("dropped electrode " + ", ".join(str(c + 1) for c in state["dropped"]))
    return "  |  ".join(parts)


class StateServer:
    """Serves the newest state to a browser: the page at /, its JSON at /state."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.split("?")[0] == "/state":
                    with outer._lock:
                        body = json.dumps(outer._state).encode()
                    kind = "application/json"
                elif self.path.split("?")[0] in ("/", "/index.html"):
                    body, kind = PAGE.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # quiet
                pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host!s}:{port}/"

    def start(self) -> StateServer:
        self._thread.start()
        return self

    def update(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> StateServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>myobridge sets</title>
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df;
  --sure:#2f8f4e; --wrong:#c2412d; --unsure:#9aa5ad; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111110; --ink:#f2f1ee; --ink2:#a3a29e; --line:#2c2b29;
    --sure:#4fb873; --wrong:#e0634d; --unsure:#7b868e; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font: 18px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  min-height:100vh; display:flex; flex-direction:column; justify-content:center;
  padding: 4vh 5vw; gap: 5vh; }
.label { color:var(--ink2); text-transform:uppercase; letter-spacing:.08em; font-size:2.2vh; }
#prompt { font-size: 9vh; font-weight: 700; }
#left { color:var(--ink2); font-size: 4vh; margin-left: .4em; font-weight: 400; }
.chips { display:flex; flex-wrap:wrap; gap: 2vh; align-items:center; min-height: 13vh; }
.lead { font-size: 5vh; color: var(--ink2); font-weight: 600; }
.chip { font-size: 7vh; font-weight: 700; padding: .15em .6em; border-radius: 999px;
  border: .5vh solid var(--unsure); }
.chip.sure { background: var(--sure); border-color: var(--sure); color: var(--bg); }
.chip.wrong { background: var(--wrong); border-color: var(--wrong); color: var(--bg); }
.chip.hit { border-color: var(--sure); }
#score { font-size: 4vh; color: var(--ink2); display:flex; flex-wrap:wrap; gap: 0 1.2em; }
#score b { color: var(--ink); font-size: 6vh; }
#score .short b { color: var(--wrong); }
#foot { color: var(--ink2); font-size: 1.8vh; }
</style>
</head>
<body>
<div id="ask" hidden>
  <div class="label">Hold</div>
  <div><span id="prompt"></span><span id="left"></span></div>
</div>
<div><div class="label">Decoder's answer</div><div id="set" class="chips"></div></div>
<div id="score" hidden></div>
<div id="foot"></div>
<script>
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? "-" : (100 * x).toFixed(d) + "%");
function chip(text, cls) {
  const el = document.createElement("span");
  el.className = "chip " + cls;
  el.textContent = text;
  return el;
}
function draw(s) {
  const asking = "prompt" in s;
  $("ask").hidden = !asking;
  if (asking) {
    $("prompt").textContent = s.prompt.replaceAll("_", " ");
    $("left").textContent = s.scoring ? Math.ceil(s.left) + " s" : "get ready";
  }
  const box = $("set");
  const set = s.set;
  if (!set) {
    box.replaceChildren(chip("-", ""));
  } else if (set.length === 1) {
    const right = !asking || !s.scoring || set[0] === s.prompt;
    box.replaceChildren(chip(set[0].replaceAll("_", " "), right ? "sure" : "wrong"));
  } else if (set.length === 0) {
    box.replaceChildren(chip("no gesture fits", ""));
  } else {
    const lead = document.createElement("span");
    lead.className = "lead";
    lead.textContent = "not sure:";
    box.replaceChildren(lead, ...set.map((g) =>
      chip(g.replaceAll("_", " "), asking && g === s.prompt ? "hit" : "")));
  }
  if (s.score) {
    const sc = s.score;
    const short = sc.coverage != null && sc.coverage < s.promised - 0.05;
    $("score").hidden = false;
    $("score").innerHTML = "";
    for (const [label, value, cls] of [
      ["Promised", pct(s.promised, 0), ""],
      ["delivered so far", pct(sc.coverage), short ? "short" : ""],
      ["answered with one gesture", pct(sc.singleton_rate, 0), ""],
    ]) {
      const span = document.createElement("span");
      span.className = cls;
      span.append(label + " ");
      const b = document.createElement("b");
      b.textContent = value;
      span.append(b);
      $("score").append(span);
    }
  }
  const foot = [];
  if (s.score) foot.push(s.score.windows + " windows scored");
  if (s.dropped && s.dropped.length) {
    foot.push("electrode " + s.dropped.map((c) => c + 1).join(", ") + " dropped");
  }
  if (s.sessions) foot.push(s.sessions);
  $("foot").textContent = foot.join(" · ");
}
async function poll() {
  try {
    const r = await fetch("/state", { cache: "no-store" });
    if (r.ok) draw(await r.json());
  } catch (e) {}
  setTimeout(poll, 100);
}
poll();
</script>
</body>
</html>
"""
