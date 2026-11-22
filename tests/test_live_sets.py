import json
import urllib.request

import pytest

from myobridge.gating import ConfidenceGate
from myobridge.live_sets import LiveSets, Prompter, Score, StateServer, describe
from myobridge.sessions import Session, fit_sets, recalibrate
from myobridge.simulator import GESTURES, EmgSimulator


@pytest.fixture(scope="module")
def sets():
    def session(seed, seconds=5.0):
        s = EmgSimulator(seed=seed)
        return Session(1000.0, GESTURES, tuple(s.record(g, seconds) for g in GESTURES))

    return recalibrate(fit_sets([session(i) for i in range(3)]).sets, session(9, 2.0))


def test_prompter_asks_for_each_gesture_every_round():
    p = Prompter(GESTURES, hold_s=2.0, grace_s=0.5, seed=1)
    asked = [p.at(i * 2.0 + 0.1)[0] for i in range(3 * len(GESTURES))]
    for r in range(3):
        assert sorted(asked[r * 5 : r * 5 + 5]) == sorted(GESTURES)
    assert all(a != b for a, b in zip(asked, asked[1:], strict=False))
    assert p.at(0.4)[1] is False and p.at(0.6)[1] is True
    assert p.at(1.5)[2] == pytest.approx(0.5)


def test_prompter_needs_a_hold_longer_than_the_grace():
    with pytest.raises(ValueError):
        Prompter(GESTURES, hold_s=1.0, grace_s=1.5)


def test_score_counts():
    s = Score()
    s.add(("fist",), "fist")
    s.add(("fist", "open"), "open")
    s.add(("open",), "fist")
    s.add((), "rest")
    d = s.as_dict()
    assert d["windows"] == 4
    assert d["coverage"] == 0.5
    assert d["singleton_rate"] == 0.5
    assert d["accuracy_when_singleton"] == 0.5


def test_score_counts_actions_and_runs_of_wrong_ones():
    s = Score(step_s=0.5)  # 120 windows a minute
    for w, (action, truth) in enumerate(
        [("fist", "fist"), (None, "fist"), ("open", "fist"), ("open", "fist"), ("fist", "fist")]
    ):
        s.add((truth,), truth)
        s.act(action, truth, w)
    s.add(("open",), "open")
    s.act("fist", "open", 9)  # not next to the last scored window: a new run
    s.add(("open",), "open")
    s.act("fist", "open", 10)  # carries on that run
    d = s.as_dict()
    assert d["acting_share"] == 6 / 7
    assert d["right_when_acting"] == 2 / 6
    # two runs of wrong actions over 7 windows of 0.5 s
    assert d["wrong_per_min"] == pytest.approx(2 / (7 * 0.5 / 60))


def test_a_score_without_a_policy_has_no_action_counts():
    assert "wrong_per_min" not in Score().as_dict()


def test_gated_live_sets_report_the_action(sets):
    sim = EmgSimulator(seed=31)
    gate = ConfidenceGate(0.7, k=1, latch=True)
    live = LiveSets(sets, 4, Prompter(GESTURES, hold_s=3.0, seed=3), policy=gate)
    actions = set()
    while live.t < 30.0:
        sim.set_gesture(live.prompt())
        state = live.push(sim.generate(20))
        actions.add(state.get("action"))
    assert "action" in state and actions - {None} <= set(GESTURES)
    score = state["score"]
    assert 0.5 < score["acting_share"] <= 1.0
    assert 0.75 < score["right_when_acting"] <= 1.0
    assert score["wrong_per_min"] >= 0
    line = describe(state)
    assert "acting: " in line or "holding" in line
    assert "wrong actions/min" in line


def test_ungated_live_sets_have_no_action(sets):
    state = LiveSets(sets, 4).push(EmgSimulator(seed=1).generate(400))
    assert "action" not in state


@pytest.mark.parametrize(("action", "text"), [("fist", "acting: fist"), (None, "holding")])
def test_describe_actions(action, text):
    assert text in describe({"set": ["fist"], "promised": 0.9, "action": action})


def test_prompted_live_sets_keep_score(sets):
    sim = EmgSimulator(seed=30)
    live = LiveSets(sets, 4, Prompter(GESTURES, hold_s=3.0, seed=2), info={"sessions": "x"})
    while live.t < 30.0:
        sim.set_gesture(live.prompt())
        state = live.push(sim.generate(20))
    score = state["score"]
    # 10 prompts of 3 s, scored from 1.5 s: about 10 x 1.3 s / 0.05 s windows
    assert 200 < score["windows"] < 300
    assert 0.75 < score["coverage"] <= 1.0
    assert state["prompt"] in GESTURES and state["sessions"] == "x"
    assert state["set"][0] == state["top"] or state["set"] == []
    assert "promised 90%, delivered" in describe(state)


def test_unprompted_live_sets_have_no_score(sets):
    live = LiveSets(sets, 4)
    state = live.push(EmgSimulator(seed=1).generate(400))
    assert "score" not in state and "prompt" not in state
    assert "set" in state


@pytest.mark.parametrize(
    ("members", "text"),
    [(["fist"], "[ fist ]"), (["fist", "open"], "not sure: fist or open"), ([], "no gesture")],
)
def test_describe_sets(members, text):
    assert text in describe({"set": members, "promised": 0.9})


def test_state_server_serves_the_page_and_the_state():
    with StateServer(port=0) as server:
        server.update({"set": ["fist"], "promised": 0.9})
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "Decoder's answer" in page
        got = json.loads(urllib.request.urlopen(server.url + "state", timeout=5).read())
        assert got == {"set": ["fist"], "promised": 0.9}
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(server.url + "nope", timeout=5)


def test_a_lifted_electrode_is_dropped(sets):
    from myobridge import faults
    from myobridge.cli import quality_monitor
    from myobridge.sessions import Session

    sim = EmgSimulator(seed=40)
    today = Session(1000.0, GESTURES, tuple(sim.record(g, 2.0) for g in GESTURES))
    live = LiveSets(sets, 4, monitor=quality_monitor(1000.0, 4, 60.0, today, rail=None))
    sim.set_gesture("fist")
    raw = sim.generate(3000).astype(float)
    broken = raw.copy()
    broken[1000:, 0] = faults.lead_off(raw[1000:, 0])
    for i in range(0, len(broken), 20):
        state = live.push(broken[i : i + 20])
    assert state["dropped"] == [0]
