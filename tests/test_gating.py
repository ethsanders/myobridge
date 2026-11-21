"""Decision policies on hand-made streams."""

from __future__ import annotations

import numpy as np
import pytest

from myobridge import (
    ConfidenceGate,
    ConformalGestures,
    EmgSimulator,
    EvidenceGate,
    GesturePipeline,
    MajorityVote,
    SetGate,
    VotePolicy,
    calibrate,
)
from myobridge.gating import HOLD

K = 3  # classes in the hand-made streams


def onehot(labels, p=0.9):
    """Probabilities with `p` on each label and the rest spread evenly."""
    out = np.full((len(labels), K), (1 - p) / (K - 1))
    out[np.arange(len(labels)), labels] = p
    return out


def sets_of(rows):
    """Conformal sets from lists of members, e.g. [[0], [0, 1], []]."""
    out = np.zeros((len(rows), K), dtype=bool)
    for i, r in enumerate(rows):
        out[i, r] = True
    return out


def test_vote_matches_majority_vote():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, K, 300)
    for k in (1, 2, 4, 5):
        mv = MajorityVote(k)
        expected = [int(mv.update(str(lab))) for lab in labels]
        assert VotePolicy(k).run(onehot(labels)).tolist() == expected


def test_vote_never_holds_and_lags_by_half_its_length():
    d = VotePolicy(5).run(onehot([0] * 6 + [1] * 6))
    assert HOLD not in d
    assert d[6:].tolist() == [0, 0, 1, 1, 1, 1]  # the new gesture needs 3 of 5


def test_confidence_gate_holds_unsure_windows():
    p = onehot([0, 0, 0, 0])
    p[2] = [0.5, 0.3, 0.2]  # unsure, though still gesture 0
    assert ConfidenceGate(0.6, k=1).run(p).tolist() == [0, 0, HOLD, 0]
    assert ConfidenceGate(0.6, k=1, latch=True).run(p).tolist() == [0, 0, 0, 0]
    assert ConfidenceGate(0.0, k=1).run(p).tolist() == [0, 0, 0, 0]


def test_confidence_gate_votes_on_every_window():
    p = onehot([1, 1, 0, 0, 0])
    p[2] = [0.4, 0.35, 0.25]  # held, but it still counts in the vote
    assert ConfidenceGate(0.6, k=3).run(p).tolist() == [1, 1, HOLD, 0, 0]


def test_latch_switches_only_on_a_sure_new_gesture():
    p = onehot([0, 0, 1, 1, 1], p=0.95)
    p[2] = [0.3, 0.5, 0.2]  # the first window of gesture 1 is unsure
    assert ConfidenceGate(0.8, k=1, latch=True).run(p).tolist() == [0, 0, 0, 1, 1]
    # before any action there is nothing to latch on to
    first = onehot([1, 1], p=0.5)
    assert ConfidenceGate(0.8, k=1, latch=True).run(first).tolist() == [HOLD, HOLD]


def test_set_gate_all_and_k_of_n():
    sets = sets_of([[0], [0], [0, 1], [0], [0], [1], [1], [1]])
    p = onehot([0] * 8)  # SetGate ignores the probabilities
    assert SetGate(2).run(p, sets).tolist() == [HOLD, 0, HOLD, HOLD, 0, HOLD, 1, 1]
    assert SetGate(3, 2).run(p, sets).tolist() == [HOLD, 0, 0, 0, 0, 0, 1, 1]
    assert SetGate(1, latch=True).run(p, sets).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_set_gate_empty_sets_and_ties():
    p = onehot([0] * 4)
    assert SetGate(1).run(p, sets_of([[], [2], [], [0, 1, 2]])).tolist() == [HOLD, 2, HOLD, HOLD]
    # two gestures qualify at once: the newest wins
    assert SetGate(2, 1).run(p[:2], sets_of([[0], [1]])).tolist() == [0, 1]


def test_set_gate_needs_sets():
    with pytest.raises(ValueError, match="conformal set"):
        SetGate(2).step([0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        SetGate(2, 3)


def test_evidence_gate_accumulates_and_forgets():
    p = onehot([0] * 5 + [1] * 8, p=0.5)  # each window alone is unsure
    d = EvidenceGate(0.8, memory=0.8).run(p)
    assert d[:4].tolist() == [HOLD] * 4  # one window at p=0.5 is not enough
    assert d[4] == 0  # five in a row are
    assert d[5:10].tolist() == [HOLD] * 5  # it holds while the evidence turns
    assert d[10:].tolist() == [1, 1, 1]  # and the leak lets it change its mind
    # without the leak, the old evidence never fades
    assert 1 not in EvidenceGate(0.8, memory=1.0).run(p)[:10]
    # with no memory it is a cut-off on the window's own probability
    assert EvidenceGate(0.45, memory=0.0).run(p).tolist() == p.argmax(1).tolist()


def test_reset_clears_state():
    gate = ConfidenceGate(0.6, k=5, latch=True)
    gate.run(onehot([0] * 5))
    assert gate.step([0.4, 0.3, 0.3]) == 0  # latched
    gate.reset()
    assert gate.step([0.4, 0.3, 0.3]) is None


def test_bad_arguments():
    for make in (
        lambda: ConfidenceGate(1.5),
        lambda: EvidenceGate(1.0),
        lambda: EvidenceGate(0.9, memory=2.0),
        lambda: VotePolicy(0),
    ):
        with pytest.raises(ValueError):
            make()
    with pytest.raises(ValueError, match="same shape"):
        SetGate(1).run(onehot([0, 1]), sets_of([[0]]))


@pytest.fixture(scope="module")
def model():
    sim = EmgSimulator(seed=0)
    m, X, y = calibrate(sim.record, ["rest", "fist", "open"], seconds=4, fs=1000)
    return m, X, y, sim


def test_pipeline_without_policy_acts_on_the_vote(model):
    m, _, _, sim = model
    pipe = GesturePipeline.from_model(m, n_channels=4)
    preds = pipe.process_block(sim.generate(1000))
    assert preds and all(p.action == p.gesture for p in preds)


def test_pipeline_with_policies(model):
    m, X, y, sim = model
    sets = ConformalGestures.calibrate(m, X, y, alpha=0.1, temperature=5.0)
    block = sim.generate(1000)
    with pytest.raises(ValueError, match="needs conformal sets"):
        GesturePipeline.from_model(m, n_channels=4, policy=SetGate(2))
    for policy in (ConfidenceGate(0.99, k=1), SetGate(2), EvidenceGate(0.9, latch=True)):
        # each pipeline gets its own copy: the APS random share is a stream
        pipe = GesturePipeline.from_model(
            m, n_channels=4, conformal=sets.with_model(m), policy=policy
        )
        plain = GesturePipeline.from_model(m, n_channels=4, conformal=sets.with_model(m))
        preds, ref = pipe.process_block(block), plain.process_block(block)
        assert [p.gesture for p in preds] == [p.gesture for p in ref]  # vote unchanged
        assert [p.gesture_set for p in preds] == [p.gesture_set for p in ref]
        labels = {str(c) for c in m.classes}
        assert all(p.action is None or p.action in labels for p in preds)
    # the confidence gate sees the temperature-scaled probabilities
    gate = ConfidenceGate(0.99, k=1)
    pipe = GesturePipeline.from_model(m, n_channels=4, conformal=sets, policy=gate)
    preds = pipe.process_block(block)
    expected_held = sets.predict_proba(_features(pipe, block)).max(axis=1) < 0.99
    assert [p.action is None for p in preds] == expected_held.tolist()


def _features(pipe, block):
    """The feature rows a fresh copy of `pipe` computes for `block`."""
    from myobridge.dsp import EmgFilter
    from myobridge.features import StreamingWindower, extract_features

    f = EmgFilter(1000, 4, mains_hz=60.0)
    w = StreamingWindower(pipe.windower.size, pipe.windower.step, 4)
    windows = w.push(f.process(block).clean)
    return extract_features(np.stack(windows), *pipe.thresholds, pipe.feature_set)
