"""myobridge: stream EMG from an ESP32 over UDP, clean it, and turn it into gestures.

The pieces most programs need are importable from the top level::

    from myobridge import EmgSimulator, GesturePipeline, calibrate

    sim = EmgSimulator(seed=0)
    model, _, _ = calibrate(sim.record, ["rest", "fist", "open"], seconds=5, fs=1000)
    pipeline = GesturePipeline.from_model(model, n_channels=4)
    for prediction in pipeline.process_block(sim.generate(1000)):
        print(prediction.gesture, prediction.confidence)

See https://ethsanders.github.io/myobridge/ for the guides and API reference.
"""

from .classifier import (
    LDAClassifier,
    MajorityVote,
    TrainingSettings,
    adapt,
    calibrate,
    recording_to_features,
)
from .conformal import ConformalGestures
from .dsp import EmgFilter
from .faults import Fault
from .features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    extract_features,
    feature_names,
)
from .gating import ConfidenceGate, DecisionPolicy, EvidenceGate, SetGate, VotePolicy
from .lsl import LslOutlets
from .pipeline import GesturePipeline, Prediction
from .protocol import Packet, decode_packet, encode_packet
from .quality import Issue, QualityConfig, QualityMonitor, QualityReport
from .receiver import Receiver
from .replay import Replay
from .sessions import Session, fit_sets, recalibrate
from .simulator import EmgSimulator

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_FEATURE_SET",
    "ConfidenceGate",
    "ConformalGestures",
    "DecisionPolicy",
    "FEATURE_SETS",
    "EmgFilter",
    "EmgSimulator",
    "EvidenceGate",
    "Fault",
    "GesturePipeline",
    "Issue",
    "LDAClassifier",
    "LslOutlets",
    "MajorityVote",
    "Packet",
    "Prediction",
    "QualityConfig",
    "QualityMonitor",
    "QualityReport",
    "Receiver",
    "Replay",
    "Session",
    "SetGate",
    "TrainingSettings",
    "VotePolicy",
    "__version__",
    "adapt",
    "calibrate",
    "decode_packet",
    "encode_packet",
    "extract_features",
    "feature_names",
    "fit_sets",
    "recalibrate",
    "recording_to_features",
]
