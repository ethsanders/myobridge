# API reference

Everything most programs need is importable from the top level:

```python
from myobridge import (
    EmgSimulator, Receiver,              # where samples come from
    EmgFilter, extract_features,         # signal processing
    LDAClassifier, calibrate, adapt,     # the model
    ConformalGestures,                   # gesture sets with a coverage promise
    ConfidenceGate, SetGate,             # when to act and when to hold
    SelectiveGestures,                   # answer only when sure, with an accuracy promise
    GesturePipeline, Prediction,         # live decoding
    QualityMonitor, QualityConfig,       # signal-quality checks
    Replay, Fault,                       # replaying recordings, injecting faults
)
```

The pages in this section are generated from the docstrings in
[`src/myobridge`](https://github.com/ethsanders/myobridge/tree/main/src/myobridge).

| Page | Modules | What's there |
|---|---|---|
| [Live decoding](decoding.md) | `pipeline`, `classifier`, `conformal`, `gating`, `selective`, `sessions`, `live_sets` | The streaming pipeline, LDA, calibration, recalibration, conformal gesture sets, decision policies, answering only when sure, and the across-day recipe on your own sessions |
| [Signal processing](signal.md) | `dsp`, `features`, `quality`, `faults` | Filters, feature sets, the quality monitor, simulated faults |
| [Streaming and replay](streaming.md) | `protocol`, `receiver`, `simulator`, `replay` | The UDP packet format, the receiver, the fake armband, replays |
| [Datasets and evaluation](evaluation.md) | `datasets`, `evaluation` | Pinned dataset loaders and leak-free scoring |
| [Firmware models](firmware.md) | `decimator`, `aliasing` | Python references for the firmware's decimation filter, and the aliasing model |
| [Hardware report](report.md) | `report` | Evidence from your own sessions: noise floor, hum, faults, decoding and the link |

The `myobridge` command is documented by `myobridge --help` and
[Getting started](../getting-started.md).
