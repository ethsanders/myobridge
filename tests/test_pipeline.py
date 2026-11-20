import numpy as np

from myobridge.classifier import LDAClassifier
from myobridge.pipeline import GesturePipeline


def test_pipeline_works_with_integer_labels():
    # Models trained by the benchmarks use integer labels; the live pipeline
    # used to assume strings and crashed when looking up the voted label.
    rng = np.random.default_rng(0)
    model = LDAClassifier().fit(rng.normal(size=(60, 20)), np.repeat([0, 1, 2], 20))
    pipe = GesturePipeline(model, fs=1000, n_channels=4)
    preds = [p for _ in range(20) for p in pipe.process_block(rng.normal(0, 50, (20, 4)))]
    assert preds and all(p.gesture in {"0", "1", "2"} for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
    assert np.all(np.diff([p.sample_index for p in preds]) == 50)
