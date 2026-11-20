"""Calibrate on a simulated armband, then decode a stream from it. No data or hardware needed.

python examples/decode_simulated.py
"""

from collections import Counter

from myobridge import EmgSimulator, GesturePipeline, calibrate

GESTURES = ["rest", "fist", "open", "flex", "extend"]

# 1. Calibration: hold each gesture for 5 seconds. With the real armband,
#    `record` would come from a Receiver instead (see `myobridge calibrate`).
arm = EmgSimulator(seed=0)
model, X, y = calibrate(arm.record, GESTURES, seconds=5, fs=arm.fs)
print(f"trained on {len(X)} windows of {X.shape[1]} features each")

# 2. Live use: feed 20-sample blocks, the size of one packet from the armband.
pipeline = GesturePipeline.from_model(model, n_channels=arm.n_channels)
for gesture in ["fist", "rest", "extend"]:
    arm.set_gesture(gesture)
    decided = Counter()
    for _ in range(100):  # 2 seconds
        for prediction in pipeline.process_block(arm.generate(20)):
            decided[prediction.gesture] += 1
    top, count = decided.most_common(1)[0]
    print(f"holding {gesture:>6}: decoded {top} in {count} of {sum(decided.values())} decisions")
