"""Decode a real person's forearm from the 3DC dataset, and pull an electrode off half-way.

    myobridge fetch 3dc                # once, about 1.3 GB
    python examples/replay_3dc.py [--person 7]

This is what the browser demo runs, without the browser.
"""

import argparse

from myobridge import Replay, datasets
from myobridge.datasets import threedc

parser = argparse.ArgumentParser()
parser.add_argument("--person", type=int, default=7, help="3DC participant, 1-22")
parser.add_argument("--data", default=None, help="dataset folder (default: $MYOBRIDGE_DATA)")
args = parser.parse_args()

ds = datasets.load("3dc", root=args.data, subjects=[args.person])
train = ds.select(sessions=[0], reps=[0, 1])  # two passes through the 11 gestures
test = ds.select(sessions=[1], reps=[0])  # a later pass the model never saw

replay = Replay(
    train,
    test,
    ds.gestures,
    mains_hz=ds.mains_hz,
    channels=[0, 2, 5, 7],  # four electrodes spread around the forearm, like myobridge
    rail=threedc.ADC_RAIL,  # lets the monitor spot clipping
)
print(f"replaying {replay.duration_s:.0f} s from person {args.person}")

broken = False
while not replay.done:
    if not broken and replay.position >= len(replay.stream) // 2:
        replay.inject(1, "lead_off")  # the second electrode comes off, from now on
        print("-- electrode 2 pulled off --")
        broken = True
    frame = replay.step(1.0)
    dropped = [c.channel for c in frame.channels if c.dropped]
    note = f"  (ignoring channel {dropped})" if dropped else ""
    print(f"{frame.t:5.1f} s  holding {frame.truth:<20} decoded {frame.prediction}{note}")

print(f"{100 * frame.accuracy:.1f}% of decisions were right")
