"""Does sampling at 1 kHz behind a first-order filter hurt? Measured, then modelled.

Part 1 uses MultiDay (recorded at 2048 Hz) to measure how much real forearm
EMG power sits above 512 Hz, then emulates a 1024 Hz system four ways and
re-runs the classifier on each. Part 2 is the white-noise model from
myobridge.aliasing, which covers what the recordings can't: wideband
electronic noise above the recorder's own anti-alias filter.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .. import datasets
from ..aliasing import (
    band_power_fraction,
    decimate_chain,
    folded_noise_ratio,
    folded_noise_ratio_decimated,
    ratio_to_db,
)
from ..classifier import LDAClassifier
from ..datasets.core import Dataset
from ..decimator import HALFBAND_Q15
from ..evaluation import FeatureConfig, accuracy, featurize, fit_thresholds, prefilter, summarize
from .common import RunOptions, provenance, timed, write_result
from .multiday_bench import TEST_DAYS, TEST_START_S, TRAIN_DAYS

CHAINS = ("ideal", "rc", "rc_ideal", "none")
MYOWARE_CUTOFF_HZ = 498.0  # upper -3 dB edge of the MyoWare 2.0 RAW output
NOISE_ORDERS = (1, 2, 4)
NOISE_RATES = (1000.0, 2000.0, 4000.0)


def spectrum(ds: Dataset) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, active in (("active", True), ("rest", False)):
        recs = [r for r in ds.recordings if (r.label != 0) == active]
        fracs = np.array([band_power_fraction(r.emg, ds.fs, 512.0) for r in recs])
        out[name] = {
            "median_fraction_above_512hz_per_channel": np.median(fracs, axis=0),
            "p95_fraction_above_512hz_per_channel": np.percentile(fracs, 95, axis=0),
            "n_recordings": len(recs),
        }
    return out


def emulated(ds: Dataset, days: list[int]) -> dict[str, Any]:
    cfg = FeatureConfig()
    out: dict[str, Any] = {}
    reference = None
    for chain_name in CHAINS:
        chain = decimate_chain(chain_name, ds.fs, 2, MYOWARE_CUTOFF_HZ)
        recs = tuple(replace(r, emg=chain(r.emg), fs=ds.fs / 2) for r in ds.recordings)
        clean = prefilter(recs, ds.mains_hz, cfg)
        if chain_name == "ideal":
            reference = clean
        th_idx = [i for i, r in enumerate(recs) if r.session in TRAIN_DAYS]
        th = fit_thresholds(
            [recs[i] for i in th_idx], ds.mains_hz, cfg, clean=[clean[i] for i in th_idx]
        )
        t = featurize(recs, ds.mains_hz, cfg, th, clean=clean)
        same, nxt = [], []
        for d in days:
            tr = t.where(sessions=[d], t_min=1.0, t_max=TEST_START_S)
            te = t.where(sessions=[d], t_min=TEST_START_S)
            same.append(accuracy(te.label, LDAClassifier().fit(tr.X, tr.label).predict(te.X)))
            if d + 1 in days:
                full = t.where(sessions=[d])
                te1 = t.where(sessions=[d + 1], t_min=TEST_START_S)
                model = LDAClassifier().fit(full.X, full.label)
                nxt.append(accuracy(te1.label, model.predict(te1.X)))
        entry: dict[str, Any] = {
            "same_day_accuracy": summarize(same),
            "next_day_accuracy": summarize(nxt),
        }
        if reference is not None and chain_name != "ideal":
            # energy of (this chain - ideal chain) inside 20-450 Hz, relative to the signal
            err = []
            for ref, x in zip(reference, clean, strict=True):
                n = min(len(ref), len(x))
                skip = int(0.25 * ds.fs / 2)
                err.append(
                    np.sum((x[skip:n] - ref[skip:n]) ** 2, axis=0)
                    / np.sum(ref[skip:n] ** 2, axis=0)
                )
            entry["in_band_error_energy_median_per_channel"] = np.median(err, axis=0)
        out[chain_name] = entry
    return out


def noise_model() -> dict[str, Any]:
    table: dict[str, Any] = {}
    for order in NOISE_ORDERS:
        for fs in NOISE_RATES:
            r = folded_noise_ratio(fs, MYOWARE_CUTOFF_HZ, order)
            table[f"order{order}_fs{int(fs)}"] = {
                "order": order,
                "fs": fs,
                "folded_over_in_band": r,
                "noise_floor_increase_db": ratio_to_db(r),
            }
    # The firmware's actual OVERSAMPLE 2 path: ADC at 2 kHz, then its halfband FIR.
    for order in NOISE_ORDERS:
        r = folded_noise_ratio_decimated(1000.0, MYOWARE_CUTOFF_HZ, order, HALFBAND_Q15)
        table[f"order{order}_fs2000_halfband"] = {
            "order": order,
            "fs": 2000.0,
            "decimator": "firmware halfband, 47 taps",
            "folded_over_in_band": r,
            "noise_floor_increase_db": ratio_to_db(r),
        }
    return {"cutoff_hz": MYOWARE_CUTOFF_HZ, "band_hz": (20.0, 450.0), "cases": table}


def run(opts: RunOptions) -> dict[str, Any]:
    days = TEST_DAYS if not opts.quick else TEST_DAYS[:4]
    with timed("load MultiDay"):
        ds = datasets.load(
            "multiday",
            root=opts.data_root,
            download=opts.download,
            days=sorted((TRAIN_DAYS[:3] if opts.quick else TRAIN_DAYS) + days),
        )
    with timed("spectrum"):
        spec = spectrum(ds)
    with timed("emulated 1024 Hz chains"):
        emu = emulated(ds, days)
    payload = {
        "provenance": provenance(
            ds,
            {
                "features": asdict(FeatureConfig()),
                "chains": CHAINS,
                "myoware_cutoff_hz": MYOWARE_CUTOFF_HZ,
                "days": days,
            },
        ),
        "spectrum": spec,
        "emulated_1024hz": emu,
        "white_noise_model": noise_model(),
    }
    print(f"  wrote {write_result(opts, 'aliasing', payload)}")
    return payload
