from pathlib import Path

import h5py
import numpy as np


def compare_raw(trial: Path, reference: Path) -> dict:
    """Compare a trial's raw network outputs with the reference's.

    Returns {group: {branch_id: {"samples": int, "max_abs": float}}}, with
    group "background" or "foreground", for each branch in both files.
    """
    results = {}
    with (
        h5py.File(trial / "timeseries.hdf5", "r") as t,
        h5py.File(reference / "timeseries.hdf5", "r") as r,
    ):
        for group in ("background", "foreground"):
            results[group] = {}
            for branch_id in t[group].keys() & r[group].keys():
                a, b = t[group][branch_id], r[group][branch_id]
                if a.attrs["sample_t0"] != b.attrs["sample_t0"]:
                    raise ValueError(f"{group} {branch_id} starts differ")
                n = min(len(a), len(b))
                if n == 0:
                    continue
                timeseries_a, timeseries_b = a[:n], b[:n]

                results[group][branch_id] = {
                    "samples": len(timeseries_a),
                    "max_abs": float(abs(timeseries_a - timeseries_b).max()),
                }

    return results


def compare_events(trial: Path, reference: Path, loudest: int = 100) -> dict:
    def loudest_first(directory: Path):
        with h5py.File(directory / "background.hdf5", "r") as f:
            return np.sort(f["parameters/detection_statistic"][:])[::-1]

    statistics = loudest_first(trial)
    reference_statistics = loudest_first(reference)

    n = min(loudest, len(statistics), len(reference_statistics))
    return {
        "events": len(statistics),
        "reference_events": len(reference_statistics),
        "max_abs": float(abs(statistics[:n] - reference_statistics[:n]).max()),
    }


def compare_foreground(trial: Path, reference: Path) -> dict:
    def injections(directory: Path):
        with h5py.File(directory / "foreground.hdf5", "r") as f:
            times = np.array(f["parameters/injection_time"][:])
            stats = np.array(f["parameters/detection_statistic"][:])
            return times, stats

    times, statistics = injections(trial)
    reference_times, reference_statistics = injections(reference)

    if not np.array_equal(times, reference_times):
        raise ValueError(
            "Injection times between trials and reference don't match"
        )
    return {
        "injections": len(statistics),
        "max_abs": float(abs(statistics - reference_statistics).max()),
    }
