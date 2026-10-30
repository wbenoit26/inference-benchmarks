import json
import sys
from pathlib import Path

import h5py
import numpy as np


def compare_raw(trial: Path, reference: Path) -> dict:
    """Compare a trial's raw network outputs with the reference's.

    Returns {group: max_abs}, with group "background" or "foreground".
    `max_abs` is the largest absolute difference over the branches in
    both files, or None if no branch in the group has samples.
    """
    results = {}
    with (
        h5py.File(trial / "timeseries.hdf5", "r") as t,
        h5py.File(reference / "timeseries.hdf5", "r") as r,
    ):
        for group in ("background", "foreground"):
            max_abs = []
            for branch_id in t[group].keys() & r[group].keys():
                a, b = t[group][branch_id], r[group][branch_id]
                if a.attrs["sample_t0"] != b.attrs["sample_t0"]:
                    raise ValueError(f"{group} {branch_id} starts differ")
                n = min(len(a), len(b))
                if n == 0:
                    continue
                max_abs.append(float(abs(a[:n] - b[:n]).max()))
            results[group] = max(max_abs, default=None)

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


def main() -> None:
    """Compare a trial's outputs with the reference's.

        python -m bench.compare <trial dir> <reference dir>

    where each directory holds timeseries.hdf5, background.hdf5 and
    foreground.hdf5.
    """
    trial, reference = Path(sys.argv[1]), Path(sys.argv[2])
    result = {
        "raw": compare_raw(trial, reference),
        "events": compare_events(trial, reference),
        "foreground": compare_foreground(trial, reference),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
