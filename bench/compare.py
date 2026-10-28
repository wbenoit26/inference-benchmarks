from pathlib import Path

import h5py


def overlap(
    t0_a: float, n_a: int, t0_b: float, n_b: int, rate: float
) -> tuple[slice, slice]:
    offset = round((t0_b - t0_a) * rate)
    start, stop = max(offset, 0), min(n_a, offset + n_b)
    if start >= stop:
        return slice(0, 0), slice(0, 0)
    return slice(start, stop), slice(start - offset, stop - offset)


def compare_raw(trial: Path, reference: Path) -> dict:
    """Compare a trial's raw network outputs with the reference's.

    Returns {group: {branch_id: {"samples": int, "max_abs": float}}}, with
    group "background" or "foreground", for each branch in both files.
    Only samples at times both series cover are compared, and branches
    with no such samples are left out.
    """
    results = {}
    with (
        h5py.File(trial / "timeseries.hdf5", "r") as t,
        h5py.File(reference / "timeseries.hdf5", "r") as r,
    ):
        rate = r.attrs["inference_sampling_rate"]
        for group in ("background", "foreground"):
            results[group] = {}
            for branch_id in t[group].keys() & r[group].keys():
                t0_a = t[group][branch_id].attrs["sample_t0"]
                n_a = t[group][branch_id].shape[-1]

                t0_b = r[group][branch_id].attrs["sample_t0"]
                n_b = r[group][branch_id].shape[-1]

                slice_a, slice_b = overlap(t0_a, n_a, t0_b, n_b, rate)

                timeseries_a = t[group][branch_id][slice_a]
                timeseries_b = r[group][branch_id][slice_b]

                if len(timeseries_a) == 0:
                    continue

                results[group][branch_id] = {
                    "samples": len(timeseries_a),
                    "max_abs": float(abs(timeseries_a - timeseries_b).max()),
                }

    return results
