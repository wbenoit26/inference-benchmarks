import h5py
import numpy as np

from bench.compare import compare_raw, overlap


def write_series(trial_dir, sample_t0, background, foreground=()):
    """One branch, "0", at 4 samples per second."""
    trial_dir.mkdir()
    with h5py.File(trial_dir / "timeseries.hdf5", "w") as f:
        f.attrs["inference_sampling_rate"] = 4.0
        for group, data in [
            ("background", background),
            ("foreground", foreground),
        ]:
            ds = f.create_group(group).create_dataset(
                "0", data=np.array(data, float)
            )
            ds.attrs["sample_t0"] = sample_t0


def test_overlap_of_series_starting_one_second_apart():
    # at 4 samples per second, b starts 4 samples into a
    assert overlap(100.0, 10, 101.0, 10, rate=4.0) == (
        slice(4, 10),
        slice(0, 6),
    )


def test_overlap_of_series_cut_short():
    assert overlap(100.0, 10, 100.0, 7, rate=4.0) == (slice(0, 7), slice(0, 7))


def test_compare_matches_samples_by_time(tmp_path):
    # the trial starts one sample later than the reference
    write_series(tmp_path / "ref", 0.0, background=[0.0, 1.0, 2.0, 3.0])
    write_series(tmp_path / "trial", 0.25, background=[1.0, 2.5, 3.0])
    result = compare_raw(tmp_path / "trial", tmp_path / "ref")
    assert result["background"] == {"0": {"samples": 3, "max_abs": 0.5}}


def test_compare_leaves_out_empty_series(tmp_path):
    write_series(tmp_path / "ref", 0.0, background=[0.0], foreground=[])
    write_series(tmp_path / "trial", 0.0, background=[0.0], foreground=[])
    assert (
        compare_raw(tmp_path / "trial", tmp_path / "ref")["foreground"] == {}
    )
