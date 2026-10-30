import h5py
import numpy as np
import pytest

from bench.compare import compare_events, compare_foreground, compare_raw


def write_series(trial_dir, background, foreground=()):
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
            ds.attrs["sample_t0"] = 0.0


def test_compare_stops_at_the_shorter_series(tmp_path):
    # the trial's series was cut short by one sample
    write_series(tmp_path / "ref", background=[0.0, 1.0, 2.0, 9.0])
    write_series(tmp_path / "trial", background=[0.5, 1.0, 2.0])
    result = compare_raw(tmp_path / "trial", tmp_path / "ref")
    assert result["background"] == 0.5


def test_compare_of_only_empty_series_is_none(tmp_path):
    write_series(tmp_path / "ref", background=[0.0], foreground=[])
    write_series(tmp_path / "trial", background=[0.0], foreground=[])
    result = compare_raw(tmp_path / "trial", tmp_path / "ref")
    assert result["foreground"] is None


def write_events(trial_dir, name, **parameters):
    trial_dir.mkdir(exist_ok=True)
    with h5py.File(trial_dir / f"{name}.hdf5", "w") as f:
        for key, values in parameters.items():
            f.create_dataset(f"parameters/{key}", data=np.array(values, float))


def test_events_compares_only_the_loudest(tmp_path):
    write_events(tmp_path / "t", "background", detection_statistic=[5, 1, 3])
    write_events(tmp_path / "r", "background", detection_statistic=[3.5, 5, 0])
    # the quietest events differ by 1, but only the loudest 2 are compared
    result = compare_events(tmp_path / "t", tmp_path / "r", loudest=2)
    assert result == {"events": 3, "reference_events": 3, "max_abs": 0.5}


def test_events_compares_fewer_when_a_trial_has_fewer(tmp_path):
    write_events(tmp_path / "t", "background", detection_statistic=[4])
    write_events(tmp_path / "r", "background", detection_statistic=[5, 3])
    result = compare_events(tmp_path / "t", tmp_path / "r", loudest=100)
    assert result == {"events": 1, "reference_events": 2, "max_abs": 1.0}


def test_foreground_compares_same_injections_entry_by_entry(tmp_path):
    write_events(
        tmp_path / "t",
        "foreground",
        injection_time=[10, 20, 30],
        detection_statistic=[1.0, 2.0, 3.0],
    )
    write_events(
        tmp_path / "r",
        "foreground",
        injection_time=[10, 20, 30],
        detection_statistic=[1.0, 2.0, 3.25],
    )
    result = compare_foreground(tmp_path / "t", tmp_path / "r")
    assert result == {"injections": 3, "max_abs": 0.25}


def test_foreground_rejects_different_injections(tmp_path):
    write_events(
        tmp_path / "t",
        "foreground",
        injection_time=[10, 20],
        detection_statistic=[1.0, 2.0],
    )
    write_events(
        tmp_path / "r",
        "foreground",
        injection_time=[20, 10],
        detection_statistic=[2.0, 1.0],
    )
    with pytest.raises(ValueError):
        compare_foreground(tmp_path / "t", tmp_path / "r")
