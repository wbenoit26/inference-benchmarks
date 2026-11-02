from pathlib import Path

import pytest

from bench.sweep import sweep


@pytest.fixture
def runs(monkeypatch):
    calls = []

    def fake_run_trial(cmd, overrides, outdir, gpu, version):
        calls.append((outdir, overrides))
        outdir.mkdir(parents=True)
        (outdir / "trial.json").write_text("{}")
        return outdir / "trial.json"

    monkeypatch.setattr("bench.sweep.run_trial", fake_run_trial)
    return calls


def test_sweep_runs_each_repeat_across_configs_before_the_next(tmp_path, runs):
    sweep(["infer-local"], {"spb": [1, 4]}, {}, 2, tmp_path, {}, "v")
    assert [outdir.relative_to(tmp_path) for outdir, _ in runs] == [
        Path("spb-1/0"),
        Path("spb-4/0"),
        Path("spb-1/1"),
        Path("spb-4/1"),
    ]


def test_sweep_skips_trials_already_written(tmp_path, runs):
    (tmp_path / "spb-1" / "0").mkdir(parents=True)
    (tmp_path / "spb-1" / "0" / "trial.json").write_text("{}")
    written = sweep(["infer-local"], {"spb": [1, 4]}, {}, 1, tmp_path, {}, "v")
    assert written == [tmp_path / "spb-4" / "0" / "trial.json"]


def test_fixed_flags_reach_the_trial_but_not_its_directory_name(
    tmp_path, runs
):
    sweep(
        ["infer-local"],
        {"spb": [4]},
        {"weights": "m.pt2"},
        1,
        tmp_path,
        {},
        "v",
    )
    ((outdir, overrides),) = runs
    assert outdir == tmp_path / "spb-4" / "0"
    assert overrides == {"weights": "m.pt2", "spb": 4}
