import json
import sys

from bench.trial import override_args, run_trial

FAKE_INFER = """
import sys, pathlib
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
pathlib.Path(args["--background_out"]).write_text("")
pathlib.Path(args["--logfile"]).write_text(
    "2026-10-27 10:00:00,000 - root - INFO - Loading model (backend=aoti)\\n"
    "2026-10-27 10:00:02,000 - root - INFO - "
    "LocalInferenceClient ready (backend=aoti, X)\\n"
)
"""


def test_override_replaces_value_of_flag_already_present():
    cmd = ["infer-local", "--backend", "export", "--batch_size", "128"]
    assert override_args(cmd, {"backend": "aoti"}) == [
        "infer-local",
        "--backend",
        "aoti",
        "--batch_size",
        "128",
    ]


def test_override_appends_flag_not_present():
    cmd = ["infer-local", "--backend", "export"]
    assert override_args(cmd, {"aoti_path": "model.pt2"}) == [
        "infer-local",
        "--backend",
        "export",
        "--aoti_path",
        "model.pt2",
    ]


def test_trial_writes_outputs_and_parsed_log_into_outdir(tmp_path):
    cmd = [
        sys.executable,
        "-c",
        FAKE_INFER,
        "--background_out",
        "elsewhere/bg.hdf5",
    ]
    path = run_trial(
        cmd, {"backend": "aoti"}, tmp_path / "trial", gpu={}, version="v"
    )
    record = json.loads(path.read_text())
    assert (tmp_path / "trial" / "bg.hdf5").exists()
    # the fake logs "ready" 2 s after its first line
    assert record["log"]["model_ready"] == 2.0
    assert record["returncode"] == 0


def test_failed_trial_is_recorded_without_log(tmp_path):
    cmd = [sys.executable, "-c", "raise SystemExit(3)"]
    path = run_trial(cmd, {}, tmp_path / "trial", gpu={}, version="v")
    record = json.loads(path.read_text())
    assert record["returncode"] == 3
    assert record["log"] is None
