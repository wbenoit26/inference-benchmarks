from bench.logs import parse_log


def entry(time, message):
    return f"2026-10-27 {time} - root - INFO - {message}"


def write_log(tmp_path, *lines):
    path = tmp_path / "infer.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_batch_takes_stats_from_inferred_line_before_it(tmp_path):
    path = write_log(
        tmp_path,
        entry("10:00:00,000", "Loading model (backend=export)"),
        entry(
            "10:00:05,500", "LocalInferenceClient ready (backend=export, NVIDIA A40)"
        ),
        entry(
            "10:00:15,500",
            "Inferred 4 streams in 10.0 s: 2.0 s loading data, "
            "7.5 s in the model, 0.5 s postprocessing. Peak GPU memory 1.25 GiB",
        ),
        entry(
            "10:00:16,000",
            "Finished branches ['0', '1'] (1920 s of strain each) in 10.5 s",
        ),
    )
    record = parse_log(path)
    assert record["gpu"] == "NVIDIA A40"
    assert record["model_ready"] == 5.5
    assert record["batches"] == [
        {
            "finished": 16.0,
            "elapsed": 10.5,
            "branches": 2,
            "strain": 3840.0,
            "streams": 4,
            "data": 2.0,
            "model": 7.5,
            "postprocessing": 0.5,
            "peak_gib": 1.25,
        }
    ]


def test_boundaries_missing_from_log_are_none(tmp_path):
    # a job that crashed while loading the model
    path = write_log(
        tmp_path,
        "/opt/env/lib/python3.12/site-packages/x.py:1: UserWarning: noise",
        entry("10:00:00,000", "Loading model (backend=export)"),
        "Traceback (most recent call last):",
    )
    record = parse_log(path)
    assert record["model_ready"] is None
    assert record["outputs_written"] is None
    assert record["batches"] == []
