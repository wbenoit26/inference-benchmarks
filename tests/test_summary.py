import pytest

from bench.summary import summarize, trial_metrics


def batch(**overrides):
    return {"strain": 1000.0, "data": 0.0, "model": 0.0, "peak_gib": 1.0,
            **overrides}  # fmt: skip


def test_steady_metrics_leave_out_the_first_batch():
    record = {
        "wall": 100.0,
        "log": {
            "model_ready": 5.0,
            "end": 70.0,
            "batches": [
                batch(elapsed=30.0, model=25.0, data=4.0),
                batch(elapsed=10.0, model=5.0, data=2.0, peak_gib=1.5),
            ],
        },
    }
    metrics = trial_metrics(record)
    assert metrics["throughput"] == pytest.approx(50)  # 2000 s in 40 s
    assert metrics["steady_throughput"] == pytest.approx(100)
    assert metrics["model_throughput"] == pytest.approx(200)
    assert metrics["data_fraction"] == pytest.approx(0.15)
    assert metrics["peak_gib"] == 1.5
    assert metrics["startup"] == 30.0


def test_summarize_gives_median_and_range_of_each_metric():
    summary = summarize([{"x": 3.0}, {"x": 1.0}, {"x": 2.0}])
    assert summary == {"x": {"median": 2.0, "min": 1.0, "max": 3.0}}


def test_single_batch_trial_uses_it_for_steady_metrics():
    record = {
        "wall": 20.0,
        "log": {
            "model_ready": 5.0,
            "end": 15.0,
            "batches": [batch(elapsed=10.0, model=4.0)],
        },
    }
    metrics = trial_metrics(record)
    assert metrics["steady_throughput"] == pytest.approx(100)  # 1000 s / 10 s
    assert metrics["model_throughput"] == pytest.approx(250)
