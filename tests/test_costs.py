import json

from bench.costs import campaign_jobs, condor_job


def ad(**overrides):
    """A successful job of group 0, never evicted, with no file transfer."""
    return {
        "Args": "--target-jobs infer_group:group_id=0 --cores all",
        "JobStatus": 4,
        "ExitCode": 0,
        "NumJobStarts": 1,
        "RemoteWallClockTime": 0,
        "QDate": 0,
        "JobCurrentStartDate": 0,
        "JobCurrentStartExecutingDate": 0,
        "CompletionDate": 0,
        **overrides,
    }


def entry(seconds, message):
    return f"2026-10-27 10:00:{seconds:02d},000 - root - INFO - {message}"


def test_condor_job_takes_intervals_from_ad_dates():
    job = condor_job(
        ad(
            Args="--target-jobs infer_group:group_id=7 --cores all",
            QDate=100,
            JobCurrentStartDate=160,
            JobCurrentStartTransferInputDate=160,
            JobCurrentFinishTransferInputDate=200,
            JobCurrentStartExecutingDate=200,
            CompletionDate=300,
        )
    )
    assert job["group"] == 7
    assert (job["queue"], job["transfer_in"], job["execute"]) == (60, 40, 100)
    assert job["slot"] == 140


def test_condor_job_without_transfer_dates_has_no_transfer_times():
    job = condor_job(ad())
    assert job["transfer_in"] is None
    assert job["transfer_out"] is None


def test_condor_job_counts_slot_time_lost_to_evictions():
    job = condor_job(
        ad(
            NumJobStarts=2,
            JobCurrentStartDate=100,
            CompletionDate=150,
            RemoteWallClockTime=80,
        )
    )
    # the last start held its slot for 50 s of the 80 s total
    assert job["lost"] == 30


def test_campaign_keeps_successful_jobs_and_adds_log_totals(tmp_path):
    ads = [ad(JobStatus=3), ad(ExitCode=1), ad(CompletionDate=40)]
    (tmp_path / "condor_ads.json").write_text(json.dumps(ads))
    (tmp_path / "logs" / "infer").mkdir(parents=True)
    (tmp_path / "logs" / "infer" / "infer_group-0.log").write_text(
        entry(0, "LocalInferenceClient ready (backend=export, GPU A)") + "\n"
        + entry(30, "Inferred 1 streams in 20.0 s: 5.0 s loading data, "
                "14.0 s in the model, 1.0 s postprocessing. "
                "Peak GPU memory 0.1 GiB") + "\n"
        + entry(30, "Finished branches ['0'] (2000 s of strain each) "
                "in 20.0 s")
    )  # fmt: skip
    (job,) = campaign_jobs(tmp_path)
    assert job["gpu"] == "GPU A"
    assert (job["strain"], job["elapsed"], job["model"]) == (2000, 20, 14)
    # executed for 40 s, but the log spans only 30 s of it
    assert job["unlogged"] == 10
    # held the slot for 40 s, of which 20 s were in the batch loop
    assert job["fixed"] == 20
