import json
import re
import sys
from pathlib import Path

from bench.logs import parse_log
from bench.summary import sum_over_batches, summarize

GROUP_ID = re.compile(r"group_id=(?P<id>\d+)")


def condor_job(ad: dict) -> dict:
    intervals = {
        # Time in queue
        "queue": ("QDate", "JobCurrentStartDate"),
        # Time spent transferring inputs
        "transfer_in": (
            "JobCurrentStartTransferInputDate",
            "JobCurrentFinishTransferInputDate",
        ),
        # Time spent executing
        "execute": ("JobCurrentStartExecutingDate", "CompletionDate"),
        # Time spent transferring outputs
        "transfer_out": (
            "JobCurrentStartTransferOutputDate",
            "JobCurrentFinishTransferOutputDate",
        ),
        # Time the job held its GPU slot
        "slot": ("JobCurrentStartDate", "CompletionDate"),
    }
    job = {
        "group": int(GROUP_ID.search(ad["Args"]).group("id")),
        "starts": ad["NumJobStarts"],
    }
    for name, (start, end) in intervals.items():
        job[name] = ad[end] - ad[start] if start in ad and end in ad else None

    # Time lost due to job evictions
    job["lost"] = (
        ad["RemoteWallClockTime"] - job["slot"]
        if "RemoteWallClockTime" in ad and job["slot"] is not None
        else None
    )

    return job


def campaign_jobs(campaign: Path) -> list[dict]:
    ads = json.loads((campaign / "condor_ads.json").read_text())

    jobs = []
    for ad in ads:
        if ad["JobStatus"] != 4 or ad.get("ExitCode") != 0:
            continue
        job = condor_job(ad)
        log = campaign / f"logs/infer/infer_group-{job['group']}.log"
        record = parse_log(log)
        job["gpu"] = record["gpu"]
        job["unlogged"] = job["execute"] - record["end"]
        job |= sum_over_batches(record["batches"])
        job["fixed"] = job["slot"] - job["elapsed"]
        jobs.append(job)

    return jobs


def main() -> None:
    """Summarize the per-job costs of one or more campaigns.

        python -m bench.costs <campaign> [<campaign> ...]

    where each campaign directory holds condor_ads.json (from
    `condor_history -json`) and logs/infer/infer_group-<group>.log for each
    job.
    """
    for campaign in sys.argv[1:]:
        jobs = campaign_jobs(Path(campaign))
        gpus = sorted({job["gpu"] for job in jobs})
        print(f"{campaign}: {len(jobs)} jobs on {', '.join(gpus)}")
        values = [
            {
                name: value
                for name, value in job.items()
                if name not in ("group", "gpu") and value is not None
            }
            for job in jobs
        ]
        for name, s in summarize(values).items():
            spread = f"{s['min']:.1f}-{s['max']:.1f}"
            print(f"  {name} {s['median']:.1f} ({spread})")


if __name__ == "__main__":
    main()
