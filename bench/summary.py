import json
import sys
from pathlib import Path

import numpy as np


def sum_over_batches(batches):
    return {
        k: sum([batch[k] for batch in batches])
        for k in ["strain", "elapsed", "data", "model"]
    }


def trial_metrics(record: dict) -> dict:
    """
    The metrics for a single trial.

    Returns a dictionary with:
        "throughput":        seconds of strain / seconds elapsed
        "steady_throughput": same, but with the first batch removed
        "model_throughput":  seconds of strain / seconds of model time
                             without the first batch
        "data_fraction":     seconds of data loading / elapsed time
        "peak_gib":          peak memory across all batches
        "first_batch":       time elapsed in the first batch
        "model_load":        seconds for the model to load
        "startup":           walltime - time recorded in log
    """
    log = record["log"]
    batches, steady_batches = log["batches"], log["batches"][1:]

    full_total = sum_over_batches(batches)
    if len(batches) == 1:
        steady_total = full_total
    else:
        steady_total = sum_over_batches(steady_batches)

    return {
        "throughput": full_total["strain"] / full_total["elapsed"],
        "steady_throughput": steady_total["strain"] / steady_total["elapsed"],
        "model_throughput": steady_total["strain"] / steady_total["model"],
        "data_fraction": full_total["data"] / full_total["elapsed"],
        "peak_gib": max([batch["peak_gib"] for batch in batches]),
        "first_batch": batches[0]["elapsed"],
        "model_load": log["model_ready"],
        "startup": record["wall"] - log["end"],
    }


def summarize(metrics: list[dict]) -> dict:
    summary = {}
    metrics = {k: [d[k] for d in metrics] for k in metrics[0]}
    for name in metrics:
        summary[name] = {
            "median": np.median(metrics[name]),
            "min": min(metrics[name]),
            "max": max(metrics[name]),
        }

    return summary


def main() -> None:
    """Summarize repeated trials of each config.

        python -m bench.summary [<trials dir>]

    where the trials directory (bench/trials by default) holds
    <config>/<repeat>/trial.json.
    """
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/trials")
    for config in sorted(p for p in root.iterdir() if p.is_dir()):
        records = [
            json.loads(p.read_text())
            for p in sorted(config.glob("*/trial.json"))
        ]
        ok = [r for r in records if r["returncode"] == 0]
        if not ok:
            continue
        summary = summarize([trial_metrics(r) for r in ok])
        items = [
            f"{name} {s['median']:.3g} ({s['min']:.3g}-{s['max']:.3g})"
            for name, s in summary.items()
        ]
        failed = len(records) - len(ok)
        print(f"{config.name}: {len(ok)} trials, {failed} failed")
        print("  " + "\n  ".join(items))


if __name__ == "__main__":
    main()
