import re
from datetime import datetime
from pathlib import Path

TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# How the messages that mark boundaries or carry per-batch stats begin, e.g.
#   LocalInferenceClient ready (backend=export, NVIDIA A40)
#   Inferred 4 streams in 10.0 s: 2.0 s loading data, 7.5 s in the model,
#     0.5 s postprocessing. Peak GPU memory 1.25 GiB
#   Finished branches ['0', '1'] (1920 s of strain each) in 10.5 s
#   Wrote group outputs to infer/groups/0
READY = "LocalInferenceClient ready"
INFERRED = "Inferred"
FINISHED = "Finished branches"
WROTE = "Wrote group outputs"

NUMBER = re.compile(r"\d[\d.]*")


def parse_log(path: Path) -> dict:
    """Read one infer-local log into its boundary times and per-batch stats.

    Times are in seconds since the log's first timestamped line.

    Returns a dict with:
        gpu: GPU name from the "ready" line, or None if it wasn't logged.
        model_ready, outputs_written: when each was logged, or None.
        end: when the last line was logged, or None if no line was.
        batches: one dict per "Finished branches" line, in order, with
            finished, elapsed (s), branches (count), strain (s, all
            branches), and from the "Inferred" line before it: streams,
            data, model, postprocessing (s) and peak_gib. Those five are
            None if no "Inferred" line came before it.
    """
    lines = []
    for line in path.read_text().splitlines():
        splits = line.split(" - ", 3)
        if len(splits) != 4:
            continue
        timestamp, _, _, message = splits
        timestamp = datetime.strptime(timestamp, TIME_FORMAT)
        if len(lines) == 0:
            first = timestamp
        diff = timestamp - first
        lines.append((diff.total_seconds(), message))

    end = lines[-1][0] if len(lines) > 0 else None
    record = {
        "gpu": None,
        "model_ready": None,
        "outputs_written": None,
        "end": end,
        "batches": [],
    }
    inferred = None
    for seconds, message in lines:
        if message.startswith(READY):
            record["gpu"] = message.split(",")[1][1:-1]
            record["model_ready"] = seconds
        if message.startswith(WROTE):
            record["outputs_written"] = seconds
        if message.startswith(INFERRED):
            inferred = NUMBER.findall(message)
        if message.startswith(FINISHED):
            finished = NUMBER.findall(message)
            branches = len(finished) - 2
            elapsed = float(finished[-1])
            strain = float(finished[-2]) * branches
            batch = {
                "finished": seconds,
                "elapsed": elapsed,
                "branches": branches,
                "strain": strain,
                "streams": None,
                "data": None,
                "model": None,
                "postprocessing": None,
                "peak_gib": None,
            }
            if inferred is not None:
                batch["streams"] = int(inferred[0])
                batch["data"] = float(inferred[2])
                batch["model"] = float(inferred[3])
                batch["postprocessing"] = float(inferred[4])
                batch["peak_gib"] = float(inferred[5])
                inferred = None

            record["batches"].append(batch)

    return record
