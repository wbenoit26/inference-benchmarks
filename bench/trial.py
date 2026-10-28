import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from bench.logs import parse_log


def gpu_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "l2_bytes": props.L2_cache_size,
        "memory_bytes": props.total_memory,
    }


def override_args(cmd: list[str], overrides: dict) -> list[str]:
    # Make a copy of cmd to return
    args = list(cmd)
    for key, value in overrides.items():
        flag = f"--{key}"
        if flag not in cmd:
            args += [flag, str(value)]
            continue
        idx = cmd.index(flag)
        args[idx + 1] = str(value)

    return args


def run_trial(
    cmd: list[str], overrides: dict, outdir: Path, gpu: dict, version: str
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "infer.log"
    redirects = {"logfile": log}
    for flag, value in zip(cmd, cmd[1:], strict=False):
        if flag.endswith("_out"):
            name = Path(value).name
            redirects[flag.removeprefix("--")] = outdir / name
    args = override_args(cmd, {**redirects, **overrides})

    start = time.perf_counter()
    with open(outdir / "output.txt", "w") as output:
        result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start

    record = {
        "cmd": args,
        "overrides": overrides,
        "gpu": gpu,
        "version": version,
        "returncode": result.returncode,
        "wall": wall,
        "log": parse_log(log) if log.exists() else None,
    }

    outfile = outdir / "trial.json"
    with open(outfile, "w") as f:
        json.dump(record, f)
    return outfile


def main() -> None:
    """Run one trial from the command line.

    From the run directory, inside the infer image:

        python -m bench.trial <cmd.txt> <outdir> [--flag value ...]

    where cmd.txt holds one group's infer-local command as `snakemake -n -p`
    prints it, and each `--flag value` overrides that flag for this trial.
    """
    cmd_file, outdir, *flags = sys.argv[1:]
    cmd = shlex.split(Path(cmd_file).read_text())
    if "&>" in cmd:
        cmd = cmd[: cmd.index("&>")]
    overrides = {
        flag.removeprefix("--"): value
        for flag, value in zip(flags[::2], flags[1::2], strict=True)
    }
    version_file = Path("code/VERSION")
    version = (
        version_file.read_text().strip()
        if version_file.exists()
        else "unknown"
    )
    print(run_trial(cmd, overrides, Path(outdir), gpu_info(), version))


if __name__ == "__main__":
    main()
