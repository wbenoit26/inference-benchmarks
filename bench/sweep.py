import itertools
import sys
from pathlib import Path

from bench.trial import code_version, gpu_info, load_cmd, run_trial


def trial_dir(root: Path, varied: dict, repeat: int) -> Path:
    config = (
        "_".join([f"{k}-{v}" for k, v in varied.items()])
        if varied
        else "default"
    )
    return root / config / str(repeat)


def sweep(
    cmd: list[str],
    varied: dict[str, list],
    fixed: dict,
    repeats: int,
    root: Path,
    gpu: dict,
    version: str,
) -> list[Path]:
    combinations = [
        dict(zip(varied.keys(), combo, strict=True))
        for combo in itertools.product(*varied.values())
    ]

    written = []
    for repeat in range(repeats):
        for combination in combinations:
            outdir = trial_dir(root, combination, repeat)
            if (outdir / "trial.json").exists():
                continue
            overrides = fixed | combination
            written.append(run_trial(cmd, overrides, outdir, gpu, version))

    return written


def main() -> None:
    """Run a sweep of trials from the command line.

    From the run directory:

        python -m bench.sweep <cmd.txt> <root> [--repeats N] [--flag value ...]

    where cmd.txt is the same as for bench.trial. A flag with one value is
    passed to every configuration. Flags with multiple values have every
    combination run.
    """
    cmd_file, root, *args = sys.argv[1:]
    flags = {}
    for arg in args:
        if arg.startswith("--"):
            values = flags[arg.removeprefix("--")] = []
        else:
            values.append(arg)
    repeats = int(flags.pop("repeats", ["1"])[0])
    varied = {
        flag: values for flag, values in flags.items() if len(values) > 1
    }
    fixed = {
        flag: values[0] for flag, values in flags.items() if len(values) == 1
    }

    cmd = load_cmd(Path(cmd_file))
    written = sweep(
        cmd, varied, fixed, repeats, Path(root), gpu_info(), code_version()
    )
    print(f"wrote {len(written)} trials under {root}")


if __name__ == "__main__":
    main()
