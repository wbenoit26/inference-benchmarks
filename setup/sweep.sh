#!/bin/bash
# Run a sweep of trials on this node's GPU, under gpus/<GPU name>/ in the
# run directory.
#
#     setup/sweep.sh <site> [--repeats N] [--flag value ...]
#
# with flags as for bench.sweep. On Delta, run the following from the root
# of the repo:
#
#     sbatch --account=bcse-delta-gpu --partition=gpuA100x4 --gpus=1 \
#         --cpus-per-task=16 --mem=64G --time=02:00:00 \
#         --output=/work/nvme/bcse/$USER/runs/bench/sweep.%j.log \
#         --wrap "setup/sweep.sh delta --repeats 3 --backend export aoti \
#             --shifts_per_batch 1 2 4 8 16 32"

set -euo pipefail
source "$(dirname "$0")/common.sh"
shift

cd "$RUN_DIR"
apptainer exec --nv \
    --bind "$RUN_DIR" --bind "$RUN_DIR/code:/opt/aframe" \
    --bind "$here" \
    --env PYTHONPATH="$here" \
    --env TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/inductor" \
    "$AFRAME_CONTAINER_ROOT/infer.sif" \
    python -m bench.sweep bench_cmd.txt gpus "$@"
