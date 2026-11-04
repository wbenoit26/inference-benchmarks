#!/bin/bash
# Time the model execution per kernel for different model variants
# and write the output to gpus/<GPU name>/kernel_curve.json
#
#     setup/kernel_curve.sh <site>
#
# On Delta, run the following from the root of the repo:
#
#     sbatch --account=bcse-delta-gpu --partition=gpuA100x4 --gpus=1 \
#         --cpus-per-task=16 --mem=64G --time=01:00:00 \
#         --output=/work/nvme/bcse/$USER/runs/bench/kernel_curve.%j.log \
#         --wrap "setup/kernel_curve.sh delta"

set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$RUN_DIR"
apptainer exec --nv \
    --bind "$RUN_DIR" \
    --bind "$here" \
    --env PYTHONPATH="$here" \
    "$AFRAME_CONTAINER_ROOT/infer.sif" \
    python -m bench.kernel_curve
