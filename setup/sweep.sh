#!/bin/bash
# Run a sweep of trials on this node's GPU.
#
#     setup/sweep.sh <site> [--repeats N] [--flag value ...]
#
# with flags as for bench.sweep.

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
    python -m bench.sweep bench_cmd.txt trials "$@"
