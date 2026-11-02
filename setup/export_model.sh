#!/bin/bash
# Export the o4c model to MODEL_DIR/model_exported.pt2. Only needs to
# happen once per site.
#
# The o4c model is archived on CIT at /home/aframe.online/archive/o4c/models/aframe.pt.
# After copying the o4c model to MODEL_DIR/aframe.pt:
#
#     setup/export_model.sh <site>

set -euo pipefail
source "$(dirname "$0")/common.sh"

apptainer exec --bind "$here" --bind "$MODEL_DIR" \
    --env PYTHONPATH="$here" "$AFRAME_CONTAINER_ROOT/train.sif" \
    python -m bench.convert_model \
    "$MODEL_DIR/aframe.pt" "$MODEL_DIR/model_exported.pt2"
