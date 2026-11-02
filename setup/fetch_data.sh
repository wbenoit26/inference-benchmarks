#!/bin/bash
# Fetch the strain and make the waveforms and branch lists needed for
# inference. Also writes the infer-local command for group 0 for the 
# benchmarking to run. With aframe-smk active:
#
#     setup/fetch_data.sh <site>

set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$RUN_DIR"

# The o4c model stands in for a trained one, so the pipeline skips training
# and every inference job, in trials or campaigns, uses it.
mkdir -p train
cp "$MODEL_DIR/model_exported.pt2" train/

./run.sh infer_inputs --executor local --cores "$LOCAL_CORES" --jobs "$LOCAL_CORES"

# Get the command for inference group 0 without running it
./run.sh -n -p --target-jobs infer_group:group_id=0 \
    --executor local --cores 1 > dry_run.txt
grep -o 'infer-local .*' dry_run.txt > bench_cmd.txt
