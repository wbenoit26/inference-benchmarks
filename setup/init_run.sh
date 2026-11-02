#!/bin/bash
# Create the benchmark's run directory with the small
# preset and the benchmark settings.
# Activate the snakemake environment and then:
#
#     setup/init_run.sh <site>

set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ -e $RUN_DIR ]]; then
    echo "$RUN_DIR already exists" >&2
    exit 1
fi

cd "$AFRAME_REPO"
uv run aframe-init snakemake --preset small -p "$AFRAME_PROFILE" -d "$RUN_DIR"

# Append parameters to override the values in the default config. Tb is enough
# to give > 100 branches per file, and branches_per_job is large enough so
# that each file's branches are in one group, meaning that group 0 has enough
# for any value of shifts_per_batch.
cat >> "$RUN_DIR/config.yaml" <<EOF
Tb: 1_000_000
branches_per_job: 1000
EOF
