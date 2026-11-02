#!/bin/bash
#SBATCH --job-name=aframe-build-images
#SBATCH --account=bcse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bcse/%u/builds/build_images.%j.log
# Build the data, train, export, infer and plots images on Delta
#
#     mkdir -p /work/nvme/bcse/$USER/builds
#     sbatch setup/build_images.sh
#

set -euo pipefail
source "$SLURM_SUBMIT_DIR/sites/delta.env"

export APPTAINER_TMPDIR=/tmp/$USER/build
export APPTAINER_CACHEDIR=/work/nvme/bcse/$USER/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$AFRAME_CONTAINER_ROOT"

cd "$AFRAME_REPO"
uv run build-containers data train export infer plots --max-workers 5
