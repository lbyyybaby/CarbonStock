#!/bin/bash
# Usage: bash submit_train.sh [Hydra overrides, e.g. seed=43 run_test=true]
set -euo pipefail

repository_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repository_dir"

# Use Bangkok time for directory names, independently of the login node's TZ.
export TZ=Asia/Bangkok
mkdir -p "$repository_dir/runs/agbd"
run_dir="$(mktemp -d "$repository_dir/runs/agbd/$(date +%Y-%m-%d_%H-%M-%S)-XXXXXX")"
export CARBONSTOCK_RUN_DIR="$run_dir"

# Archive the submitted job script alongside its logs and training outputs.
cp "$repository_dir/train.slurm" "$run_dir/train.slurm"
printf 'Run directory: %s\n' "$run_dir"
job_id="$(sbatch --parsable \
  --chdir="$repository_dir" \
  --export=ALL \
  --output="$run_dir/train-%j.out" \
  --error="$run_dir/train-%j.err" \
  "$repository_dir/train.slurm" "$@")"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"
printf 'Submitted batch job: %s\n' "$job_id"
