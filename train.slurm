#!/bin/bash
#SBATCH --job-name=carbonstock
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

# Use the project's Python directly; loading a Python 3.13 module does not
# select the separate Python 3.11 environment used for these dependencies.
if [ -n "${CARBONSTOCK_MODULE:-}" ]; then
  module purge
  module load "$CARBONSTOCK_MODULE"
fi
python_bin="${CARBONSTOCK_PYTHON:-$HOME/envs/carbonstock/bin/python}"
if [ ! -x "$python_bin" ]; then
  printf 'Python not found: %s. Set CARBONSTOCK_PYTHON before submission.\n' "$python_bin" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

printf 'Python: %s\n' "$python_bin"
"$python_bin" --version
"$python_bin" -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'"

# Random initialization only needs config.json. When pretrained=true, the
# model loader validates the weights in the configured model.hf_model_path.

"$python_bin" -u main.py "$@"
