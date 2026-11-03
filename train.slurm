#!/bin/bash
#SBATCH --job-name=carbonstock
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.13-ai-pytorch
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

which python
python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'"

test -s offline_models/depth-any-canopy-small/config.json
test -s offline_models/depth-any-canopy-small/model.safetensors

python -u main.py "$@"
