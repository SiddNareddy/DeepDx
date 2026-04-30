#!/bin/bash
#SBATCH -J MURA
#SBATCH -p gpu --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/oscar/home/snareddy/data/snareddy/cs1470_final_project"

module purge
module load python/3.11

cd "$PROJECT_ROOT"

source "venv311/bin/activate"

python -u pytorch_alt_model.py \
  --workspace-root "$PROJECT_ROOT" \
  --device cuda \
  --num-workers 8 \
  --num-epochs 20 \
  --model resnet50
