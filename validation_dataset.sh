#!/bin/bash
#SBATCH -J GRAZPED
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.11

source "venv311/bin/activate"

python -u graz_ped.py \
  --checkpoint checkpoints/best.pt \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --control-policy no-fracture-text-ok