#!/bin/bash
# Zaratan (UMD) GPU job. EDIT the account / partition / module lines for your allocation --
# they are placeholders, check the Zaratan documentation. One job trains one model.
#   sbatch --export=MODEL=ddpm_sev scripts/slurm/train.sbatch
#SBATCH --job-name=inpaint-%x
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
# module load python/3.11 cuda      # <- adapt to the cluster's module names
# source .venv/bin/activate
MODEL=${MODEL:?set MODEL to unet | ddpm | ddpm_sev}
python -m inpaint.train --model "$MODEL" --preset full --root data --out runs/full
