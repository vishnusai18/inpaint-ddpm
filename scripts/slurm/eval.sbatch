#!/bin/bash
# Evaluate all three trained models on the fixed test masks (needs LPIPS weights cached: run
# `python -c "import lpips; lpips.LPIPS(net='alex')"` once on a node with internet access).
#SBATCH --job-name=inpaint-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
python -m inpaint.evaluate --runs runs/full --preset full --root data --out results/full
python -m inpaint.make_figures --runs runs/full --results results/full --out results/full/figures
