#!/usr/bin/env bash
# Reduced real-CelebA run for a laptop (6k/500/500 subset, 64x64). Resumable: re-run after an interruption.
#   python scripts/prepare_celeba.py --raw <celeba_dir> --out data --size 64 --n-train 6000 --n-val 500 --n-test 500
#   STEPS=3000 bash scripts/run_celeba_local.sh
set -euo pipefail
STEPS=${STEPS:-3000}
RUNS=runs/celeba_local
RES=results/celeba_local
ARGS="--preset smoke --source celeba --root data --size 64 --n-train 6000 --n-val 500 --n-test 500 \
      --base 32 --mults 1 2 2 2 --attn-res 8 --batch 32 --val-every 250 --log-every 25 --n-eval 100 --seed 0"
mkdir -p "$RES"
python -m inpaint.verify --out "$RES/verification.json" || true
for m in unet ddpm ddpm_sev; do
  python -m inpaint.train --model "$m" $ARGS --steps "$STEPS" --out "$RUNS"
done
python -m inpaint.evaluate --runs "$RUNS" $ARGS --steps "$STEPS" --out "$RES" --no-lpips
python -m inpaint.evaluate --runs "$RUNS" $ARGS --steps "$STEPS" --out "${RES}_paste" --paste-known --no-lpips
python -m inpaint.make_figures --runs "$RUNS" --results "$RES" --results-paste "${RES}_paste" --out "$RES/figures"
echo "done: $RES  (zip results/celeba_local, results/celeba_local_paste, runs/celeba_local and send them back)"
