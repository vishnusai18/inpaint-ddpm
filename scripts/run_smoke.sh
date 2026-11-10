#!/usr/bin/env bash
# CPU smoke test on synthetic faces (no GPU / no CelebA needed). ~1 h on 2 cores.
set -euo pipefail
OUT=${1:-runs/smoke}
STEPS=${STEPS:-1200}
for m in unet ddpm ddpm_sev; do
  python -m inpaint.train --model "$m" --preset smoke --steps "$STEPS" --out "$OUT"
done
