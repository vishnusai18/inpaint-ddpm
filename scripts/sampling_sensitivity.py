#!/usr/bin/env python
"""Does the DDPM result depend on sampler settings? Sweeps sampling steps and RePaint-style
per-step pasting for the standard and severity-aware DDPM on one fixed (mask, level) condition.

    python scripts/sampling_sensitivity.py --runs runs/smoke --preset smoke --out results/smoke/sampling_sensitivity.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inpaint.config import add_common_args, resolve  # noqa: E402
from inpaint.data import FixedInpaintDataset, load_splits  # noqa: E402
from inpaint.diffusion import GaussianDiffusion  # noqa: E402
from inpaint.evaluate import TEST_MASK_SEED, load_trained, predict  # noqa: E402
from inpaint.masks import FixedMaskSet  # noqa: E402
from inpaint.metrics import psnr_hole, ssim_hole  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mask", default="irregular")
ap.add_argument("--level", type=float, default=0.2)
ap.add_argument("--steps-list", type=int, nargs="+", default=[25, 50, 100, 200])
add_common_args(ap)
args = resolve(ap.parse_args())
data = load_splits(args.source, args.size, args.n_train, args.n_val, args.n_test, args.root, seed=0)
ds = FixedInpaintDataset(data["test"], FixedMaskSet(args.mask, args.level, args.n_eval, args.size, TEST_MASK_SEED))
diff = GaussianDiffusion(args.T, args.schedule)
rows = []
for name in ("ddpm", "ddpm_sev"):
    model, _ = load_trained(args.runs, name, args, args.device)
    for steps in args.steps_list:
        for paste in (False, True):
            ps, ss = [], []
            for bi, b in enumerate(DataLoader(ds, batch_size=32)):
                clean, cor, mask = (b[k] for k in ("clean", "corrupted", "mask"))
                pred = predict(name, model, diff, cor, mask, steps, bi, args.device, paste_known=paste)
                ps += psnr_hole(pred, clean, mask).tolist()
                ss += ssim_hole(pred, clean, mask).tolist()
            rows.append(dict(model=name, sample_steps=steps, paste_known=int(paste), psnr_hole=np.mean(ps), ssim_hole=np.mean(ss)))
            print(rows[-1], flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
