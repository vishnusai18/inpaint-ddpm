"""Evaluate trained models on fixed test masks, per mask type x missing level.

    python -m inpaint.evaluate --runs runs/smoke --preset smoke --out results/smoke

Every model sees the SAME test images and the SAME masks (fixed seed per condition), and both
diffusion models start the sampler from the SAME initial noise, so comparisons are paired.
Metrics are computed over the hole only (see metrics.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .config import add_common_args, resolve
from .data import FixedInpaintDataset, load_splits
from .diffusion import GaussianDiffusion
from .masks import LEVELS, MASK_TYPES, FixedMaskSet
from .metrics import LPIPSHole, psnr_full, psnr_hole, ssim_hole
from .models import MODEL_NAMES, build_model

TEST_MASK_SEED = 1234


def composite(pred, corrupted, mask):
    return mask * pred + (1 - mask) * corrupted


def mean_fill(corrupted, mask):
    """Trivial reference: fill the hole with the mean colour of the visible pixels."""
    vis = (1 - mask)
    mean = (corrupted * vis).sum((2, 3), keepdim=True) / vis.sum((2, 3), keepdim=True).clamp_min(1)
    return composite(mean.expand_as(corrupted), corrupted, mask)


def load_trained(run_dir: str, name: str, args, dev):
    ck = torch.load(os.path.join(run_dir, name, "ckpt.pt"), map_location=dev)
    a = argparse.Namespace(**{**vars(args), **ck["args"]})
    model = build_model(name, a).to(dev)
    model.load_state_dict(ck["ema"])  # evaluate the EMA weights
    return model.eval(), ck["step"]


@torch.no_grad()
def predict(name, model, diff, cor, mask, sample_steps, batch_idx, dev, paste_known=False):
    if name == "unet":
        return composite(model(torch.cat([cor, mask], 1)), cor, mask)
    g = torch.Generator(device=dev).manual_seed(777 + batch_idx)  # identical init noise for ddpm / ddpm_sev
    frac = mask.mean((1, 2, 3)) if name == "ddpm_sev" else None
    return diff.sample(model, cor, mask, frac=frac, steps=sample_steps, generator=g, paste_known=paste_known)


def bootstrap_ci(d: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = d[~np.isnan(d)]
    means = rng.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--paste-known", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--n-examples", type=int, default=4, help="qualitative rows per condition")
    add_common_args(ap)
    args = resolve(ap.parse_args(argv))
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    data = load_splits(args.source, args.size, args.n_train, args.n_val, args.n_test, args.root, seed=0)
    test = data["test"]
    models, steps = {}, {}
    for m in args.models:
        models[m], steps[m] = load_trained(args.runs, m, args, dev)
    diff = GaussianDiffusion(args.T, args.schedule)
    lp = None if args.no_lpips else LPIPSHole(device=dev)
    if lp is not None and lp.error:
        print("LPIPS unavailable -> NaN:", lp.error)

    rows, grids = [], {}
    names = list(models) + ["mean_fill"]
    for kind in MASK_TYPES:
        for level in LEVELS:
            cache = os.path.join(args.out, f"cache_{kind}_{level}.pt")
            if os.path.exists(cache):  # resume: this condition was already evaluated
                try:
                    c = torch.load(cache, weights_only=False)
                    rows += c["rows"]; grids[(kind, level)] = c["grid"]
                    continue
                except Exception:
                    pass
            n_rows0 = len(rows)
            ds = FixedInpaintDataset(test, FixedMaskSet(kind, level, args.n_eval, args.size, TEST_MASK_SEED))
            t0 = time.time()
            keep = {n: [] for n in names}
            for bi, b in enumerate(DataLoader(ds, batch_size=32)):
                clean, cor, mask = (b[k].to(dev) for k in ("clean", "corrupted", "mask"))
                for n in names:
                    pred = mean_fill(cor, mask) if n == "mean_fill" else predict(
                        n, models[n], diff, cor, mask, args.sample_steps, bi, dev, args.paste_known)
                    ps, pf = psnr_hole(pred, clean, mask).cpu().numpy(), psnr_full(pred, clean).cpu().numpy()
                    ss = ssim_hole(pred, clean, mask)
                    lpv = lp(pred, clean, mask) if lp is not None else np.full(len(ps), np.nan)
                    for j in range(len(ps)):
                        rows.append(dict(model=n, mask=kind, level=level, idx=bi * 32 + j, psnr_hole=ps[j],
                                         ssim_hole=ss[j], lpips_hole=lpv[j], psnr_full=pf[j]))
                    if bi == 0:
                        keep[n].append(pred[: args.n_examples].cpu())
                if bi == 0:
                    keep["_clean"] = clean[: args.n_examples].cpu()
                    keep["_cor"] = (cor + mask * 2.0 - 1.0 * mask)[: args.n_examples].cpu()  # hole -> white-ish
                    keep["_mask"] = mask[: args.n_examples].cpu()
            grids[(kind, level)] = {k: (torch.cat(v) if isinstance(v, list) else v) for k, v in keep.items()
                                    if len(v) > 0}
            torch.save({"rows": rows[n_rows0:], "grid": grids[(kind, level)]}, cache)
            print(f"{kind} {level:.2f}: {time.time()-t0:.0f}s", flush=True)

    # ---- tables ----------------------------------------------------------------------
    cols = ["model", "mask", "level", "idx", "psnr_hole", "ssim_hole", "lpips_hole", "psnr_full"]
    with open(os.path.join(args.out, "metrics_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summary = []
    for n in names:
        for kind in MASK_TYPES:
            for level in LEVELS:
                r = [x for x in rows if x["model"] == n and x["mask"] == kind and x["level"] == level]
                s = dict(model=n, mask=kind, level=level, n=len(r))
                for k in ("psnr_hole", "ssim_hole", "lpips_hole", "psnr_full"):
                    v = np.array([x[k] for x in r], dtype=float)
                    s[k] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
                    s[k + "_std"] = float(np.nanstd(v)) if not np.all(np.isnan(v)) else float("nan")
                summary.append(s)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ---- paired comparisons: severity-aware vs standard DDPM (and vs U-Net) ------------
    paired = []
    pairs = [("ddpm_sev", "ddpm"), ("ddpm_sev", "unet"), ("ddpm", "unet")]
    for a_, b_ in pairs:
        if a_ not in models or b_ not in models:
            continue
        for kind in MASK_TYPES:
            for level in LEVELS:
                for k in ("psnr_hole", "ssim_hole", "lpips_hole"):
                    A = np.array([x[k] for x in rows if x["model"] == a_ and x["mask"] == kind and x["level"] == level])
                    B = np.array([x[k] for x in rows if x["model"] == b_ and x["mask"] == kind and x["level"] == level])
                    d = A - B
                    if np.all(np.isnan(d)):
                        continue
                    mean, lo, hi = bootstrap_ci(d)
                    p = float(stats.wilcoxon(d[~np.isnan(d)]).pvalue) if np.any(d != 0) else 1.0
                    paired.append(dict(a=a_, b=b_, mask=kind, level=level, metric=k, mean_diff=mean, ci_lo=lo,
                                       ci_hi=hi, wilcoxon_p=p, n=int(np.sum(~np.isnan(d)))))
    if paired:
        with open(os.path.join(args.out, "paired.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired[0]))
            w.writeheader()
            w.writerows(paired)
    torch.save({f"{k[0]}_{k[1]}": v for k, v in grids.items()}, os.path.join(args.out, "examples.pt"))
    json.dump({"train_steps": steps, "sample_steps": args.sample_steps, "n_eval": args.n_eval,
               "lpips_error": None if lp is None else lp.error, "source": args.source, "size": args.size},
              open(os.path.join(args.out, "eval_meta.json"), "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
