"""Figures from training logs and evaluation outputs.

    python -m inpaint.make_figures --runs runs/smoke --results results/smoke --out results/smoke/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

COL = {"unet": "#c77d1a", "ddpm": "#3b78b8", "ddpm_sev": "#7a0f1a", "mean_fill": "#8d8d8d"}
LAB = {"unet": "U-Net (direct)", "ddpm": "DDPM", "ddpm_sev": "DDPM + severity", "mean_fill": "Mean fill"}


def _show(x):  # (3,H,W) in [-1,1] -> HWC in [0,1]
    return ((x.permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()


def smooth(v, k=5):
    v = np.asarray(v, float)
    if len(v) < k:
        return v
    pad = np.concatenate([np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def loss_curves(runs: str, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for name, a in (("unet", ax[0]), ("ddpm", ax[1]), ("ddpm_sev", ax[1])):
        p = os.path.join(runs, name, "log.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        a.plot(d.step, smooth(d.train_loss), color=COL[name], lw=1.4, label=f"{LAB[name]} train")
        v = d.dropna(subset=["val_loss"])
        a.plot(v.step, v.val_loss, "o--", color=COL[name], ms=3.5, lw=1, label=f"{LAB[name]} val")
    ax[0].set_title("(a) U-Net: weighted L1 (hole x6)", fontsize=9, loc="left")
    ax[1].set_title("(b) DDPM: noise-prediction MSE", fontsize=9, loc="left")
    for a in ax:
        a.set_xlabel("training step")
        a.legend(fontsize=6.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def metric_curves(summary_csv: str, path: str) -> None:
    s = pd.read_csv(summary_csv)
    has_lp = s.lpips_hole.notna().any()
    metrics = [("psnr_hole", "PSNR (hole, dB)"), ("ssim_hole", "SSIM (hole)")] + ([("lpips_hole", "LPIPS (hole)")] if has_lp else [])
    fig, ax = plt.subplots(len(metrics), 2, figsize=(6.4, 2.1 * len(metrics)), squeeze=False)
    for j, kind in enumerate(("rect", "irregular")):
        for i, (m, lab) in enumerate(metrics):
            a = ax[i, j]
            for name in ("mean_fill", "unet", "ddpm", "ddpm_sev"):
                d = s[(s["model"] == name) & (s["mask"] == kind)].sort_values("level")
                if len(d):
                    a.errorbar(d.level * 100, d[m], yerr=d[m + "_std"] / np.sqrt(d.n), marker="o", ms=3.5, lw=1.2,
                               capsize=2, color=COL[name], label=LAB[name])
            a.set_xticks([10, 20, 40])
            a.set_ylabel(lab, fontsize=8)
            if i == 0:
                a.set_title({"rect": "Rectangular masks", "irregular": "Irregular masks"}[kind], fontsize=9)
            if i == len(metrics) - 1:
                a.set_xlabel("missing pixels (%)")
            a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def sample_grid(examples_pt: str, path: str, paste_pt: str | None = None, conds=("rect_0.2", "irregular_0.4"), rows=2) -> None:
    """Columns: original, input, U-Net, DDPM and DDPM+severity with the composite-only sampler and
    (if paste_pt is given) with per-step pasting of the known pixels."""
    ex = torch.load(examples_pt)
    ep = torch.load(paste_pt) if paste_pt else None
    cols = ["Original", "Input", "U-Net", "DDPM\ncomposite", "DDPM+sev\ncomposite"] + (["DDPM\npasting", "DDPM+sev\npasting"] if ep else [])
    fig, axes = plt.subplots(rows * len(conds), len(cols), figsize=(1.0 * len(cols), 1.0 * rows * len(conds) + 0.35))
    for ci, c in enumerate(conds):
        e = ex[c]
        for r in range(rows):
            row = ci * rows + r
            imgs = [e["_clean"][r], e["_cor"][r], e["unet"][r], e["ddpm"][r], e["ddpm_sev"][r]]
            if ep:
                imgs += [ep[c]["ddpm"][r], ep[c]["ddpm_sev"][r]]
            for k, im in enumerate(imgs):
                a = axes[row, k]
                a.imshow(_show(im), interpolation="nearest")
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                if row == 0:
                    a.set_title(cols[k], fontsize=6)
            kind, lvl = c.split("_")
            axes[row, 0].set_ylabel(f"{'Rect.' if kind == 'rect' else 'Irreg.'} {int(round(float(lvl) * 100))}%" if r == 0 else "", fontsize=6.5)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def data_examples(examples_pt: str, path: str) -> None:
    """Fixed test masks: original / input for all six conditions."""
    ex = torch.load(examples_pt)
    conds = list(ex)
    fig, axes = plt.subplots(2, len(conds), figsize=(1.1 * len(conds), 2.2))
    for k, c in enumerate(conds):
        axes[0, k].imshow(_show(ex[c]["_clean"][0]))
        axes[1, k].imshow(_show(ex[c]["_cor"][0]))
        axes[0, k].set_title(f"{c.split('_')[0][:5]} {int(round(float(c.split('_')[1]) * 100))}%", fontsize=6)
        for r in range(2):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--results-paste", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    loss_curves(a.runs, os.path.join(a.out, "loss_curves.png"))
    if os.path.exists(os.path.join(a.results, "summary.csv")):
        metric_curves(os.path.join(a.results, "summary.csv"), os.path.join(a.out, "metrics_vs_level.png"))
        sample_grid(os.path.join(a.results, "examples.pt"), os.path.join(a.out, "samples.png"),
                    os.path.join(a.results_paste, "examples.pt") if a.results_paste else None)
        data_examples(os.path.join(a.results, "examples.pt"), os.path.join(a.out, "conditions.png"))
    print("figures in", a.out)


if __name__ == "__main__":
    main()
