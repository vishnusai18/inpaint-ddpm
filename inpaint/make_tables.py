"""LaTeX table fragments generated from the result files, so report numbers are never typed by hand.

    python -m inpaint.make_tables --results results/smoke --verification results/verification.json --out report/tables
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

GROUPS = {
    "A": ("Masks", "area control, binary format, rectangle geometry, free-form shape, determinism, dynamic vs fixed, train/test separation"),
    "B": ("Data", "value range, hole zeroed / visible pixels untouched, split disjointness"),
    "C": ("Diffusion", "schedule, forward marginal, x0 recovery, posterior vs independent Bayes derivation, respacing, oracle sampler, seeded determinism"),
    "D": ("Conditioning", "visible-pixel preservation, 7-channel input layout, severity branch (identical init, ignored by DDPM, active when trained, small)"),
    "E": ("Metrics", "hole-only PSNR vs manual calculation, SSIM, LPIPS wrapper"),
    "F": ("Learning", "overfit one batch (U-Net, DDPM, severity-aware DDPM)"),
}
NAMES = {"mean_fill": "Mean fill", "unet": "U-Net", "ddpm": "DDPM", "ddpm_sev": "DDPM+sev."}


def verification_table(path: str) -> str:
    v = json.load(open(path))
    rows = []
    for g, (title, desc) in GROUPS.items():
        ks = [k for k in v if k.startswith(g)]
        c, f = sum(v[k]["checked"] for k in ks), sum(v[k]["failed"] for k in ks)
        rows.append(f"{g} & {title} & {desc} & {c:,} & {f} \\\\")
    tc, tf = sum(x["checked"] for x in v.values()), sum(x["failed"] for x in v.values())
    rows.append(f"\\midrule & \\textbf{{Total}} & & \\textbf{{{tc:,}}} & \\textbf{{{tf}}} \\\\")
    return "\n".join(rows)


def _get(df, model, kind, lvl, col):
    r = df[(df["model"] == model) & (df["mask"] == kind) & (np.isclose(df["level"], lvl))]
    return float(r[col].iloc[0])


def metrics_table(composite_csv: str, paste_csv: str | None = None) -> str:
    """Rows: mask type x level. Columns: mean fill, U-Net, then DDPM / DDPM+sev with the composite-only
    sampler and (if given) with per-step pasting. Cells: PSNR(hole) / SSIM(hole). The best DDPM variant per
    metric in a row is bold; the U-Net is a different model class and is reported for reference."""
    comp = pd.read_csv(composite_csv)
    paste = pd.read_csv(paste_csv) if paste_csv and os.path.exists(paste_csv) else None
    cols = [("mean_fill", comp), ("unet", comp), ("ddpm", comp), ("ddpm_sev", comp)]
    if paste is not None:
        cols += [("ddpm", paste), ("ddpm_sev", paste)]
    out = []
    for kind, kname in (("rect", "Rect."), ("irregular", "Irreg.")):
        for lvl in (0.1, 0.2, 0.4):
            vals = [(_get(df, m, kind, lvl, "psnr_hole"), _get(df, m, kind, lvl, "ssim_hole")) for m, df in cols]
            diff_idx = [i for i, (m, _) in enumerate(cols) if m.startswith("ddpm")]
            bp = max(diff_idx, key=lambda i: vals[i][0])
            bs = max(diff_idx, key=lambda i: vals[i][1])
            cells = []
            for i, (p_, q_) in enumerate(vals):
                ps, qs = f"{p_:.2f}", f"{q_:.3f}"
                if i == bp:
                    ps = f"\\textbf{{{ps}}}"
                if i == bs:
                    qs = f"\\textbf{{{qs}}}"
                cells.append(f"{ps}/{qs}")
            out.append(f"{kname if lvl == 0.1 else ''} {int(lvl * 100)}\\% & " + " & ".join(cells) + " \\\\")
        if kind == "rect":
            out.append("\\midrule")
    return "\n".join(out)


def paired_table(composite_csv: str, paste_csv: str | None = None) -> str:
    """Severity-aware minus standard DDPM, hole PSNR (dB): mean paired difference, 95% bootstrap CI, Wilcoxon p."""
    srcs = [("composite", composite_csv)] + ([("pasting", paste_csv)] if paste_csv and os.path.exists(paste_csv) else [])
    tabs = {}
    for name, path in srcs:
        d = pd.read_csv(path)
        tabs[name] = d[(d.a == "ddpm_sev") & (d.b == "ddpm") & (d.metric == "psnr_hole")].reset_index(drop=True)
    rows = []
    for i in range(len(tabs["composite"])):
        r0 = tabs["composite"].iloc[i]
        kn = "Rect." if r0["mask"] == "rect" else "Irreg."
        cells = []
        for name, _ in srcs:
            r = tabs[name].iloc[i]
            cells.append(f"{r.mean_diff:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {r.wilcoxon_p:.3f}")
        rows.append(f"{kn} {int(r0.level * 100)}\\% & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--results-paste", default=None, help="evaluation run with --paste-known")
    ap.add_argument("--verification", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "verification.tex"), "w").write(verification_table(a.verification))
    rp = a.results_paste
    open(os.path.join(a.out, "metrics.tex"), "w").write(metrics_table(os.path.join(a.results, "summary.csv"),
                                                                       rp and os.path.join(rp, "summary.csv")))
    open(os.path.join(a.out, "paired.tex"), "w").write(paired_table(os.path.join(a.results, "paired.csv"),
                                                                     rp and os.path.join(rp, "paired.csv")))
    print("tables in", a.out)


if __name__ == "__main__":
    main()
