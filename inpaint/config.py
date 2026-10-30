"""Presets. `smoke` is sized for a 2-core CPU sandbox; `full` for one Zaratan GPU (CelebA 64x64)."""
from __future__ import annotations

import argparse

PRESETS = {
    "smoke": dict(source="synthetic", size=32, n_train=3000, n_val=120, n_test=600, base=32, mults=(1, 2, 2),
                  attn_res=(8,), T=200, sample_steps=50, batch=64, steps=1500, lr=1e-3, ema=0.995,
                  val_every=250, log_every=25, n_eval=48),
    "full": dict(source="celeba", size=64, n_train=30000, n_val=3000, n_test=3000, base=64, mults=(1, 2, 3, 4),
                 attn_res=(16,), T=1000, sample_steps=250, batch=64, steps=100000, lr=2e-4, ema=0.9999,
                 val_every=2000, log_every=100, n_eval=500),
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=list(PRESETS), default="smoke")
    p.add_argument("--source", choices=["synthetic", "celeba"])
    p.add_argument("--root", default="data", help="folder with celeba{size}_{split}.npy")
    for k in ("size", "n_train", "n_val", "n_test", "base", "T", "sample_steps", "batch", "steps", "val_every",
              "log_every", "n_eval"):
        p.add_argument(f"--{k.replace('_', '-')}", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--ema", type=float)
    p.add_argument("--mults", type=int, nargs="+")
    p.add_argument("--attn-res", type=int, nargs="+")
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """Fill every unset option from the chosen preset."""
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k, None) is None:
            setattr(args, k, tuple(v) if isinstance(v, tuple) else v)
    args.mults, args.attn_res = tuple(args.mults), tuple(args.attn_res)
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
