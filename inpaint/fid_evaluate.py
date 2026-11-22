"""Compute full-image FID per model, mask type, and corruption level."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from .config import add_common_args, resolve
from .data import FixedInpaintDataset, load_splits
from .diffusion import GaussianDiffusion
from .evaluate import TEST_MASK_SEED, load_trained, mean_fill, predict
from .masks import LEVELS, MASK_TYPES, FixedMaskSet
from .models import MODEL_NAMES


def unit_range(x: torch.Tensor) -> torch.Tensor:
    """Convert project images from [-1, 1] to the [0, 1] range FID expects."""
    return ((x.clamp(-1, 1) + 1.0) / 2.0).float()


def write_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mask", "level", "n", "fid"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--paste-known", action="store_true")
    add_common_args(ap)
    args = resolve(ap.parse_args(argv))

    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "fid_summary.csv")

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    completed = {(r["model"], r["mask"], float(r["level"])) for r in rows}

    data = load_splits(
        args.source, args.size, args.n_train, args.n_val, args.n_test, args.root, seed=0
    )
    test = data["test"]

    models, train_steps = {}, {}
    for name in args.models:
        models[name], train_steps[name] = load_trained(args.runs, name, args, dev)

    diff = GaussianDiffusion(args.T, args.schedule)
    names = list(models) + ["mean_fill"]
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)

    for kind in MASK_TYPES:
        for level in LEVELS:
            ds = FixedInpaintDataset(
                test, FixedMaskSet(kind, level, args.n_eval, args.size, TEST_MASK_SEED)
            )
            for name in names:
                key = (name, kind, float(level))
                if key in completed:
                    print(f"skip {name} {kind} {level:.2f}", flush=True)
                    continue

                fid.reset()
                t0 = time.time()
                for bi, batch in enumerate(DataLoader(ds, batch_size=args.batch_size)):
                    clean, cor, mask = (
                        batch[k].to(dev) for k in ("clean", "corrupted", "mask")
                    )
                    pred = mean_fill(cor, mask) if name == "mean_fill" else predict(
                        name,
                        models[name],
                        diff,
                        cor,
                        mask,
                        args.sample_steps,
                        bi,
                        dev,
                        args.paste_known,
                    )
                    fid.update(unit_range(clean), real=True)
                    fid.update(unit_range(pred), real=False)

                score = float(fid.compute().cpu())
                rows.append(
                    {"model": name, "mask": kind, "level": level, "n": len(ds), "fid": score}
                )
                write_rows(csv_path, rows)
                completed.add(key)
                print(
                    f"{name} {kind} {level:.2f}: FID={score:.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with open(os.path.join(args.out, "fid_meta.json"), "w") as f:
        json.dump(
            {
                "train_steps": train_steps,
                "sample_steps": args.sample_steps,
                "n_eval": args.n_eval,
                "feature": 2048,
                "normalize": True,
                "scope": "full_image",
                "source": args.source,
                "size": args.size,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
