"""Train one model.   python -m inpaint.train --model ddpm_sev --preset smoke --out runs/smoke"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import add_common_args, resolve
from .data import MixedFixedDataset, TrainInpaintDataset, load_splits
from .diffusion import EMA, GaussianDiffusion
from .models import MODEL_NAMES, build_model
from .unet import count_params


def unet_loss(model, clean, corrupted, mask):
    """Direct-reconstruction loss: L1 with the hole weighted 6x (as in partial-conv inpainting)."""
    pred = model(torch.cat([corrupted, mask], 1))
    l1 = (pred - clean).abs()
    return (6.0 * (l1 * mask).sum() + (l1 * (1 - mask)).sum()) / l1.numel()


def batch_loss(name, model, diff, b, dev, fixed=None):
    clean, corrupted, mask, frac = (b[k].to(dev) for k in ("clean", "corrupted", "mask", "frac"))
    if name == "unet":
        return unet_loss(model, clean, corrupted, mask)
    t = noise = None
    if fixed is not None:  # deterministic validation noise / timesteps
        g = torch.Generator().manual_seed(fixed)
        t = torch.randint(0, diff.T, (clean.shape[0],), generator=g).to(dev)
        noise = torch.randn(clean.shape, generator=g).to(dev)
    return diff.training_loss(model, clean, corrupted, mask, frac if name == "ddpm_sev" else None, t, noise)


@torch.no_grad()
def validate(name, model, diff, val_loader, dev):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(val_loader):
        loss = batch_loss(name, model, diff, b, dev, fixed=1000 + i)
        tot += float(loss) * len(b["clean"])
        n += len(b["clean"])
    model.train()
    return tot / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--out", default="runs/smoke")
    add_common_args(ap)
    args = resolve(ap.parse_args(argv))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    run = os.path.join(args.out, args.model)
    os.makedirs(run, exist_ok=True)

    data = load_splits(args.source, args.size, args.n_train, args.n_val, args.n_test, args.root, seed=0)
    train_ds = TrainInpaintDataset(data["train"], args.size, seed=args.seed)
    val_ds = MixedFixedDataset(data["val"], min(args.n_val, len(data["val"])), args.size, seed=7)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = build_model(args.model, args).to(dev)
    ema = EMA(model, args.ema)
    diff = GaussianDiffusion(args.T, args.schedule)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = count_params(model)
    print(f"[{args.model}] params={n_params/1e6:.2f}M device={dev} train={len(train_ds)} val={len(val_ds)}", flush=True)

    step, epoch = 0, 0
    ck = os.path.join(run, "ckpt.pt")
    cands = []
    for f in (ck, os.path.join(run, "ckpt_a.pt"), os.path.join(run, "ckpt_b.pt")):
        if os.path.exists(f):
            try:
                cands.append(torch.load(f, map_location=dev))
            except Exception:  # truncated by an interrupted save
                print(f"ignoring unreadable checkpoint {f}", flush=True)
    if cands:  # resume from the newest readable checkpoint
        s = max(cands, key=lambda c: c["step"])
        model.load_state_dict(s["model"]); ema.shadow = s["ema"]; opt.load_state_dict(s["opt"])
        step, epoch = s["step"], s["epoch"]
        print(f"resumed from step {step}", flush=True)

    log_path = os.path.join(run, "log.csv")
    new = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if new:
        log.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "seconds"])
    t0, running, nrun = time.time(), 0.0, 0
    model.train()
    while step < args.steps:
        train_ds.set_epoch(epoch)
        for b in loader:
            lr = args.lr * min(1.0, (step + 1) / 100)  # linear warm-up
            for g in opt.param_groups:
                g["lr"] = lr
            loss = batch_loss(args.model, model, diff, b, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            running += loss.item(); nrun += 1
            if step % args.log_every == 0 or step == args.steps:
                val = ""
                if step % args.val_every == 0 or step == args.steps:
                    shadow = build_model(args.model, args).to(dev)
                    ema.copy_to(shadow)
                    val = validate(args.model, shadow, diff, val_loader, dev)
                log.writerow([step, epoch, f"{running/nrun:.6f}", val if val == "" else f"{val:.6f}", f"{lr:.6g}",
                              f"{time.time()-t0:.1f}"])
                logf.flush()
                print(f"step {step:6d} loss {running/nrun:.4f} val {val} {time.time()-t0:.0f}s", flush=True)
                running, nrun = 0.0, 0
            if step % args.val_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
                            "step": step, "epoch": epoch, "args": vars(args)},
                           os.path.join(run, "ckpt_a.pt" if (step // args.val_every) % 2 else "ckpt_b.pt"))
                if step == args.steps:  # final weights under the canonical name used by evaluate.py
                    torch.save({"model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
                                "step": step, "epoch": epoch, "args": vars(args)}, ck)
            if step >= args.steps:
                break
        epoch += 1
    json.dump({"model": args.model, "params": n_params, "steps": step, "seconds": time.time() - t0,
               "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}},
              open(os.path.join(run, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
