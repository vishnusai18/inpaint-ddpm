"""Verification suite for the restoration pipeline.

    python -m inpaint.verify --out results/verification.json

Every check returns {"checked": n, "failed": k, ...details}. Groups:
  A masks      : area control, binary/shape, rectangle geometry, irregular shape, determinism,
                 dynamic-vs-fixed behaviour
  B data       : value range, corruption correctness, split disjointness
  C diffusion  : schedule, forward marginals, x0 recovery, posterior vs an independent Bayes
                 derivation, respacing, oracle-sampler recovery, seeded determinism
  D conditioning: model input layout, visible-pixel preservation, severity branch behaviour,
                 identical initialisation of shared weights, parameter-count delta
  E metrics    : PSNR against a manual calculation, hole-only invariance, SSIM, LPIPS wrapper
  F learning   : overfit-one-batch for the diffusion model and the U-Net baseline
The tests in tests/test_verification.py also inject faults and assert these checks catch them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from . import masks as M
from .data import FixedInpaintDataset, MixedFixedDataset, TrainInpaintDataset, corrupt, load_splits
from .diffusion import GaussianDiffusion, Schedule, make_betas
from .metrics import PSNR_CAP, LPIPSHole, psnr_full, psnr_hole, ssim_hole
from .unet import UNet, count_params


def _res(checked: int, failed: int, **kw) -> dict:
    return {"checked": int(checked), "failed": int(failed), **kw}


# ----------------------------------------------------------------------------- A: masks
def check_mask_area(sizes=(32, 64), n=300, seed=0) -> dict:
    """Each mask hits its target missing fraction. Rect: within 1.2 points. Irregular: grown until the
    target is reached, so never under and over by at most 3 points."""
    checked = failed = 0
    worst = {}
    for size in sizes:
        for kind in M.MASK_TYPES:
            for lvl in M.LEVELS:
                a = M.FixedMaskSet(kind, lvl, n, size, seed).masks.mean((1, 2))
                dev = a - lvl
                worst[f"{kind}_{lvl}_{size}"] = float(np.abs(dev).max())
                bad = (np.abs(dev) > 0.012) if kind == "rect" else ((dev < -1e-9) | (dev > 0.03))
                checked += n
                failed += int(bad.sum())
    return _res(checked, failed, worst_abs_deviation=worst)


def check_mask_format(size=64, n=100, seed=0) -> dict:
    checked = failed = 0
    for kind in M.MASK_TYPES:
        for lvl in M.LEVELS:
            ms = M.FixedMaskSet(kind, lvl, n, size, seed).masks
            ok = (ms.shape == (n, size, size)) & (ms.dtype == np.float32) & np.isin(ms, (0.0, 1.0)).all((1, 2))
            checked += n
            failed += int((~ok).sum()) if hasattr(ok, "sum") else int(not ok)
    return _res(checked, failed)


def check_rect_geometry(size=64, n=300, seed=0) -> dict:
    """Rectangular masks are exactly one filled axis-aligned rectangle (bbox area == mask area)."""
    checked = failed = 0
    for lvl in M.LEVELS:
        for m in M.FixedMaskSet("rect", lvl, n, size, seed).masks:
            ys, xs = np.nonzero(m)
            bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            checked += 1
            failed += int(bbox != m.sum())
    return _res(checked, failed)


def check_irregular_shape(size=64, n=300, seed=0) -> dict:
    """Irregular masks must be genuinely free-form: at least 90 percent have a bounding-box fill below 0.9."""
    checked = failed = 0
    fills = []
    for lvl in M.LEVELS:
        for m in M.FixedMaskSet("irregular", lvl, n, size, seed).masks:
            ys, xs = np.nonzero(m)
            fills.append(m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))
    fills = np.array(fills)
    frac_free = float((fills < 0.9).mean())
    checked, failed = 1, int(frac_free < 0.9)
    return _res(checked, failed, fraction_non_rectangular=frac_free, mean_bbox_fill=float(fills.mean()))


def check_mask_determinism(size=64, n=50, seed=3) -> dict:
    checked = failed = 0
    for kind in M.MASK_TYPES:
        for lvl in M.LEVELS:
            a = M.FixedMaskSet(kind, lvl, n, size, seed).masks
            b = M.FixedMaskSet(kind, lvl, n, size, seed).masks
            c = M.FixedMaskSet(kind, lvl, n, size, seed + 1).masks
            checked += 2
            failed += int(not np.array_equal(a, b)) + int(np.array_equal(a, c))
    # conditions use independent streams: masks of one level do not depend on whether other levels exist
    a = M.FixedMaskSet("rect", 0.2, n, size, seed).masks
    _ = M.FixedMaskSet("rect", 0.4, n, size, seed)
    checked += 1
    failed += int(not np.array_equal(a, M.FixedMaskSet("rect", 0.2, n, size, seed).masks))
    return _res(checked, failed)


def check_train_masks_dynamic(size=32, n=64) -> dict:
    """Training masks change across epochs, are reproducible for a given (seed, epoch, index), and
    the fraction handed to the model equals the true area of the mask."""
    imgs = np.zeros((n, size, size, 3), np.uint8)
    ds = TrainInpaintDataset(imgs, size, seed=5, flip=False)
    checked = failed = 0
    ds.set_epoch(0)
    e0 = [ds[i]["mask"].numpy().copy() for i in range(n)]
    e0b = [ds[i]["mask"].numpy() for i in range(n)]
    ds.set_epoch(1)
    e1 = [ds[i]["mask"].numpy() for i in range(n)]
    for i in range(n):
        checked += 3
        failed += int(not np.array_equal(e0[i], e0b[i])) + int(np.array_equal(e0[i], e1[i]))
        ds.set_epoch(0)
        s = ds[i]
        failed += int(abs(float(s["frac"]) - float(s["mask"].mean())) > 1e-7)
    return _res(checked, failed)


def check_train_test_masks_differ(size=32, n=200) -> dict:
    """Fixed evaluation masks are not reused as training masks (checked by exact comparison)."""
    ds = TrainInpaintDataset(np.zeros((n, size, size, 3), np.uint8), size, seed=0, flip=False)
    train = {ds[i]["mask"].numpy().tobytes() for i in range(n)}
    clash = 0
    for kind in M.MASK_TYPES:
        for lvl in M.LEVELS:
            for m in M.FixedMaskSet(kind, lvl, n, size, 1234).masks:
                clash += int(m[None].tobytes() in train)
    return _res(6 * n, clash)


# ------------------------------------------------------------------------------ B: data
def check_corruption(size=32, n=64) -> dict:
    imgs = np.random.default_rng(0).integers(0, 256, (n, size, size, 3), dtype=np.uint8)
    ds = TrainInpaintDataset(imgs, size, seed=0, flip=False)
    checked = failed = 0
    for i in range(n):
        s = ds[i]
        c, m, x = s["clean"], s["mask"], s["corrupted"]
        ok = (c.min() >= -1 and c.max() <= 1 and c.dtype == torch.float32 and c.shape == (3, size, size)
              and torch.all(x[m.expand_as(x) == 1] == 0) and torch.equal(x[m.expand_as(x) == 0], c[m.expand_as(c) == 0]))
        checked += 1
        failed += int(not ok)
    return _res(checked, failed)


def check_split_disjoint(size=32) -> dict:
    """Synthetic splits share no image; also verifies sizes. (CelebA partitions are checked by
    scripts/prepare_celeba.py on the official list_eval_partition file.)"""
    d = load_splits("synthetic", size, 400, 100, 100, seed=0)
    h = {k: {hashlib.md5(x.tobytes()).hexdigest() for x in v} for k, v in d.items()}
    overlap = len(h["train"] & h["val"]) + len(h["train"] & h["test"]) + len(h["val"] & h["test"])
    sizes_ok = (len(d["train"]), len(d["val"]), len(d["test"])) == (400, 100, 100)
    return _res(4, overlap + int(not sizes_ok))


# ------------------------------------------------------------------------- C: diffusion
def check_schedule() -> dict:
    checked = failed = 0
    for T in (100, 200, 1000):
        for sch in ("linear", "cosine"):
            b = make_betas(T, sch)
            ab = np.cumprod(1 - b)
            ok = bool((b > 0).all() and (b < 1).all() and np.all(np.diff(ab) < 0) and 0 < ab[-1] < 0.01 and ab[0] > 0.9)
            checked += 1
            failed += int(not ok)
    return _res(checked, failed)


def check_forward_marginal(T=200, n=20000, seed=0) -> dict:
    """At the last step x_T must be ~ N(0, 1) regardless of x0 (mean 0, std 1 within sampling error)."""
    g = torch.Generator().manual_seed(seed)
    D = GaussianDiffusion(T)
    x0 = torch.rand(n, 1, 1, 1, generator=g) * 2 - 1
    xt = D.q_sample(x0, torch.full((n,), T - 1), torch.randn(n, 1, 1, 1, generator=g))
    ok = abs(float(xt.mean())) < 0.03 and abs(float(xt.std()) - 1.0) < 0.03
    return _res(2, int(not ok), mean=float(xt.mean()), std=float(xt.std()))


def check_x0_recovery(T=200, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    D = GaussianDiffusion(T)
    x0 = torch.rand(64, 3, 8, 8, generator=g) * 2 - 1
    eps = torch.randn(x0.shape, generator=g)
    t = torch.randint(0, T, (64,), generator=g)
    err = float((D.predict_x0(D.q_sample(x0, t, eps), t, eps) - x0).abs().max())
    return _res(64, int(err > 1e-4), max_abs_error=err)


def check_posterior_bayes(T=200) -> dict:
    """Posterior q(x_{t-1}|x_t,x0) coefficients from Schedule vs an independent derivation:
    product of Gaussians N(x_{t-1}; sqrt(ab_{t-1}) x0, 1-ab_{t-1}) and N(x_t; sqrt(alpha_t) x_{t-1}, beta_t)."""
    b = make_betas(T)
    ab = np.cumprod(1 - b)
    sch = Schedule(b)
    checked = failed = 0
    for t in range(1, T):
        prec = 1.0 / (1 - ab[t - 1]) + (1 - b[t]) / b[t]          # precision of the posterior
        var = 1.0 / prec
        c_x0 = (np.sqrt(ab[t - 1]) / (1 - ab[t - 1])) * var       # coefficient on x0
        c_xt = (np.sqrt(1 - b[t]) / b[t]) * var                   # coefficient on x_t
        ok = (np.isclose(sch.post_var[t], var, rtol=1e-8) and np.isclose(sch.post_c1[t], c_x0, rtol=1e-8)
              and np.isclose(sch.post_c2[t], c_xt, rtol=1e-8))
        checked += 1
        failed += int(not ok)
    ok0 = sch.post_var[0] == 0.0
    return _res(checked + 1, failed + int(not ok0))


def check_respacing(T=200) -> dict:
    b = make_betas(T)
    full = np.cumprod(1 - b)
    checked = failed = 0
    for steps in (10, 50, 100):
        ts = np.unique(np.linspace(0, T - 1, steps).round().astype(int))
        s = Schedule(b, ts)
        ok = (np.allclose(s.ab, full[ts]) and s.post_var[0] == 0 and np.all(s.beta > 0) and np.all(np.diff(s.ab) < 0))
        # alphas over the respaced chain multiply back to the original cumulative product
        ok &= np.isclose(np.prod(1 - s.beta), s.ab[-1])
        checked += 1
        failed += int(not ok)
    return _res(checked, failed)


class _Oracle(torch.nn.Module):
    """Noise predictor that knows the clean image: the best possible denoiser."""

    def __init__(self, diff, x0):
        super().__init__()
        self.ab = torch.tensor(diff.full.ab, dtype=torch.float32)
        self.x0 = x0

    def forward(self, xin, t, frac=None):
        ab = self.ab[t].view(-1, 1, 1, 1)
        return (xin[:, :3] - ab.sqrt() * self.x0) / (1 - ab).sqrt()


def check_oracle_sampler(T=200, seed=0) -> dict:
    """With a perfect denoiser the whole reverse chain (full and respaced) returns x0."""
    g = torch.Generator().manual_seed(seed)
    D = GaussianDiffusion(T)
    x0 = torch.rand(4, 3, 16, 16, generator=g) * 2 - 1
    mask = torch.zeros(4, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    cor = corrupt(x0, mask)
    checked = failed = 0
    worst = 0.0
    for steps in (None, 100, 25):
        gen = torch.Generator().manual_seed(1)
        _, raw = D.sample(_Oracle(D, x0), cor, mask, steps=steps, generator=gen, return_raw=True)
        err = float((raw - x0).abs().max())
        worst = max(worst, err)
        checked += 1
        failed += int(err > 1e-3)
    return _res(checked, failed, worst_error=worst)


def check_sampler_determinism(seed=0) -> dict:
    torch.manual_seed(seed)
    D = GaussianDiffusion(50)
    net = UNet(7, 3, base=16, mults=(1, 2), image_size=16, attn_res=(8,))
    net.eval()
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    m = torch.zeros(2, 1, 16, 16)
    m[:, :, 4:10, 4:10] = 1
    a = D.sample(net, corrupt(x, m), m, steps=10, generator=torch.Generator().manual_seed(9))
    b = D.sample(net, corrupt(x, m), m, steps=10, generator=torch.Generator().manual_seed(9))
    c = D.sample(net, corrupt(x, m), m, steps=10, generator=torch.Generator().manual_seed(10))
    return _res(2, int(not torch.equal(a, b)) + int(torch.equal(a, c)))


# ------------------------------------------------------------------- D: conditioning etc.
def check_visible_preserved(seed=0) -> dict:
    """The output must equal the input at every visible pixel (bit-exact) for random-weight models,
    with and without per-step pasting."""
    torch.manual_seed(seed)
    D = GaussianDiffusion(40)
    x = torch.rand(4, 3, 16, 16) * 2 - 1
    checked = failed = 0
    for sev in (False, True):
        net = UNet(7, 3, base=16, mults=(1, 2), image_size=16, attn_res=(8,), use_severity=sev).eval()
        for kind in M.MASK_TYPES:
            m = torch.from_numpy(M.FixedMaskSet(kind, 0.3, 4, 16, 0).masks)[:, None]
            for paste in (False, True):
                out = D.sample(net, corrupt(x, m), m, frac=m.mean((1, 2, 3)) if sev else None, steps=8, paste_known=paste)
                vis = (1 - m).expand_as(x) == 1
                checked += 1
                failed += int(not torch.equal(out[vis], x[vis]))
    return _res(checked, failed)


def check_input_layout(seed=0) -> dict:
    """The network receives [x_t (3), corrupted (3), mask (1)] in that order."""
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, xin, t, frac=None):
            seen["x"] = xin.clone()
            return torch.zeros_like(xin[:, :3])

    D = GaussianDiffusion(20)
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    m = torch.zeros(2, 1, 8, 8)
    m[:, :, 2:5, 2:5] = 1
    cor = corrupt(x, m)
    D.training_loss(Probe(), x, cor, m, t=torch.tensor([3, 5]), noise=torch.zeros_like(x))
    xin = seen["x"]
    ok = xin.shape[1] == 7 and torch.equal(xin[:, 3:6], cor) and torch.equal(xin[:, 6:], m)
    return _res(3, int(not ok))


def check_severity_branch(seed=0) -> dict:
    """(1) identical shared init, (2) same function at init, (3) the standard model ignores frac,
    (4) a trained-like (non-zero) severity branch changes the output when frac changes,
    (5) the severity branch is the ONLY parameter difference, and is small."""
    kw = dict(base=16, mults=(1, 2), image_size=16, attn_res=(8,))
    torch.manual_seed(seed)
    a = UNet(7, 3, **kw)
    torch.manual_seed(seed)
    b = UNet(7, 3, **kw, use_severity=True)
    sa, sb = a.state_dict(), b.state_dict()
    failed = 0
    failed += int(not all(torch.equal(sa[k], sb[k]) for k in sa))
    x, t = torch.randn(2, 7, 16, 16), torch.tensor([3, 90])
    f1, f2 = torch.tensor([0.1, 0.1]), torch.tensor([0.4, 0.4])
    failed += int(not torch.allclose(a(x, t), b(x, t, f1), atol=1e-6))
    failed += int(not torch.equal(a(x, t, f1), a(x, t, f2)))
    with torch.no_grad():
        for p in b.sev_mlp.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    failed += int(torch.allclose(b(x, t, f1), b(x, t, f2), atol=1e-6))
    extra = {k for k in sb if k not in sa}
    failed += int(not all(k.startswith("sev_mlp") for k in extra))
    delta = count_params(b) - count_params(a)
    return _res(5, failed, severity_extra_params=delta, relative_increase=delta / count_params(a))


# ---------------------------------------------------------------------------- E: metrics
def check_psnr() -> dict:
    g = torch.Generator().manual_seed(0)
    tgt = torch.rand(3, 3, 16, 16, generator=g) * 2 - 1
    pred = (tgt + torch.randn(tgt.shape, generator=g) * 0.2).clamp(-1, 1)
    m = torch.zeros(3, 1, 16, 16)
    m[:, :, 3:9, 4:12] = 1
    got = psnr_hole(pred, tgt, m).numpy()
    failed = 0
    for i in range(3):  # manual computation on the hole pixels only
        a, b = ((pred[i] + 1) / 2)[:, 3:9, 4:12], ((tgt[i] + 1) / 2)[:, 3:9, 4:12]
        manual = 10 * np.log10(1.0 / float(((a - b) ** 2).mean()))
        failed += int(abs(manual - got[i]) > 1e-4)
    failed += int(float(psnr_hole(tgt, tgt, m)[0]) != PSNR_CAP)  # perfect reconstruction -> cap
    # hole-only: altering VISIBLE pixels must not change the hole metric, but must change full-image PSNR
    pred2 = pred.clone()
    pred2[:, :, :2, :] = -pred2[:, :, :2, :]
    failed += int(not np.allclose(psnr_hole(pred2, tgt, m).numpy(), got, atol=1e-6))
    failed += int(np.allclose(psnr_full(pred2, tgt).numpy(), psnr_full(pred, tgt).numpy()))
    return _res(6, failed)


def check_ssim() -> dict:
    g = torch.Generator().manual_seed(0)
    tgt = torch.rand(2, 3, 32, 32, generator=g) * 2 - 1
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1
    same = ssim_hole(tgt, tgt, m)
    noisy = ssim_hole((tgt + torch.randn(tgt.shape, generator=g) * 0.5).clamp(-1, 1), tgt, m)
    pred2 = tgt.clone()
    pred2[:, :, :4, :] = 0.0  # change visible pixels far from the hole's window influence
    far = ssim_hole(pred2, tgt, m)
    failed = int(not np.allclose(same, 1.0, atol=1e-6)) + int(not np.all(noisy < 0.9)) + int(not np.allclose(far, 1.0, atol=1e-3))
    return _res(3, failed)


def check_lpips_wrapper() -> dict:
    """The wrapper averages the spatial map over the hole only, and degrades to NaN (not an error)
    when backbone weights are unavailable."""
    w = LPIPSHole.__new__(LPIPSHole)
    w.error = None
    w.fn = lambda a, b: torch.arange(16.0).reshape(1, 1, 4, 4).repeat(a.shape[0], 1, 1, 1)
    m = torch.zeros(2, 1, 4, 4)
    m[:, :, 0, :] = 1  # hole = first row -> values 0,1,2,3 -> mean 1.5
    got = w(torch.zeros(2, 3, 4, 4), torch.zeros(2, 3, 4, 4), m)
    failed = int(not np.allclose(got, 1.5))
    nan = LPIPSHole.__new__(LPIPSHole)
    nan.fn, nan.error = None, "stub"
    failed += int(not np.all(np.isnan(nan(torch.zeros(2, 3, 4, 4), torch.zeros(2, 3, 4, 4), m))))
    return _res(2, failed)


# ---------------------------------------------------------------------------- F: learning
def _overfit(model_kind: str, steps=150, seed=0) -> tuple[float, float]:
    torch.manual_seed(seed)
    n, s = 8, 16
    x = torch.rand(n, 3, s, s) * 2 - 1
    m = torch.from_numpy(M.FixedMaskSet("rect", 0.25, n, s, 0).masks)[:, None]
    cor = corrupt(x, m)
    if model_kind == "unet":
        net = UNet(4, 3, base=16, mults=(1, 2), image_size=s, attn_res=(8,), use_time=False)
        loss_fn = lambda: ((net(torch.cat([cor, m], 1)) - x).abs() * (6 * m + (1 - m))).mean()
    else:
        net = UNet(7, 3, base=16, mults=(1, 2), image_size=s, attn_res=(8,), use_severity=(model_kind == "ddpm_sev"))
        D = GaussianDiffusion(50)
        t = torch.randint(0, 50, (n,))
        noise = torch.randn(x.shape)
        frac = m.mean((1, 2, 3))
        loss_fn = lambda: D.training_loss(net, x, cor, m, frac if model_kind == "ddpm_sev" else None, t, noise)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    first = float(loss_fn().detach())
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    return first, float(loss_fn().detach())


def check_overfit(steps=150) -> dict:
    """A correct pipeline can memorise one tiny batch: the loss must fall by at least 70 percent."""
    out, failed = {}, 0
    for kind in ("unet", "ddpm", "ddpm_sev"):
        a, b = _overfit(kind, steps)
        out[kind] = {"initial": a, "final": b}
        failed += int(b > 0.3 * a)
    return _res(3, failed, losses=out)


ALL_CHECKS = [
    ("A1_mask_area", check_mask_area), ("A2_mask_format", check_mask_format),
    ("A3_rect_geometry", check_rect_geometry), ("A4_irregular_shape", check_irregular_shape),
    ("A5_mask_determinism", check_mask_determinism), ("A6_train_masks_dynamic", check_train_masks_dynamic),
    ("A7_train_test_masks_differ", check_train_test_masks_differ),
    ("B1_corruption", check_corruption), ("B2_split_disjoint", check_split_disjoint),
    ("C1_schedule", check_schedule), ("C2_forward_marginal", check_forward_marginal),
    ("C3_x0_recovery", check_x0_recovery), ("C4_posterior_bayes", check_posterior_bayes),
    ("C5_respacing", check_respacing), ("C6_oracle_sampler", check_oracle_sampler),
    ("C7_sampler_determinism", check_sampler_determinism),
    ("D1_visible_preserved", check_visible_preserved), ("D2_input_layout", check_input_layout),
    ("D3_severity_branch", check_severity_branch),
    ("E1_psnr", check_psnr), ("E2_ssim", check_ssim), ("E3_lpips_wrapper", check_lpips_wrapper),
    ("F1_overfit_one_batch", check_overfit),
]


def run_all() -> dict:
    return {name: fn() for name, fn in ALL_CHECKS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification.json")
    args = ap.parse_args(argv)
    res = run_all()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    tot_c = sum(v["checked"] for v in res.values())
    tot_f = sum(v["failed"] for v in res.values())
    for k, v in res.items():
        print(f"{k:28s} checked={v['checked']:6d} failed={v['failed']}")
    print(f"TOTAL checked={tot_c} failed={tot_f}")
    raise SystemExit(1 if tot_f else 0)


if __name__ == "__main__":
    main()
