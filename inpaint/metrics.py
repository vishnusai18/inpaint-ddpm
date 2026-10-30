"""Metrics computed over the MISSING region only (mask == 1), images in [-1, 1].

PSNR  : 10 log10(1 / MSE_hole) on the [0, 1] scale, per image.
SSIM  : Gaussian-window SSIM map (Wang et al. 2004), averaged over hole pixels.
LPIPS : AlexNet LPIPS spatial map, averaged over hole pixels (needs the `lpips` package and
        pretrained backbone weights; returns NaN if they are unavailable).
"""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

PSNR_CAP = 100.0


def _unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0, 1)


def psnr_hole(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) PSNR over hole pixels. mask: (B,1,H,W)."""
    se = ((_unit(pred) - _unit(target)) ** 2) * mask
    mse = se.sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * pred.shape[1]).clamp_min(1.0)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).clamp(max=PSNR_CAP)


def psnr_full(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((_unit(pred) - _unit(target)) ** 2).mean((1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).clamp(max=PSNR_CAP)


def ssim_hole(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    p = _unit(pred).permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    t = _unit(target).permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    m = mask[:, 0].cpu().numpy() > 0.5
    out = []
    for i in range(len(p)):
        _, smap = structural_similarity(p[i], t[i], channel_axis=-1, data_range=1.0, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False, full=True)
        out.append(smap.mean(-1)[m[i]].mean() if m[i].any() else np.nan)
    return np.array(out)


class LPIPSHole:
    """LPIPS restricted to the hole. Falls back to NaN when weights cannot be loaded."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        self.fn, self.error = None, None
        try:
            import lpips
            self.fn = lpips.LPIPS(net=net, spatial=True, verbose=False).to(device).eval()
        except Exception as e:  # no internet for backbone weights, package missing, ...
            self.error = repr(e)[:200]

    @torch.no_grad()
    def __call__(self, pred, target, mask) -> np.ndarray:
        if self.fn is None:
            return np.full(pred.shape[0], np.nan)
        d = self.fn(pred.clamp(-1, 1), target.clamp(-1, 1))  # (B,1,H,W) spatial map, inputs in [-1,1]
        return ((d * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1.0)).cpu().numpy()
