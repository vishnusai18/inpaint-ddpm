"""Every verification check must pass on correct code AND fail when a fault is injected.
(Sizes are reduced here so the suite runs in about a minute on a laptop CPU.)"""
import numpy as np
import pytest
import torch

import inpaint.diffusion as diffmod
import inpaint.masks as masks
from inpaint import verify as V


def test_masks_pass():
    assert V.check_mask_area(sizes=(32,), n=60)["failed"] == 0
    assert V.check_mask_format(n=30)["failed"] == 0
    assert V.check_rect_geometry(n=60)["failed"] == 0
    assert V.check_irregular_shape(n=60)["failed"] == 0
    assert V.check_mask_determinism(n=20)["failed"] == 0
    assert V.check_train_masks_dynamic(n=24)["failed"] == 0
    assert V.check_train_test_masks_differ(n=60)["failed"] == 0


def test_data_pass():
    assert V.check_corruption(n=24)["failed"] == 0
    assert V.check_split_disjoint()["failed"] == 0


def test_diffusion_pass():
    for fn in (V.check_schedule, V.check_forward_marginal, V.check_x0_recovery, V.check_posterior_bayes,
               V.check_respacing, V.check_oracle_sampler, V.check_sampler_determinism):
        assert fn()["failed"] == 0, fn.__name__


def test_conditioning_pass():
    assert V.check_visible_preserved()["failed"] == 0
    assert V.check_input_layout()["failed"] == 0
    r = V.check_severity_branch()
    assert r["failed"] == 0 and r["relative_increase"] < 0.05


def test_metrics_pass():
    assert V.check_psnr()["failed"] == 0
    assert V.check_ssim()["failed"] == 0
    assert V.check_lpips_wrapper()["failed"] == 0


def test_overfit_pass():
    assert V.check_overfit(steps=120)["failed"] == 0


# ----------------------------------------------------------------- fault injection
def test_fault_wrong_mask_area_is_caught(monkeypatch):
    real = masks.rect_mask
    monkeypatch.setattr(masks, "rect_mask", lambda size, frac, rng: real(size, frac * 1.5, rng))
    assert V.check_mask_area(sizes=(32,), n=40)["failed"] > 0


def test_fault_non_rectangular_rect_is_caught(monkeypatch):
    monkeypatch.setattr(masks, "rect_mask", lambda size, frac, rng: masks.irregular_mask(size, frac, rng))
    assert V.check_rect_geometry(n=40)["failed"] > 0


def test_fault_dropped_compositing_is_caught(monkeypatch):
    """If the sampler stops pasting visible pixels back, the preservation check must fail."""
    orig = diffmod.GaussianDiffusion.sample

    def broken(self, model, corrupted, mask, frac=None, steps=None, paste_known=False, generator=None, return_raw=False):
        out, raw = orig(self, model, corrupted, mask, frac, steps, paste_known, generator, return_raw=True)
        return raw  # returns the un-composited sample
    monkeypatch.setattr(diffmod.GaussianDiffusion, "sample", broken)
    assert V.check_visible_preserved()["failed"] > 0


def test_fault_wrong_forward_scaling_is_caught(monkeypatch):
    monkeypatch.setattr(diffmod.GaussianDiffusion, "q_sample",
                        lambda self, x0, t, noise: 0.9 * x0 + 0.5 * noise)
    assert V.check_x0_recovery()["failed"] > 0 or V.check_forward_marginal()["failed"] > 0


def test_fault_wrong_posterior_is_caught(monkeypatch):
    real = diffmod.Schedule.__init__

    def bad(self, betas, timesteps=None):
        real(self, betas, timesteps)
        self.post_c2 = self.post_c2 * 1.05
    monkeypatch.setattr(diffmod.Schedule, "__init__", bad)
    assert V.check_posterior_bayes()["failed"] > 0 or V.check_oracle_sampler()["failed"] > 0


def test_fault_leaky_severity_branch_is_caught(monkeypatch):
    """A non-zero-initialised severity branch breaks the 'same function at init' guarantee."""
    import torch.nn as nn
    from inpaint import unet as U
    real = U.UNet.__init__

    def leaky(self, *a, **kw):
        real(self, *a, **kw)
        if getattr(self, "use_severity", False):
            nn.init.normal_(self.sev_mlp[2].weight, std=0.1)
    monkeypatch.setattr(U.UNet, "__init__", leaky)
    assert V.check_severity_branch()["failed"] > 0


def test_fault_metric_over_full_image_is_caught(monkeypatch):
    """If 'hole' PSNR secretly averaged over the whole image, the hole-only invariance check fails."""
    import inpaint.metrics as me

    monkeypatch.setattr(V, "psnr_hole", lambda p, t, m: me.psnr_full(p, t))
    assert V.check_psnr()["failed"] > 0
