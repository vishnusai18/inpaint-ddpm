"""Factory for the three models compared in the project."""
from __future__ import annotations

from .unet import UNet

MODEL_NAMES = ("unet", "ddpm", "ddpm_sev")


def build_model(name: str, args) -> UNet:
    kw = dict(base=args.base, mults=tuple(args.mults), image_size=args.size, attn_res=tuple(args.attn_res))
    if name == "unet":      # direct reconstruction: [corrupted RGB, mask] -> RGB, no timestep
        return UNet(in_ch=4, out_ch=3, use_time=False, **kw)
    if name == "ddpm":      # conditional DDPM: [noisy RGB, corrupted RGB, mask] -> noise
        return UNet(in_ch=7, out_ch=3, use_time=True, **kw)
    if name == "ddpm_sev":  # same, plus missing-fraction embedding
        return UNet(in_ch=7, out_ch=3, use_time=True, use_severity=True, **kw)
    raise ValueError(name)
