"""Gaussian diffusion (Ho et al. 2020) with epsilon-prediction, conditional on a corrupted
image + mask, DDPM ancestral sampling with optional timestep respacing, and exact
preservation of visible pixels in the output."""
from __future__ import annotations

import numpy as np
import torch


def make_betas(T: int, schedule: str = "linear") -> np.ndarray:
    if schedule == "linear":  # Ho et al.: 1e-4 .. 0.02 for T = 1000, rescaled for other T
        s = 1000.0 / T
        return np.linspace(s * 1e-4, s * 0.02, T, dtype=np.float64)
    if schedule == "cosine":  # Nichol & Dhariwal 2021
        f = lambda t: np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2
        ab = np.array([f(t) / f(0) for t in range(T + 1)])
        return np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    raise ValueError(schedule)


class Schedule:
    """Per-step coefficients for an arbitrary increasing subsequence of timesteps."""

    def __init__(self, betas: np.ndarray, timesteps: np.ndarray | None = None):
        full_ab = np.cumprod(1.0 - betas)
        self.timesteps = np.arange(len(betas)) if timesteps is None else np.asarray(timesteps)
        ab = full_ab[self.timesteps]
        ab_prev = np.concatenate([[1.0], ab[:-1]])
        beta = 1.0 - ab / ab_prev
        self.ab, self.ab_prev, self.beta = ab, ab_prev, beta
        self.sqrt_ab, self.sqrt_1m_ab = np.sqrt(ab), np.sqrt(1.0 - ab)
        self.post_var = beta * (1.0 - ab_prev) / (1.0 - ab)
        self.post_c1 = np.sqrt(ab_prev) * beta / (1.0 - ab)
        self.post_c2 = np.sqrt(1.0 - beta) * (1.0 - ab_prev) / (1.0 - ab)


def _g(a: np.ndarray, idx, like: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(a[idx], dtype=torch.float32, device=like.device)
    return v.reshape(-1, *([1] * (like.dim() - 1)))


class GaussianDiffusion:
    def __init__(self, T: int = 1000, schedule: str = "linear"):
        self.T = T
        self.betas = make_betas(T, schedule)
        self.full = Schedule(self.betas)

    # ---- forward process / training ------------------------------------------------
    def q_sample(self, x0, t, noise):
        return _g(self.full.sqrt_ab, t.cpu().numpy(), x0) * x0 + _g(self.full.sqrt_1m_ab, t.cpu().numpy(), x0) * noise

    def predict_x0(self, xt, t, eps):
        return (xt - _g(self.full.sqrt_1m_ab, t.cpu().numpy(), xt) * eps) / _g(self.full.sqrt_ab, t.cpu().numpy(), xt)

    def training_loss(self, model, clean, corrupted, mask, frac=None, t=None, noise=None):
        """MSE between true and predicted noise. Input = [x_t, corrupted, mask] (7 channels)."""
        b = clean.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (b,), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        xt = self.q_sample(clean, t, noise)
        eps = model(torch.cat([xt, corrupted, mask], 1), t, frac)
        return ((eps - noise) ** 2).mean()

    # ---- reverse process ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, corrupted, mask, frac=None, steps: int | None = None, paste_known: bool = False,
               generator: torch.Generator | None = None, return_raw: bool = False):
        """Ancestral sampling. `steps` < T uses evenly spaced respaced timesteps
        (Nichol & Dhariwal 2021); the network is always queried with the ORIGINAL timestep index.
        Output keeps every visible pixel exactly: out = mask * x0 + (1 - mask) * corrupted."""
        dev = corrupted.device
        ts = np.arange(self.T) if steps is None or steps >= self.T else np.unique(
            np.linspace(0, self.T - 1, steps).round().astype(int))
        sch = Schedule(self.betas, ts)
        x = torch.randn(corrupted.shape, device=dev, generator=generator)
        for i in reversed(range(len(ts))):
            t = torch.full((x.shape[0],), int(ts[i]), device=dev, dtype=torch.long)
            eps = model(torch.cat([x, corrupted, mask], 1), t, frac)
            x0 = ((x - float(sch.sqrt_1m_ab[i]) * eps) / float(sch.sqrt_ab[i])).clamp(-1, 1)
            mean = float(sch.post_c1[i]) * x0 + float(sch.post_c2[i]) * x
            if i > 0:
                x = mean + float(np.sqrt(sch.post_var[i])) * torch.randn(x.shape, device=dev, generator=generator)
                if paste_known:  # RePaint-style: re-noise the known pixels to the previous level
                    known = float(sch.sqrt_ab[i - 1]) * corrupted + float(sch.sqrt_1m_ab[i - 1]) * torch.randn(
                        x.shape, device=dev, generator=generator)
                    x = mask * x + (1 - mask) * known
            else:
                x = mean  # last step is noiseless (post_var[0] == 0)
        out = mask * x + (1.0 - mask) * corrupted
        return (out, x) if return_raw else out


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)
