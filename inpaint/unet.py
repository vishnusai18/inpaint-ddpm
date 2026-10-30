"""Compact U-Net used by all three models (same backbone => controlled comparison).

* UNet(in_ch=7, use_time=True)                  conditional DDPM noise predictor
* UNet(in_ch=7, use_time=True, use_severity=True) severity-aware DDPM (adds an MLP on the
  missing-pixel fraction, summed into the timestep embedding; its last layer is
  zero-initialised so, at initialisation, it computes exactly the same function as the
  standard DDPM and is built last so shared weights are identical under the same seed)
* UNet(in_ch=4, use_time=False)                 direct reconstruction baseline
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = x.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int | None, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout) if emb_dim else None
        self.n2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb=None):
        h = self.c1(F.silu(self.n1(x)))
        if self.emb is not None and emb is not None:
            h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 7, out_ch: int = 3, base: int = 64, mults=(1, 2, 3, 4), n_res: int = 2,
                 image_size: int = 64, attn_res=(16,), use_time: bool = True, use_severity: bool = False,
                 dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.use_time, self.use_severity, self.base = use_time, use_severity, base
        emb_dim = base * 4 if use_time else None
        if use_time:
            self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        chans, skips, res, ch = [], [base], image_size, base
        for lvl, m in enumerate(mults):
            for _ in range(n_res):
                blocks = [ResBlock(ch, base * m, emb_dim, groups, dropout)]
                ch = base * m
                if res in attn_res:
                    blocks.append(Attention(ch, groups=groups))
                self.down.append(nn.ModuleList(blocks))
                skips.append(ch)
            if lvl < len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skips.append(ch)
                res //= 2
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, groups, dropout), Attention(ch, groups=groups),
                                  ResBlock(ch, ch, emb_dim, groups, dropout)])
        for lvl, m in reversed(list(enumerate(mults))):
            for i in range(n_res + 1):
                blocks = [ResBlock(ch + skips.pop(), base * m, emb_dim, groups, dropout)]
                ch = base * m
                if res in attn_res:
                    blocks.append(Attention(ch, groups=groups))
                if lvl > 0 and i == n_res:
                    blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blocks.append(nn.Conv2d(ch, ch, 3, padding=1))
                    res *= 2
                self.up.append(nn.ModuleList(blocks))
        self.out = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))
        # severity branch is created LAST so every shared parameter draws the same RNG stream
        if use_severity:
            assert use_time, "severity embedding is added to the timestep embedding"
            self.sev_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
            nn.init.zeros_(self.sev_mlp[2].weight)
            nn.init.zeros_(self.sev_mlp[2].bias)

    def forward(self, x, t=None, frac=None):
        emb = None
        if self.use_time:
            emb = self.time_mlp(sinusoidal(t, self.base))
            if self.use_severity:
                assert frac is not None, "severity-aware model needs the missing fraction"
                emb = emb + self.sev_mlp(sinusoidal(frac * 1000.0, self.base))
        h = self.inc(x)
        hs = [h]
        for blocks in self.down:
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
            hs.append(h)
        for b in self.mid:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], 1)
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return self.out(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
