"""Mask generation. Convention everywhere in this repo: mask == 1 marks MISSING (hole)
pixels, mask == 0 marks visible pixels. Masks are float32 arrays of shape (H, W).

* rectangular: one axis-aligned rectangle with random aspect ratio and position.
* irregular: brush-like random-walk strokes, grown stamp by stamp until the requested
  missing fraction is reached (so area control is tight even for free-form shapes).

Training masks are drawn on the fly (different every epoch) with a missing fraction
sampled uniformly from [lo, hi]. Validation / test masks are generated once from a fixed
seed per (type, level) and never change, so every model sees identical damage.
"""
from __future__ import annotations

import numpy as np

MASK_TYPES = ("rect", "irregular")
LEVELS = (0.10, 0.20, 0.40)


def rect_mask(size: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    target = frac * size * size
    for _ in range(50):
        aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))  # h / w
        w = int(round(np.sqrt(target / aspect)))
        h = int(round(target / max(w, 1)))
        if 1 <= h <= size and 1 <= w <= size:
            break
    else:  # fall back to a square
        w = h = int(round(np.sqrt(target)))
    h, w = min(max(h, 1), size), min(max(w, 1), size)
    y = int(rng.integers(0, size - h + 1))
    x = int(rng.integers(0, size - w + 1))
    m = np.zeros((size, size), dtype=np.float32)
    m[y : y + h, x : x + w] = 1.0
    return m


def _disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    return (yy * yy + xx * xx) <= radius * radius


def irregular_mask(size: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Brush strokes: random walks that stamp discs; stop as soon as coverage >= frac."""
    target = frac * size * size
    m = np.zeros((size, size), dtype=bool)
    covered = 0
    radius = max(2, int(round(size * rng.uniform(0.05, 0.09))))
    disk = _disk(radius)
    guard = 0
    while covered < target and guard < 200000:
        # start a new stroke
        y, x = rng.uniform(0, size, 2)
        angle = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(10, 40))):
            angle += rng.normal(0, 0.6)
            y += np.sin(angle) * radius * 0.7
            x += np.cos(angle) * radius * 0.7
            y = float(np.clip(y, 0, size - 1))
            x = float(np.clip(x, 0, size - 1))
            yi, xi = int(round(y)), int(round(x))
            y0, y1 = max(yi - radius, 0), min(yi + radius + 1, size)
            x0, x1 = max(xi - radius, 0), min(xi + radius + 1, size)
            d = disk[(y0 - yi + radius) : (y1 - yi + radius), (x0 - xi + radius) : (x1 - xi + radius)]
            m[y0:y1, x0:x1] |= d
            covered = int(m.sum())
            guard += 1
            if covered >= target:
                break
    return m.astype(np.float32)


def make_mask(kind: str, size: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "rect":
        return rect_mask(size, frac, rng)
    if kind == "irregular":
        return irregular_mask(size, frac, rng)
    raise ValueError(f"unknown mask type {kind!r}")


class TrainMaskSampler:
    """Dynamic training masks: random type (rect/irregular), fraction ~ U[lo, hi]."""

    def __init__(self, size: int, lo: float = 0.10, hi: float = 0.40):
        self.size, self.lo, self.hi = size, lo, hi

    def __call__(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        kind = MASK_TYPES[int(rng.integers(len(MASK_TYPES)))]
        frac = float(rng.uniform(self.lo, self.hi))
        m = make_mask(kind, self.size, frac, rng)
        return m, float(m.mean())  # the TRUE missing fraction is what the model is told


class FixedMaskSet:
    """n masks of one (type, level), generated once from a seed. Index i is always the same."""

    def __init__(self, kind: str, level: float, n: int, size: int, seed: int):
        self.kind, self.level, self.n, self.size, self.seed = kind, level, n, size, seed
        # separate stream per (kind, level) so adding a level never changes the others
        ss = np.random.SeedSequence([seed, MASK_TYPES.index(kind), int(round(level * 1000))])
        rng = np.random.default_rng(ss)
        self.masks = np.stack([make_mask(kind, size, level, rng) for _ in range(n)]).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> np.ndarray:
        return self.masks[i]
