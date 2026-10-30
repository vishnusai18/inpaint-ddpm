"""Procedural toy faces for smoke tests (NOT CelebA).

The sandbox used to develop this repo has no GPU and cannot reach the CelebA hosts, so the
pipeline is smoke-tested on programmatically drawn faces: symmetric structure (eyes, brows,
nose, mouth, hair, glasses) with random identity attributes. They are structured enough
that filling a hole requires modelling facial layout, but they are not natural images and
no result on them should be reported as a CelebA result.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

_SKIN = [(255, 224, 189), (241, 194, 125), (224, 172, 105), (198, 134, 66), (141, 85, 36),
         (255, 205, 148), (234, 192, 134), (176, 112, 74)]
_HAIR = [(20, 15, 10), (60, 40, 25), (110, 70, 35), (190, 150, 80), (150, 40, 25), (200, 200, 200), (30, 30, 60)]
_IRIS = [(60, 90, 160), (70, 120, 70), (110, 70, 40), (40, 40, 40), (100, 130, 150)]
_LIP = [(180, 70, 80), (200, 100, 100), (150, 60, 60), (190, 110, 110)]


def _jit(c, rng, s=10):
    return tuple(int(np.clip(v + rng.integers(-s, s + 1), 0, 255)) for v in c)


def render_face(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    S = 256
    img = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(img)
    # background vertical gradient
    top = np.array([rng.integers(90, 230) for _ in range(3)])
    bot = np.array([rng.integers(90, 230) for _ in range(3)])
    for y in range(S):
        d.line([(0, y), (S, y)], fill=tuple(int(v) for v in top + (bot - top) * y / S))
    cx = 128 + rng.integers(-12, 13)
    cy = 130 + rng.integers(-10, 11)
    rx = int(rng.integers(62, 80))
    ry = int(rng.integers(80, 98))
    skin = _jit(_SKIN[int(rng.integers(len(_SKIN)))], rng)
    hair = _jit(_HAIR[int(rng.integers(len(_HAIR)))], rng, 12)
    # hair back
    if rng.random() > 0.1:
        d.ellipse([cx - rx - 12, cy - ry - 22, cx + rx + 12, cy + ry * 0.3], fill=hair)
    # ears
    ey = cy + int(rng.integers(-8, 9))
    for sx in (-1, 1):
        d.ellipse([cx + sx * rx - 10, ey - 16, cx + sx * rx + 10, ey + 16], fill=skin)
    # head
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=skin)
    # fringe
    if rng.random() > 0.3:
        h = int(rng.integers(18, 40))
        d.pieslice([cx - rx, cy - ry, cx + rx, cy + ry], 180, 360, fill=hair)
        d.ellipse([cx - rx + 14, cy - ry + h, cx + rx - 14, cy + ry * 0.2], fill=skin)
    # eyes + brows
    eye_y = cy - int(0.18 * ry) + int(rng.integers(-5, 6))
    dx = int(0.40 * rx) + int(rng.integers(-4, 5))
    iris = _IRIS[int(rng.integers(len(_IRIS)))]
    ew, eh = int(rng.integers(12, 17)), int(rng.integers(7, 11))
    for sx in (-1, 1):
        ex = cx + sx * dx
        d.ellipse([ex - ew, eye_y - eh, ex + ew, eye_y + eh], fill=(250, 250, 250))
        d.ellipse([ex - 7, eye_y - 7, ex + 7, eye_y + 7], fill=iris)
        d.ellipse([ex - 3, eye_y - 3, ex + 3, eye_y + 3], fill=(10, 10, 10))
        tilt = int(rng.integers(-6, 7))
        d.line([(ex - ew, eye_y - eh - 12 + sx * tilt), (ex + ew, eye_y - eh - 12 - sx * tilt)],
               fill=tuple(int(v * 0.8) for v in hair), width=int(rng.integers(4, 8)))
    # nose
    ny = cy + int(0.15 * ry)
    shade = tuple(int(v * 0.82) for v in skin)
    d.polygon([(cx, eye_y + 8), (cx - 9, ny), (cx + 9, ny)], fill=shade)
    # mouth (smile / flat / frown via arc offset)
    my = cy + int(0.52 * ry)
    mw = int(rng.uniform(0.28, 0.46) * rx)
    curve = int(rng.integers(-10, 16))
    lip = _LIP[int(rng.integers(len(_LIP)))]
    pts = [(cx - mw + i * (2 * mw) // 12, my + int(curve * (1 - ((i - 6) / 6) ** 2))) for i in range(13)]
    d.line(pts, fill=lip, width=int(rng.integers(5, 10)))
    # glasses
    if rng.random() < 0.25:
        r = int(rng.integers(20, 26))
        for sx in (-1, 1):
            d.ellipse([cx + sx * dx - r, eye_y - r, cx + sx * dx + r, eye_y + r], outline=(25, 25, 25), width=4)
        d.line([(cx - dx + r, eye_y), (cx + dx - r, eye_y)], fill=(25, 25, 25), width=4)
    arr = np.asarray(img.resize((size, size), Image.LANCZOS)).astype(np.float32)
    # lighting + noise
    grad = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(1.0, 1.15), size)[None, :, None]
    if rng.random() < 0.5:
        grad = grad[:, ::-1]
    arr = arr * grad + rng.normal(0, 2.0, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_synthetic_split(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([render_face(rng, size) for _ in range(n)])
