"""Datasets: CelebA (real runs) and synthetic faces (smoke test), with dynamic / fixed masks.

Images are held in memory as uint8 arrays (N, H, W, 3). A sample is a dict:
    clean      (3, H, W) float in [-1, 1]   target
    mask       (1, H, W) float {0, 1}       1 = missing
    corrupted  (3, H, W)                    clean with the hole zeroed
    frac       ()                           true missing fraction of this mask
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import FixedMaskSet, TrainMaskSampler
from .synthetic import make_synthetic_split


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(img_u8, copy=True)).permute(2, 0, 1).float() / 127.5 - 1.0

def corrupt(clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return clean * (1.0 - mask)


class TrainInpaintDataset(Dataset):
    """Dynamic masks: mask for (epoch, index) is a deterministic function of the seed, but
    differs across epochs. Call set_epoch(e) at the start of every epoch."""

    def __init__(self, images: np.ndarray, size: int, lo=0.10, hi=0.40, seed=0, flip=True):
        self.images, self.sampler, self.seed, self.flip, self.epoch = images, TrainMaskSampler(size, lo, hi), seed, flip, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1, self.epoch, i]))
        img = self.images[i]
        if self.flip and rng.random() < 0.5:
            img = img[:, ::-1]
        m, frac = self.sampler(rng)
        clean = to_tensor(np.ascontiguousarray(img))
        mask = torch.from_numpy(m)[None]
        return {"clean": clean, "mask": mask, "corrupted": corrupt(clean, mask), "frac": torch.tensor(frac, dtype=torch.float32)}


class FixedInpaintDataset(Dataset):
    """Evaluation set: image i is always paired with fixed mask i of one (type, level)."""

    def __init__(self, images: np.ndarray, maskset: FixedMaskSet):
        assert len(images) >= len(maskset)
        self.images, self.maskset = images, maskset

    def __len__(self) -> int:
        return len(self.maskset)

    def __getitem__(self, i: int) -> dict:
        clean = to_tensor(self.images[i])
        mask = torch.from_numpy(self.maskset[i])[None]
        return {"clean": clean, "mask": mask, "corrupted": corrupt(clean, mask), "frac": mask.mean()}


# ---------------------------------------------------------------- split loading
def load_splits(source: str, size: int, n_train: int, n_val: int, n_test: int, root: str | None = None, seed: int = 0):
    """Returns dict split -> uint8 array (N, size, size, 3).

    source == 'synthetic': procedural faces, disjoint seeds per split.
    source == 'celeba'   : arrays produced by scripts/prepare_celeba.py in `root`
                           (celeba{size}_{split}.npy); the official partitions are respected there.
    """
    if source == "synthetic":
        return {
            "train": make_synthetic_split(n_train, size, seed * 10 + 1),
            "val": make_synthetic_split(n_val, size, seed * 10 + 2),
            "test": make_synthetic_split(n_test, size, seed * 10 + 3),
        }
    if source == "celeba":
        out = {}
        for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
            path = os.path.join(root, f"celeba{size}_{split}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing; run scripts/prepare_celeba.py first")
            arr = np.load(path, mmap_mode="r")
            out[split] = np.ascontiguousarray(arr[:n])
        return out
    raise ValueError(source)


class MixedFixedDataset(Dataset):
    """Fixed validation set: image i gets a fixed mask from condition (i mod 6) over the
    6 (type, level) conditions, so one scalar validation loss covers all damage settings."""

    def __init__(self, images: np.ndarray, n: int, size: int, seed: int):
        from .masks import LEVELS, MASK_TYPES
        conds = [(k, l) for k in MASK_TYPES for l in LEVELS]
        per = -(-n // len(conds))
        sets = [FixedMaskSet(k, l, per, size, seed) for k, l in conds]
        self.images = images[:n]
        self.masks = np.stack([sets[i % len(conds)][i // len(conds)] for i in range(n)])

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, i: int) -> dict:
        clean = to_tensor(self.images[i])
        mask = torch.from_numpy(self.masks[i])[None]
        return {"clean": clean, "mask": mask, "corrupted": corrupt(clean, mask), "frac": mask.mean()}
