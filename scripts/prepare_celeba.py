#!/usr/bin/env python
"""Build the CelebA subsets used in the experiments (official partitions respected).

Expected raw layout (the standard CelebA release / Kaggle copy):
    <raw>/img_align_celeba/000001.jpg ...       (178x218 aligned images; the Kaggle copy nests one level deeper)
    <raw>/list_eval_partition.txt  (or .csv)    (0 = train, 1 = val, 2 = test)

    python scripts/prepare_celeba.py --raw /path/to/celeba --out data --size 64 \
        --n-train 30000 --n-val 3000 --n-test 3000

Each split is a seeded random sample from its OWN official partition, centre-cropped to 178x178
and resized to size x size. Output: data/celeba{size}_{split}.npy (uint8, N x size x size x 3) and
data/celeba_{split}_files.txt. The script asserts that the three file lists are disjoint.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def read_partition(raw: str) -> dict:
    cands = glob.glob(os.path.join(raw, "**", "list_eval_partition.*"), recursive=True)
    if not cands:
        raise SystemExit("list_eval_partition.txt/.csv not found under --raw")
    parts = {0: [], 1: [], 2: []}
    for line in open(cands[0]):
        tok = line.replace(",", " ").split()
        if len(tok) == 2 and tok[1] in "012" and tok[0].endswith(".jpg"):
            parts[int(tok[1])].append(tok[0])
    return parts


def find_image_dir(raw: str) -> str:
    for d in glob.glob(os.path.join(raw, "**", "img_align_celeba"), recursive=True):
        if glob.glob(os.path.join(d, "*.jpg")):
            return d
    raise SystemExit("img_align_celeba/*.jpg not found under --raw")


def load(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    parts, img_dir = read_partition(a.raw), find_image_dir(a.raw)
    names = {"train": parts[0], "val": parts[1], "test": parts[2]}
    print({k: len(v) for k, v in names.items()}, "official partition sizes (expect 162770 / 19867 / 19962)")
    rng = np.random.default_rng(a.seed)
    chosen = {}
    for split, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        pool = sorted(names[split])
        chosen[split] = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    assert not (set(chosen["train"]) & set(chosen["val"])) and not (set(chosen["train"]) & set(chosen["test"])) \
        and not (set(chosen["val"]) & set(chosen["test"])), "splits overlap!"
    print({k: len(v) for k, v in chosen.items()}, "selected; splits are disjoint")
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    for split, files in chosen.items():
        arr = np.stack([load(os.path.join(img_dir, f), a.size) for f in files])
        np.save(os.path.join(a.out, f"celeba{a.size}_{split}.npy"), arr)
        open(os.path.join(a.out, f"celeba_{split}_files.txt"), "w").write("\n".join(files))
        print(split, arr.shape)


if __name__ == "__main__":
    main()
