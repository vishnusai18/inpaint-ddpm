import numpy as np
import torch

from inpaint.data import FixedInpaintDataset, MixedFixedDataset, TrainInpaintDataset
from inpaint.masks import FixedMaskSet
from inpaint.models import build_model
from inpaint.unet import UNet, count_params


class A:  # minimal args namespace
    base, mults, attn_res, size = 16, (1, 2), (8,), 16


def test_models_forward_shapes():
    x7, x4, t = torch.randn(2, 7, 16, 16), torch.randn(2, 4, 16, 16), torch.tensor([1, 5])
    assert build_model("ddpm", A)(x7, t).shape == (2, 3, 16, 16)
    assert build_model("ddpm_sev", A)(x7, t, torch.tensor([.1, .4])).shape == (2, 3, 16, 16)
    assert build_model("unet", A)(x4).shape == (2, 3, 16, 16)


def test_full_config_builds_and_runs_at_64():
    net = UNet(7, 3, base=64, mults=(1, 2, 3, 4), image_size=64, attn_res=(16,))
    assert net(torch.randn(1, 7, 64, 64), torch.tensor([10])).shape == (1, 3, 64, 64)
    assert 15e6 < count_params(net) < 30e6


def test_fixed_eval_pairs_image_with_same_mask():
    imgs = np.random.default_rng(0).integers(0, 255, (20, 16, 16, 3), dtype=np.uint8)
    ds = FixedInpaintDataset(imgs, FixedMaskSet("rect", 0.2, 20, 16, 1))
    assert torch.equal(ds[3]["mask"], ds[3]["mask"]) and not torch.equal(ds[3]["mask"], ds[4]["mask"])


def test_mixed_val_dataset_covers_all_six_conditions():
    imgs = np.zeros((36, 16, 16, 3), np.uint8)
    ds = MixedFixedDataset(imgs, 36, 16, seed=0)
    fr = np.array([float(ds[i]["mask"].mean()) for i in range(12)])
    assert set(np.round(fr[:6], 1)) == {0.1, 0.2, 0.4}
