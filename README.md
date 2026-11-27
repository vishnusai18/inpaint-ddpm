# Diffusion-based image restoration with severity-aware conditioning

MSML612 Deep Learning group project. A small conditional DDPM reconstructs regions removed from
face images (CelebA, 64x64). We compare a direct U-Net baseline, a standard conditional DDPM, and a
**severity-aware** DDPM that is also told the fraction of missing pixels, on rectangular and
irregular masks covering about 10%, 20% and 40% of the image.

> **Status (final).** All three models were trained on CelebA 64x64 for 100k steps on Zaratan GPUs and
> evaluated with EMA weights on 500 fixed test images per condition. The final numbers are in
> [`results/full-500/`](results/full-500/). `results/celeba32/` holds an earlier, much smaller local run
> (32x32, 1.5k steps, 100 test images) kept for reference only.

## Results (CelebA 64x64, 100k steps, 500 test images per condition, 250 sampling steps)

Averaged over all six conditions (2 mask types x 3 missing-area levels). PSNR/SSIM/LPIPS are computed on the hole
only; FID is on the full image.

| model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|---|
| U-Net | **22.66** | **0.776** | 0.0371 | 15.55 |
| DDPM | 19.73 | 0.669 | 0.0344 | **12.68** |
| severity-aware DDPM | 19.75 | 0.670 | **0.0341** | 12.71 |
| mean fill (baseline) | 12.13 | 0.220 | 0.2358 | 83.04 |

- **Perception-distortion trade-off.** The U-Net has the best PSNR and SSIM in all six conditions. Both diffusion models
  have lower FID than the U-Net in all six conditions, so their outputs are more realistic as a distribution.
- **Severity conditioning did not help in a meaningful way.** Against the standard DDPM, paired differences are tiny
  (at most about 0.07 dB PSNR and 0.003 SSIM) and inconsistent in sign across conditions (`results/full-500/paired.csv`).
- All learned models are far better than mean fill, and every model gets worse as the hole grows from 10% to 40%.
- FID from 500 images is biased upwards and only comparable between models in this table, not with published FID numbers.

Per-condition numbers: `summary.csv`, `fid_summary.csv`; paired bootstrap CIs and Wilcoxon tests: `paired.csv`;
figures: `results/full-500/figures/`.

## Layout
```
inpaint/
  masks.py        rectangular + irregular masks; dynamic training masks; fixed val/test mask sets
  data.py         CelebA / synthetic loaders, corruption, train (dynamic) and eval (fixed) datasets
  synthetic.py    procedural toy faces for smoke tests
  unet.py         shared U-Net backbone (+ zero-initialised severity branch)
  diffusion.py    DDPM (epsilon-prediction), ancestral sampler, timestep respacing, EMA
  models.py       unet | ddpm | ddpm_sev
  train.py        training CLI (resume, EMA, validation loss, CSV log)
  evaluate.py     fixed-mask evaluation, hole-only PSNR/SSIM/LPIPS, paired comparisons, example grids
  fid_evaluate.py full-image FID per model, mask type and level (resumable)
  metrics.py      hole-restricted metrics
  verify.py       verification suite (23 groups of checks)
  make_figures.py loss curves, metric curves, qualitative grids
scripts/          prepare_celeba.py, run_smoke.sh, slurm/{train,eval}.sbatch
tests/            pytest (incl. fault-injection tests for the verification suite)
```

## Models (identical backbone and training setup so only the studied factor differs)
| name | input | output | extra |
|---|---|---|---|
| `unet` | corrupted RGB + mask (4 ch) | RGB | direct reconstruction, L1 with hole weighted 6x |
| `ddpm` | noisy RGB + corrupted RGB + mask (7 ch) | noise | timestep embedding |
| `ddpm_sev` | same 7 ch | noise | + MLP on missing fraction, added to the timestep embedding; last layer zero-initialised so it equals `ddpm` at initialisation |

Sampling keeps every visible pixel exactly (`out = mask * x0 + (1 - mask) * corrupted`). Metrics are computed over the hole only.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                         # ~2 min on a laptop CPU
python -m inpaint.verify --out results/verification.json
STEPS=1200 ./scripts/run_smoke.sh runs/smoke      # ~1 h on 2 CPU cores, synthetic faces, 32x32
python -m inpaint.evaluate --runs runs/smoke --preset smoke --out results/smoke --no-lpips
python -m inpaint.make_figures --runs runs/smoke --results results/smoke --out results/smoke/figures
```

## CelebA on Zaratan
1. Download CelebA (img_align_celeba + list_eval_partition) and build the subsets
   (30k train / 3k val / 3k test, official partitions, centre crop, 64x64):
   `python scripts/prepare_celeba.py --raw /path/to/celeba --out data --size 64`
2. Cache LPIPS weights once on a node with internet: `python -c "import lpips; lpips.LPIPS(net='alex')"`.
3. Edit the account/partition/module lines in `scripts/slurm/*.sbatch` (they are placeholders), then
   `for m in unet ddpm ddpm_sev; do sbatch --export=MODEL=$m scripts/slurm/train.sbatch; done`
   and, when they finish, `sbatch scripts/slurm/eval.sbatch`.

`--preset full` = 64x64, base width 64, mults (1,2,3,4), T=1000, 250 sampling steps (respaced), 100k steps.

## What has and has not been verified
Verified (see `results/verification.json`, `tests/`): mask geometry and area control, determinism, fixed-vs-dynamic masks,
corruption correctness, split disjointness, diffusion math against independent derivations, oracle-sampler recovery,
visible-pixel preservation, severity-branch identity at initialisation, metric correctness, overfit-one-batch, and
fault injection (each injected bug is caught). The full CelebA training, LPIPS and FID evaluation ran on Zaratan GPUs.

Not yet in the repo: the Zaratan job logs and per-step loss logs for the 100k-step runs, and the script that draws
`results/full-500/figures/fid_vs_level.*`. The `scripts/slurm/*.sbatch` files still contain placeholder account lines.

## References
Ho et al., DDPM (NeurIPS 2020); Nichol & Dhariwal, Improved DDPM (ICML 2021); Lugmayr et al., RePaint (CVPR 2022);
Ju et al., BrushNet (ECCV 2024); Liu et al., CelebA (ICCV 2015); Zhang et al., LPIPS (CVPR 2018); lucidrains/denoising-diffusion-pytorch (reference implementation).
