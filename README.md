# Diffusion-based image restoration with severity-aware conditioning

MSML612 Deep Learning group project. A small conditional DDPM reconstructs regions removed from
face images (CelebA, 64x64). We compare a direct U-Net baseline, a standard conditional DDPM, and a
**severity-aware** DDPM that is also told the fraction of missing pixels, on rectangular and
irregular masks covering about 10%, 20% and 40% of the image.

> **Status (interim).** The full pipeline is implemented and verified. All numbers produced inside the
> development sandbox come from a **CPU smoke test on procedurally generated toy faces at 32x32**
> (no GPU, CelebA hosts unreachable there). They demonstrate that the pipeline trains and evaluates; they are
> **not CelebA results**. CelebA runs are scheduled on Zaratan (`scripts/slurm/`).

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

## What has and has not been verified here
Verified (see `results/verification.json`, `tests/`): mask geometry and area control, determinism, fixed-vs-dynamic masks,
corruption correctness, split disjointness, diffusion math against independent derivations, oracle-sampler recovery,
visible-pixel preservation, severity-branch identity at initialisation, metric correctness, overfit-one-batch, and
fault injection (each injected bug is caught).
Not verified in the sandbox: CelebA loading on the real files (only on a fake folder), LPIPS (backbone weights
unreachable), GPU/AMP code paths, the Slurm scripts, and anything about image quality on real faces.

## References
Ho et al., DDPM (NeurIPS 2020); Nichol & Dhariwal, Improved DDPM (ICML 2021); Lugmayr et al., RePaint (CVPR 2022);
Ju et al., BrushNet (ECCV 2024); Liu et al., CelebA (ICCV 2015); Zhang et al., LPIPS (CVPR 2018); lucidrains/denoising-diffusion-pytorch (reference implementation).
