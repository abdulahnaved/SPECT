## Background

SPECT uses a lead collimator to restrict which gamma photons reach the detector. Simulating photon transport through the collimator with full Monte Carlo physics (OpenGATE / Geant4) is accurate but slow — it is the main computational bottleneck in realistic SPECT system modelling.

This project trains a surrogate that, given an incoming photon (position, direction, energy), either predicts that the photon is absorbed or samples a realistic outgoing photon. Two generative families are supported per class: WGAN-GP and Flow Matching with an RK4 ODE solver. Discrete output structure (Pb K X-ray lines, geometry-induced phi/theta peaks) is handled outside the generator with empirical PMF sampling and rank-preserving CDF warping.

## Collimator Geometry

```
material:          G4_Pb (lead)
outer dimensions:  550 × 405 × 23.8 mm
hole diameter:     1.2 mm
septum thickness:  0.2 mm
hole count:        ~128,000 parallel cylindrical holes, hex-packed
```

## Physics

Fluorescence is enabled in the physics list with low production cuts in lead. This makes lead characteristic X-rays (Pb Kα ≈ 73–75 keV, Pb Kβ ≈ 85–87 keV) appear as a distinct photon class in the output spectrum.

## Data

Each photon is represented as 11 values per row. Inputs (incoming) and outputs (outgoing) match within a single GATE batch via `EventID`/`TrackID`. A `is_secondary` flag distinguishes original photons from secondaries produced inside the lead.

```
in_x       mm     entry position X
in_y       mm     entry position Y
in_theta   rad    entry polar angle
in_phi     rad    entry azimuthal angle
in_E       keV    entry kinetic energy

out_x      mm     exit position X
out_y      mm     exit position Y
out_theta  rad    exit polar angle
out_phi    rad    exit azimuthal angle
out_E      keV    exit kinetic energy   (0 if absorbed)

is_secondary       1 if outgoing TrackID > 1, else 0
```

Photon classes (defined from physics, not energy thresholds):

```
blocked   out_E == 0
direct    passed, TrackID = 1, |out_E - in_E| / in_E < 5%
xray      passed, TrackID > 1                       (Pb K X-ray emitted in lead)
scatter   passed, TrackID = 1, not direct           (Compton-scattered primary)
```

Transmission rate from the 10B primary simulation: ~0.13%.

## Model Architecture

Two-stage pipeline.

**Stage 1 — Multiclass classifier**

A 4-class MLP that routes each incoming photon to its physical fate:

```
input:  5  (in_x, in_y, in_theta, in_phi, in_E)
hidden: 256 × 256 × 256 × 256
output: 4 class logits  (blocked / direct / xray / scatter)
```

**Stage 2 — Per-class conditional generator**

One generator per passed class. Each generator samples a realistic outgoing photon conditioned on the incoming photon. Two architectures are supported:

```
WGAN-GP (train_gan.py):
  generator: 5 + z_dim → 256 × 256 × 256 × 256 → 5
  critic:    10        → 256 × 256 × 256 × 256 → 1
  training:  WGAN-GP, gradient penalty λ=10, 5 critic steps per generator step,
             Adam lr=1e-4, batch 512

Flow Matching (train_flow.py):
  velocity:  5 + 5 + t_dim → 256 × 256 × 256 × 256 → 5
             (incoming, current x_t, sinusoidal time embedding)
  training:  linear-path conditional flow matching, MSE on velocity,
             Adam lr=3e-4, cosine schedule, batch 512
  inference: RK4 ODE solver from x(0) ~ N(0, I) to x(1), 50 steps
```

Flow Matching trains as a stable supervised regression (no adversarial loop, no mode collapse). On every output it tested it beats the GAN — direct position MAE drops sub-millimeter, xray and scatter position MAE roughly halves without needing any helper flags. See `EXPERIMENT_LOG.md` for the full comparison.

**Stage 3 — Discrete-structure handling at inference**

Neither a continuous GAN nor a flow-matching ODE can reproduce delta-function-like distributions exactly. Two post-processing tools fix this without breaking the conditional structure the generator learned (both flags are wired into `train_gan.py` and `train_flow.py`):

- **PMF sampling (`--fix-energy`, `--fix-phi`)** — drop the variable from the generator's outputs, sample it independently from the empirical distribution measured during training. Works only when the variable is genuinely independent of the rest (Pb K X-ray energies are atomic constants, so they pass this test; phi does not).

- **CDF warping (`--warp-phi`, `--warp-theta`)** — keep the variable in the generator's outputs (so correlations are preserved), then rank-warp the marginal at inference so it exactly matches the empirical distribution. Sharp peaks appear automatically while the joint structure stays intact.

The empirical PMFs and CDFs are measured once from training data and saved inside the model file. No live Monte Carlo connection is needed at deployment.

**Inference (Markov-chain sampling)**

```
incoming photon
      ↓
  classifier → sample class
      ↓
  blocked  → no output (absorbed)
  direct   → direct GAN     → outgoing photon
  xray     → xray GAN       → outgoing photon (with energy from PMF, phi from CDF warp)
  scatter  → scatter GAN    → outgoing photon (with phi from CDF warp)
```

## Best Models

For the final 10B simulation with TrackID-based classification:

| Class | Config | val Wasserstein | out_x MAE | out_y MAE | out_E MAE |
|-------|--------|----------------|-----------|-----------|-----------|
| direct  | plain GAN (z=32)               | -0.10 | 5.5 mm | 3.0 mm | 1.4 keV |
| xray    | `--fix-energy --warp-phi` (z=64) | -0.22 | 5.4 mm | 11.4 mm | 4.9 keV |
| scatter | `--warp-phi` (z=64)              | -0.36 | 10.3 mm | 8.4 mm | 11.6 keV |

MAE on its own is misleading for a generative model — what matters is the full output distribution. Histogram overlap with Monte Carlo is the actual quality metric and is essentially perfect for direct, xray, and scatter on all five output variables (saved plots are in `ml_artifacts/<run>/gan_histograms_final.png`).

## Repository Structure

```
collimator_transport/       OpenGATE simulation package
  geometry.py               collimator and world geometry (hex-packed holes)
  source.py                 gamma flood source
  physics.py                Geant4 physics list settings
  actors.py                 phase-space scoring actors
  main.py                   run one simulation batch
  batch_worker.py           subprocess entry point for one batch
  run.py                    parallel multi-batch runner

postprocess.py              match incoming/outgoing photons, export .npy
train_prototype.py          MLP classifier + regressor (architecture search)
train_multiclass.py         4-class classifier + per-class regressors
train_gan.py                WGAN-GP conditional GAN per class
                            (--fix-energy, --fix-phi, --warp-phi, --warp-theta)
train_flow.py               Conditional Flow Matching per class, RK4 ODE solver
                            (--fix-energy, --fix-phi, --warp-phi, --warp-theta)
validate_model.py           histogram and calibration plots

EXPERIMENT_LOG.md           experiment history with results and reasoning
```

## Setup

The project uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Main dependencies: `opengate`, `uproot`, `numpy`, `torch`, `matplotlib`.

## Running A Simulation

```bash
uv run python -m collimator_transport.run \
    --total 10000000000 \
    --batches 1000 \
    --workers 36 \
    --output-dir output
```

- `--total`: total primary photons
- `--batches`: independent batches, each with its own seed (`base_seed + batch_id`)
- `--workers`: parallel GATE instances (one per CPU core)

Batches are statistically independent and can be merged after postprocessing. Long runs go in a tmux session.

On the very first run, Geant4 downloads physics data. Run a single-worker warm-up before launching parallel workers — otherwise parallel downloads can corrupt each other.

## Postprocessing

Matches incoming and outgoing photons inside each batch (matching across batches would mis-pair because EventID is only unique within a single GATE process), then writes one `.npy` array.

```bash
# all photons
uv run python postprocess.py --output-dir output --out-file postprocessed.npy

# only one class (memory-efficient for rare classes)
uv run python postprocess.py --output-dir output --out-file xray.npy    --class-filter xray
uv run python postprocess.py --output-dir output --out-file scatter.npy --class-filter scatter
```

## Training

```bash
# direct
uv run python train_gan.py --data postprocessed.npy --class direct --z-dim 32 --epochs 300

# xray  (best: fix-energy + warp-phi)
uv run python train_gan.py --data xray.npy --class xray --z-dim 64 --epochs 300 \
    --fix-energy --warp-phi

# scatter (best: warp-phi only)
uv run python train_gan.py --data scatter.npy --class scatter --z-dim 64 --epochs 300 \
    --warp-phi
```

Each run produces histogram plots every 20 epochs, a `report.json`, and a `generator.pt` checkpoint with the GAN weights plus any PMFs/CDFs needed at inference.

Flow Matching uses the same flags and produces `flow.pt` instead of `generator.pt`:

```bash
# direct
uv run python train_flow.py --data postprocessed.npy --class direct --epochs 200

# xray
uv run python train_flow.py --data xray.npy --class xray --epochs 200 \
    --fix-energy --warp-phi

# scatter
uv run python train_flow.py --data scatter.npy --class scatter --epochs 200 \
    --warp-phi
```

## Notes

- `EXPERIMENT_LOG.md` contains the chronological record of what was tried and what worked, including the failures (the energy-threshold class definition, `--fix-phi` breaking correlations, etc.). It's the primary write-up of the project.
