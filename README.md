# SPECT Collimator Transport Surrogate

A neural network surrogate for Monte Carlo photon transport through a SPECT collimator. The goal is to replace expensive Geant4/OpenGATE simulations with a fast generative model that reproduces the same statistical output.

## Background

Single Photon Emission Computed Tomography (SPECT) uses a lead collimator to restrict which gamma photons reach the detector. Simulating photon transport through the collimator with full Monte Carlo physics (OpenGATE/Geant4) is accurate but slow — it is the main computational bottleneck in realistic SPECT system modelling.

This project trains a surrogate that takes an incoming photon (position, direction, energy) and samples a realistic outgoing photon, or predicts that the photon is absorbed. The surrogate runs orders of magnitude faster than the Monte Carlo simulator.

## Collimator Geometry

```
material:          G4_Pb (lead)
outer dimensions:  550 × 405 × 23.8 mm
hole diameter:     1.2 mm
septum thickness:  0.2 mm
hole count:        ~128,000 parallel cylindrical holes
```

## Physics

Fluorescence is enabled in the physics list with low production cuts in lead. This allows lead characteristic X-rays (Pb Kα ≈ 75 keV, Pb Kβ ≈ 85 keV) to appear as a distinct photon class in the output.

## Data

Each photon is represented as 10 values:

```
in_x      mm     entry position X
in_y      mm     entry position Y
in_theta  rad    entry polar angle
in_phi    rad    entry azimuthal angle
in_E      keV    entry kinetic energy

out_x     mm     exit position X
out_y     mm     exit position Y
out_theta rad    exit polar angle
out_phi   rad    exit azimuthal angle
out_E     keV    exit kinetic energy  (0 if absorbed)
```

Photon classes:

```
blocked   out_E == 0                                    (99.87% of all photons)
direct    |out_E - in_E| / in_E < 5%                   (photon passes nearly unchanged)
xray      out_E in [70, 90] keV, in_E > 95 keV         (Pb characteristic X-ray emission)
scatter   everything else that passes                   (Compton scatter)
```

Transmission rate from 1B primary simulation: ~0.13%.

## Model Architecture

The surrogate is a two-stage pipeline:

**Stage 1 — Multiclass classifier**

A 4-class MLP that routes each incoming photon to its physical fate:

```
input:  5 (in_x, in_y, in_theta, in_phi, in_E)
hidden: 256 × 256 × 256 × 256
output: 4 class logits (blocked / direct / xray / scatter)
```

**Stage 2 — Per-class conditional GAN (WGAN-GP)**

One generator per physical class. Each generator samples a realistic outgoing photon conditioned on the incoming photon:

```
generator input:  5 (incoming photon) + z_dim (latent noise)
generator hidden: 256 × 256 × 256 × 256
generator output: 5 (outgoing photon)

critic input:  10 (incoming + outgoing concatenated)
critic hidden: 256 × 256 × 256 × 256
critic output: scalar Wasserstein score
```

Training: WGAN-GP, gradient penalty λ=10, 5 critic steps per generator step, Adam lr=1e-4.

**Inference (Markov chain sampling):**

```
incoming photon
      ↓
  classifier → sample class
      ↓
  blocked  → no output (photon absorbed)
  direct   → direct GAN   → outgoing photon
  xray     → xray GAN     → outgoing photon
  scatter  → scatter GAN  → outgoing photon
```

## Results

| Model | out_x (mm) | out_y (mm) | out_theta (rad) | out_phi (rad) | out_E (keV) |
|-------|-----------|-----------|----------------|--------------|------------|
| Regressor (binary, 100M) | 3.36 | 3.63 | 0.065 | 0.268 | 11.23 |
| Regressor (direct class, 100M) | 1.11 | 1.00 | 0.006 | 0.037 | 0.44 |
| GAN direct (1B data, z=32) | 5.49 | 2.99 | 0.020 | 0.050 | 1.35 |

MAE is a misleading metric for GANs — the generator samples from a distribution rather than predicting the mean. Histogram overlap with Monte Carlo ground truth is the correct evaluation.

The direct GAN (covering ~70% of all passed photons) reproduces the output distribution well. Xray and scatter GANs are in mode collapse due to data scarcity and require a larger simulation.

## Repository Structure

```
collimator_transport/       OpenGATE simulation package
  geometry.py               collimator and world geometry
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
validate_model.py           histogram and calibration plots

EXPERIMENT_LOG.md           full experiment history with results
```

## Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

Main dependencies: `opengate`, `uproot`, `numpy`, `torch`, `matplotlib`.

## Running A Simulation

```bash
uv run python -m collimator_transport.run --total 1000000000 --batches 1000 --workers 36 --output-dir output
```

- `--total`: total primary photons to simulate
- `--batches`: number of independent batches (each gets its own seed)
- `--workers`: parallel GATE instances (set to number of CPU cores)
- `--output-dir`: where ROOT files are written

Each batch is a fully independent simulation with a deterministic seed (`base_seed + batch_id`). Results are statistically mergeable.

For long runs, use tmux:

```bash
tmux new -s sim && cd gate10_playground && uv run python -m collimator_transport.run --total 1000000000 --batches 1000 --workers 36 --output-dir output
```

## Postprocessing

Matches incoming and outgoing photons within each batch, outputs a single `.npy` array:

```bash
# all photons
uv run python postprocess.py --output-dir output --out-file postprocessed.npy

# only xray photons (for targeted GAN training)
uv run python postprocess.py --output-dir output --out-file xray.npy --class-filter xray

# only scatter photons
uv run python postprocess.py --output-dir output --out-file scatter.npy --class-filter scatter
```

**Important:** matching must happen inside each batch before merging. EventID is only unique within one batch — matching across batches would incorrectly pair photons from different simulations.

## Training

```bash
# architecture search / baseline
uv run python train_prototype.py --data postprocessed.npy

# multiclass classifier + per-class regressors
uv run python train_multiclass.py --data postprocessed.npy

# conditional GAN for one class
uv run python train_gan.py --data postprocessed.npy --class direct
uv run python train_gan.py --data xray.npy --class xray
uv run python train_gan.py --data scatter.npy --class scatter
```

## Notes

- OpenGATE uses one CPU core per instance. Parallelism comes from running many instances simultaneously.
- On first run, Geant4 downloads physics data. Run a single-worker warm-up before launching parallel workers to avoid corrupted downloads.
- The transmission rate (~0.13%) means xray and scatter classes are extremely rare. A 1B primary simulation yields ~47K xray and ~17K scatter photons — insufficient for GAN training. A 10B simulation is in progress.
