# Experiment Log

This file tracks model/data changes and their results. The goal is to keep a clear record of what changed, why it changed, and whether it helped.

## Current Baseline: 100M Collimator Dataset

### Data

```text
simulation: OpenGATE/Geant4 collimator transport
total primaries: 100M
processed incoming rows: 85,116,126
passed photons: 117,743
transmission: 0.138%
```

### Important Postprocessing Fix

Changed photon matching from:

```text
merge all batches -> match incoming/outgoing
```

to:

```text
match incoming/outgoing inside each batch -> combine matched batches
```

Reason: `EventID` can repeat across batches, so matching after merging can pair photons from different batches incorrectly.

### Baseline Models

Classifier:

```text
task: incoming photon -> pass/not-pass
architecture: 5 -> 64 -> 64 -> 1
hidden layers: 2
neurons per hidden layer: 64
activation: ReLU
training data: balanced passed/not-passed sample
```

Regressor:

```text
task: incoming photon -> outgoing photon values
architecture: 5 -> 64 -> 64 -> 5
hidden layers: 2
neurons per hidden layer: 64
activation: ReLU
training data: passed photons only
```

### Results

Classifier test results:

```text
balanced accuracy: 91.9%
precision: 91.3%
recall: 92.7%
specificity: 91.1%
```

Regressor test errors:

```text
out_x MAE:      3.36 mm
out_y MAE:      3.63 mm
out_theta MAE:  0.065 rad
out_phi MAE:    0.268 rad
out_E MAE:      11.23 keV
```

### Validation Notes

Global 1D histograms match reasonably well, especially outgoing position.

Known weaknesses:

```text
classifier underpredicts pass probability at higher incoming energies
regressor smooths sharp energy features around ~75-90 keV
direction variables show spread in predicted-vs-Monte-Carlo scatter plots
```

Professor noted that the jumps around ~80 keV are most likely characteristic X-rays of lead.

Interpretation:

```text
The baseline is useful, but a deterministic regressor is not enough for Monte Carlo-like sampling.
```

## Architecture Search: Deeper and Wider Networks

### What Changed

Upgraded `train_prototype.py` to support configurable architectures via `--hidden-dims`, with optional `--batchnorm` and `--dropout`. Also added:

```text
ReduceLROnPlateau scheduler (halves LR when val loss stagnates)
early stopping with patience
gradient clipping (max_norm=5.0)
best-model restore at end of training
auto-named output directories (e.g. h4_w256, h4_w256_bn)
```

All runs used the same data, seed, and train/val/test split for fair comparison.

### Results

| Architecture | Params | Cls bal_acc | out_x MAE | out_y MAE | out_theta MAE | out_phi MAE | out_E MAE |
|---|---|---|---|---|---|---|---|
| 2×64 (baseline) | ~8K | 91.9% | 3.36 mm | 3.63 mm | 0.065 rad | 0.268 rad | 11.23 keV |
| 4×256 | 199K | **95.05%** | **2.08 mm** | **1.91 mm** | **0.049 rad** | **0.247 rad** | 10.52 keV |
| 4×512 | 791K | 94.97% | 2.57 mm | 2.55 mm | 0.053 rad | 0.270 rad | 11.14 keV |
| 4×256 + BN | 201K | 94.61% | 2.89 mm | 2.75 mm | 0.054 rad | 0.260 rad | 10.32 keV |
| 4×512 + BN + DO=0.1 | 796K | 94.08% | 2.73 mm | 2.58 mm | 0.050 rad | **0.242 rad** | 10.50 keV |

### Findings

```text
best overall:       4×256, no regularization (h4_w256)
best out_phi:       4×512 + BN + Dropout=0.1 (0.242 rad)
going wider:        hurts — 4×512 is worse than 4×256 across the board
adding BatchNorm:   hurts position and classifier, marginal effect on energy
adding Dropout:     helps out_phi slightly but hurts everything else
```

Architecture is near its ceiling for this loss function and dataset size.

### Why the Remaining Errors Are Hard to Fix

```text
out_phi:  two symmetric peaks at ±π/2. A point-estimate regressor predicts the
          mean between them, which is not a real physical value.

out_E:    sharp Pb characteristic X-ray spikes at ~75-88 keV. These are
          discrete atomic emissions, not a smooth function of incoming energy.
          MSE regression averages over them and smooths the spikes out.
```

Conclusion: further architecture tuning will not meaningfully reduce out_phi or out_E errors. The problem is the loss function and model type, not the depth or width.

## Multi-Class Classifier And Per-Class Regressors

### What Changed

Replaced the binary pass/fail classifier with a 4-class classifier. Classes are defined from the postprocessed data:

```text
0  blocked  out_E == 0
1  direct   out_E > 0, |out_E - in_E| / in_E < 5%
2  xray     out_E > 0, out_E in [70, 90] keV, in_E > 95 keV
3  scatter  out_E > 0, not direct, not xray
```

Trained one 4-class classifier and three separate regressors (one per pass class).

### Class Distribution In 100M Dataset

```text
blocked :  84,998,383  (99.862%)
direct  :     108,649  (0.128%)
xray    :       6,740  (0.008%)
scatter :       2,354  (0.003%)
```

xray and scatter are extremely rare.

### Results

Classifier:

```text
overall accuracy: 74.6%
blocked  accuracy: 75.3%
direct   accuracy: 77.3%
xray     accuracy: 21.7%   ← poor — too few examples
scatter  accuracy: 61.3%
```

Regressor MAE:

```text
             out_x    out_y  out_theta  out_phi   out_E
direct        1.11     1.00      0.006    0.037    0.44   ← excellent
xray          4.35     8.48      0.299    1.540    4.90   ← poor
scatter       7.57     8.90      0.274    1.582   13.78   ← poor
```

### Key Findings

**Direct regressor is a major breakthrough:**

Separating direct photons (in ≈ out) from xray and scatter photons makes the direct mapping nearly trivial. MAE dropped by roughly 2x on position and 7x on direction compared to the single binary regressor.

**Xray and scatter regressors fail due to data scarcity, not architecture:**

With only 6,740 xray and 2,354 scatter photons in 85M total, there is not enough signal to train a reliable regressor. The xray out_E histogram clearly shows two distinct Pb K X-ray lines (Kα ~75 keV, Kβ ~85 keV) which the NN merges into one smeared peak — a sign of averaging over a multi-modal distribution with too few samples.

**Conclusion:**

Architecture changes will not fix xray and scatter. The problem is data scarcity. A larger simulation is needed to produce enough rare-event photons.

### What The Plots Show

```text
direct:   near-perfect overlap on all outputs — ready for GAN stage
xray:     out_theta predicted at wrong angle, out_phi collapses to zero,
          out_E merges the two Pb K lines into one broad peak
scatter:  nothing matches — distributions are completely off
```



## GAN Training On 1B Simulation Data

**Date:** 2026-05-25
**Script:** `train_gan.py` — WGAN-GP conditional GAN, one model per physical class
**Data:** `/media/storage/nabdullah/postprocessed_1B.npy` (851,474,815 rows total)
**Architecture:** Generator and Critic both 4×256 MLP, z_dim=16 (32 for direct), GP lambda=10, n_critic=5

### 1B Dataset Statistics

```text
total incoming:      851,474,815
passed photons:        1,085,697  (0.13%)

class breakdown:
  direct  :    759,832  (70.0% of passed)
  xray    :     47,043  (4.3%  of passed)
  scatter :     16,924  (1.6%  of passed)
  blocked : ~850,389,118 (99.87% of all)
```

Compared to 100M simulation: xray went from 6,740 → 47,043 (~7x), scatter from 2,354 → 16,924 (~7x). Not quite the 10x needed.

### GAN Architecture And Training Setup

```python
class Generator(nn.Module):
    # input: 5 (in photon) + z_dim (noise) -> 5 (out photon)
    self.net = build_mlp(5 + z_dim, 5, hidden_dims)

class Critic(nn.Module):
    # input: 5 (in photon) + 5 (out photon) -> scalar score
    self.net = build_mlp(10, 1, hidden_dims, dropout=0.1)
```

Training: Adam lr=1e-4, 5 critic steps per generator step, 200–300 epochs, batch=512.
Best generator restored by lowest validation Wasserstein distance.

### Results: Direct GAN (z_dim=32, 300 epochs)

```text
best_val_w:  -0.098
final MAE:
  out_x:      5.49 mm
  out_y:      2.99 mm
  out_theta:  0.020 rad
  out_phi:    0.050 rad
  out_E:      1.35 keV
train/val rows: 759,832 / 162,821
```

**Histogram analysis:**

```text
out_x, out_y:   tight, near-perfect overlap with real data
out_theta:      excellent — distribution shape matches
out_phi:        peaks correct but magnitude slightly soft (generator under-samples the sharp peaks)
out_E:          excellent — ~140 keV peak reproduced cleanly
```

Overall the direct GAN captures the distribution well. The slight softness in `out_phi` is expected — WGAN-GP encourages broad coverage rather than sharp peaks. More latent dimensions (z_dim=32 vs 16) helped.

**Key insight — GAN vs Regressor:**
The regressor MAE on direct (out_phi ≈ 0.037 rad) looks better than GAN MAE (0.050 rad) but this comparison is misleading. The regressor predicts the conditional mean — it cannot produce the spread of the real distribution. The GAN samples from the distribution, which is what a Monte Carlo surrogate needs. MAE is the wrong metric for generative models.

### Results: Xray GAN (z_dim=16, 200 epochs)

```text
best_val_w:  -2.225
final MAE:
  out_x:     129.2 mm
  out_y:      90.8 mm
  out_theta:   0.32 rad
  out_phi:     1.57 rad
  out_E:       4.37 keV
train/val rows: 47,043 / 10,080
```

**Diagnosis: mode collapse.** The Wasserstein distance is ~22x worse than the direct GAN. All positional MAEs are at noise floor (129 mm, 90 mm). The generator is not learning the conditional mapping.

Root cause: 47,043 samples is still insufficient for a 5→5 conditional generative model with this architecture. The GAN needs to learn a complex conditional distribution over all 5 output dimensions simultaneously.

### Results: Scatter GAN (z_dim=16, 200 epochs)

```text
best_val_w:  -2.199
final MAE:
  out_x:     129.4 mm
  out_y:      90.3 mm
  out_theta:   0.28 rad
  out_phi:     1.59 rad
  out_E:      36.4 keV
train/val rows: 16,924 / 3,626
```

**Diagnosis: mode collapse.** Same pattern as xray — worst-case MAEs, near-random output. 16,924 samples is far too few for GAN training on 5 output dimensions. The energy MAE of 36 keV (vs ~8 keV typical Compton scatter spread) confirms the generator is not tracking the real distribution.

### Key Findings

**Direct GAN works — ready for integration:**

The direct class GAN (70% of all passed photons) achieves good distributional fidelity. It is the primary path to a working Monte Carlo surrogate and covers the dominant physical process through the collimator.

**Xray and scatter need at least 10x more data:**

Even with 1B primary photons, xray (47K) and scatter (17K) fall short. Rough minimum estimate for GAN training: ~500K per class. That implies ~10B primary photons.

**Alternative for xray: physics-constrained model**

The xray energy distribution is nearly deterministic (Pb Kα ≈ 75 keV, Kβ ≈ 85 keV fixed by atomic physics). A physics-informed conditional model — e.g. fix out_E to a mixture of the two lines and only learn the positional/angular output — could work with far fewer samples.

---

## GAN Training On 10B Data (Xray And Scatter)

**Date:** 2026-06-03
**Data:** xray_10B.npy (676,692 rows), scatter_10B.npy (244,025 rows)
**Architecture:** same as 1B run — Generator and Critic 4×256, z_dim=32, GP lambda=10, n_critic=5, 300 epochs

### Dataset Yield From 10B Simulation

```text
xray:    676,692  (up from 47,043 in 1B — ~14x)
scatter: 244,025  (up from 16,924 in 1B — ~14x)
```

### Results: Xray GAN

```text
best_val_w:  -0.2975
final MAE:
  out_x:      8.93 mm
  out_y:     12.22 mm
  out_theta:  0.31 rad
  out_phi:    1.59 rad
  out_E:      4.79 keV
```

### Results: Scatter GAN

```text
best_val_w:  -1.2189
final MAE:
  out_x:     38.63 mm
  out_y:     51.99 mm
  out_theta:  0.32 rad
  out_phi:    2.34 rad
  out_E:     17.82 keV
```

### Histogram Analysis

**Xray:**
```text
out_x, out_y:  near-perfect overlap — position learned well
out_theta:     correct general shape, sharp spike near 0 is softer than MC
out_phi:       GAN produces broad bumps at correct positions (-2, +2 rad)
               but cannot reproduce the razor-sharp MC spikes
out_E:         GAN learned the correct energy range (~70-90 keV)
               but outputs a broad smear instead of the 4 discrete Pb K lines
```

**Scatter:**
```text
out_x, out_y:  rough shape correct but distributions are shifted
out_theta:     GAN misses the sharp peak near 0, produces broad bell instead
out_phi:       complete failure — MC has two sharp spikes at ±2.5 rad,
               GAN outputs a flat blob
out_E:         broad high-energy hump (100-250 keV) captured reasonably,
               low-energy cluster near 75 keV missed entirely
```

### Key Findings

**Xray position is solved.** With 676K samples, the xray GAN learned position very well. This is a clear improvement over the 1B run where position was at noise floor.

**The phi sharpness problem is not a data problem.** Both xray and scatter have the same failure: real MC out_phi has razor-sharp discrete spikes, and the GAN produces broad bumps at the right locations. This is a fundamental limitation of continuous generators — they cannot reproduce delta-function-like distributions. More data will not fix this.

**Xray out_E is a quantization problem.** The 4 discrete Pb K emission lines are fixed atomic physics values, not a continuous distribution. A continuous GAN will always smear them. The correct approach is to sample out_E from a discrete mixture of the known lines, not learn it.

**Scatter is still partially failing.** Wasserstein improved 1.8x over 1B run but position MAEs (38-52mm) are still far from usable. The scatter distribution is genuinely more complex than xray — wide energy range, broad angular spread — and may need architectural changes in addition to more data.

**Scatter out_E spikes are misclassified xray photons.** The scatter out_E histogram has sharp spikes at exactly ~73 and ~75 keV — the Pb Kα line energies. These are Pb K X-rays, not Compton scattered photons. The cause is a wrong threshold in the class definition: `XRAY_MIN_IN_E = 95 keV`. The Pb K-edge (minimum incoming energy needed to produce a Pb K X-ray) is **88 keV**, not 95 keV. Photons entering with in_E between 88–95 keV can produce Pb K X-rays but get classified as scatter. Fix: lower `XRAY_MIN_IN_E` to 88 keV in both `postprocess.py` and `train_gan.py`.

---

## Physics-Correct Class Definition Using TrackID

**Date:** 2026-06-03

### Problem With Energy-Threshold Classification

The previous xray class definition used energy thresholds:

```python
xray = out_E in [70, 90] keV AND in_E > 95 keV
```

This is imprecise. The 95 keV threshold was arbitrary — the real physics threshold is the Pb K-edge at 88 keV. Any photon entering with in_E > 88 keV can produce a Pb K X-ray. This caused photons with in_E between 88–95 keV to be misclassified as scatter, producing the ~73–75 keV spikes visible in the scatter out_E histogram.

### Fix: TrackID-Based Classification

In Geant4, every particle has a TrackID:

```
TrackID = 1  →  primary particle (original photon from source)
TrackID > 1  →  secondary particle (new particle created during simulation)
```

A Pb K X-ray is always a secondary — it is a new photon born inside the lead atom. The original scattered photon is always primary (TrackID = 1). This gives exact classification with no thresholds:

```
xray    =  passed AND is_secondary (TrackID > 1)
direct  =  passed AND primary AND |out_E - in_E| / in_E < 5%
scatter =  passed AND primary AND not direct
```

### What Changed

Added column 10 (`is_secondary`) to the postprocessed `.npy` array:

```python
COL_IS_SECONDARY = 10   # 1 if outgoing TrackID > 1, else 0
N_COLS = 11
```

Updated `classify_chunk()` in `postprocess.py` and `get_class_indices()` in `train_gan.py` to use `is_secondary` instead of energy thresholds.

Re-running postprocess on 10B data to produce corrected files:

```
xray_10B_v2.npy    — xray photons with physics-correct TrackID definition
scatter_10B_v2.npy — scatter photons, now clean of misclassified Pb K X-rays
```

### Xray GAN With Fixed Energy (fix-energy mode)

While waiting for v2 data, training xray GAN on old xray_10B.npy with `--fix-energy` flag. In this mode:

```
generator learns:  out_x, out_y, out_theta, out_phi  (4 outputs)
energy at inference: sampled from empirical PMF built from training data
```

Generator: 5+z_dim → 4, Critic: 9 → 1. Energy PMF built from 473,684 training samples, 467 bins. Best val_w reached ~-0.22 around epoch 102 before oscillating — best checkpoint saved automatically.

---

## GAN Retraining On v2 Data (TrackID Classification)

**Date:** 2026-06-04
**Data:** xray_10B_v2.npy, scatter_10B_v2.npy
**Change:** TrackID-based classification replaces energy threshold classification

### Dataset Sizes

```text
xray_v2:    not yet recorded (similar to 676K + 88-95 keV additions)
scatter_v2: 219,704  (down from 244,025 — misclassified Pb K photons removed)
```

### Results: Xray v2 GAN (fix-energy, z_dim=32, 300 epochs)

```text
best_val_w:  -0.2421
final MAE:
  out_x:      9.59 mm
  out_y:      9.97 mm
  out_theta:  0.31 rad
  out_phi:    1.60 rad
  out_E:      4.87 keV
```

### Results: Scatter v2 GAN (z_dim=32, 300 epochs)

```text
best_val_w:  -0.6628
final MAE:
  out_x:     10.96 mm
  out_y:     10.26 mm
  out_theta:  0.28 rad
  out_phi:    1.65 rad
  out_E:     14.39 keV
```

### Histogram Analysis

**Xray v2:**
```text
out_x, out_y:  good overlap, position well learned
out_theta:     sharp near-zero spike slightly softer than MC but improved
out_phi:       broad bumps at correct locations, spikes not reproduced
out_E:         all 4 Pb K lines now visible including 85-87 keV Kβ lines
               — v2 data fixed the missing Kβ lines from old 95 keV threshold
```

**Scatter v2:**
```text
out_x, out_y:  near-perfect overlap — massive improvement over v1
out_theta:     shape mostly right, GAN slightly broader
out_phi:       broad bumps instead of sharp spikes — same problem as xray
out_E:         clean Compton distribution, no more Pb K spikes, shape matches well
```

### Key Findings

**TrackID classification fixed scatter dramatically.** Position MAE dropped from 38-52mm to ~10mm. The v1 scatter class was polluted with misclassified Pb K X-rays which had completely different spatial distributions — once removed, the GAN learned scatter position cleanly.

**Xray Kβ lines now reproduced.** The v2 xray PMF includes the 88-95 keV photons that produce Kβ lines (~85-87 keV). All 4 Pb K lines now appear correctly in the energy histogram.

**Phi is the only remaining open problem.** Both xray and scatter out_phi distributions have razor-sharp spikes at ±2 rad (collimator geometry constraint) that a continuous GAN cannot reproduce. Same fix as energy — sample phi from empirical PMF, only ask GAN to learn x, y, theta.

---

## Fix-Phi Attempt: Why Decoupling Failed

**Date:** 2026-06-06

Tried generalizing the `--fix-energy` approach to phi: GAN learns only spatial outputs (x, y, theta), phi sampled from empirical PMF at inference.

### Results

**Xray (fix-energy + fix-phi):**
```text
                  fix-E only (v2)   fix-E + fix-phi
val_w:            -0.24            -0.96
out_x:            9.59 mm          75.55 mm     ← 8x worse
out_y:            9.97 mm          42.32 mm     ← 4x worse
out_theta:        0.31 rad         0.39 rad
```

**Scatter (fix-phi only):**
```text
                  v2 (no fix)      fix-phi
val_w:            -0.66            -0.78
out_x:            10.96 mm         14.92 mm     ← worse
out_E:            14.39 keV        33.20 keV    ← 2x worse
```

### Why This Failed

The lesson is about **independence vs correlation**:

- **Energy is independent.** Pb K X-ray lines are atomic emissions at fixed energies (73, 75, 85, 87 keV) determined by lead's electron shell structure. They do not depend on photon position or trajectory. Removing energy from the GAN's outputs lost no information.

- **Phi is correlated.** The cylindrical hole geometry that produces the discrete phi peaks also constrains position, theta, and (via Compton scattering for the scatter class) energy. Stripping phi from the GAN's joint output destroyed the correlation structure it needed to learn the remaining variables.

For scatter, the Compton scattering energy-angle relationship is especially tight. Removing phi broke the GAN's ability to learn the energy distribution itself — energy MAE doubled even though energy was still being learned.

### General Principle

```text
fix-mode works ONLY for outputs that are statistically independent
of the variables the GAN is learning.

fix-energy works for xray because Pb K lines are atomic, independent
of trajectory.

fix-phi fails because phi is geometrically coupled to position and
angle, and via scattering kinematics to energy.
```

### Best Models So Far

```text
xray:    --fix-energy on v2 data         (val_w -0.24, position ~10 mm)
scatter: no fix flags on v2 data         (val_w -0.66, position ~10 mm)
```

---

## Warp-Phi: Rank-Preserving Post-Processing

**Date:** 2026-06-08

After fix-phi failed (decoupling broke correlations), tried a different approach: keep phi in the GAN's joint output so correlations are preserved, but post-process the phi marginal at inference using empirical CDF warping (quantile transformation).

### Method

```
1. GAN trains on all 5 outputs normally — joint correlations intact.
2. At inference, rank-sort the GAN's batch of phi values.
3. Rank-sort an equal-size set of phi values from training data.
4. Replace each generated phi with the training value at the same rank.
```

This guarantees the warped phi marginal exactly matches the empirical distribution (sharp spikes appear automatically) while preserving rank correlations with x, y, theta, E.

### Implementation

Added `build_cdf()`, `warp_to_cdf()`, `--warp-phi` flag, and `warp_cdfs` argument to `evaluate_generator()` in `train_gan.py`. Independent of `--fix-phi` — they cannot both be applied to the same column.

### Results: Scatter (z_dim=64, --warp-phi)

```text
                  v2 baseline      warp-phi (z=64)
val_w:            -0.66            -0.36       ← raw GAN val_w looks worse
out_x:            10.96 mm         10.33 mm    ← better
out_y:            10.26 mm         8.40 mm     ← better
out_theta:        0.28 rad         0.26 rad    ← better
out_phi:          1.65 rad         1.60 rad    ← marginal MAE, but see histogram
out_E:            14.39 keV        11.61 keV   ← better
```

### Histogram Analysis

```text
out_x, out_y:  clean overlap, no degradation from warping
out_theta:     near-perfect overlap, sharp spike near 0 captured
out_phi:       sharp spikes at -1.5 and +1.5 rad now reproduced exactly,
               background between spikes also matches MC well
out_E:         even cleaner than v2, broad Compton shape essentially identical
```

The phi MAE number is misleading — it stays ~random because the GAN's original phi was noise, and warping noise into a sharp distribution doesn't reduce per-sample MAE. What matters is the **distribution shape**, which now matches MC almost perfectly.

### Why This Works (And Caveats)

```text
The GAN learns the joint distribution (x, y, theta, phi, E) including
correlations. Phi rank-warping changes only the absolute phi values,
not their order. So any monotonic correlation with the other outputs
is preserved.

Caveat 1: This is statistical post-processing, not a physics-derived
fix. The marginal is correct by construction; the joint is only
"best-effort" correct.

Caveat 2: If the GAN's raw phi output is noise (no real correlation),
warping just gives noise reshaped to the right marginal. Caveats can
only be ruled out by validating downstream image reconstruction.
```

---

## Xray With fix-energy + warp-phi

**Date:** 2026-06-09
**Data:** xray_10B_v2.npy
**Config:** `--fix-energy --warp-phi`, z_dim=64, 300 epochs

### Results

```text
                 fix-E only (v2)    fix-E + warp-phi (z=64)
val_w:           -0.24              -0.22
out_x:           9.59 mm            5.43 mm   ← better
out_y:           9.97 mm            11.42 mm  ← slightly worse
out_theta:       0.31 rad           0.31 rad
out_phi:         1.60 rad           1.59 rad
out_E:           4.87 keV           4.88 keV
```

### Histogram Analysis

```text
out_x, out_y:  clean overlap
out_theta:     GAN starts at 0 with a soft slope, MC has a sharp
               near-vertical peak right at theta=0. Same fundamental
               GAN limitation that affected phi — continuous generator
               cannot produce a delta-like peak.
out_phi:       sharp spikes at -1.5 and +1.5 reproduced exactly
               (warping working as intended)
out_E:         all 4 Pb K lines perfect (PMF working as intended)
```

phi and energy are solved. The remaining issue is the sharp theta peak at 0.

---

## Warp-Theta Extension

**Date:** 2026-06-09

The same continuous-GAN limitation that produces soft phi peaks also produces a soft theta peak at 0. Applying the same CDF warping technique to theta should sharpen it.

### What Changed

Added `--warp-theta` flag to `train_gan.py`. Same mechanism as `--warp-phi`:

```text
1. Build empirical CDF of theta from training data.
2. Save it with the model.
3. At inference, rank-warp the GAN's theta output to match.
```

Implementation: extended the existing `warp_cdfs` dict in `train()` and `evaluate_generator()` to handle theta in addition to phi. No core algorithm changes.

### Why Theta Warping Is Reasonable

```text
- Theta peak at 0 = "ballistic" photons going straight through the
  collimator without scattering. The dominant signal in any SPECT image.

- The same rank-preservation argument applies: where GAN already matches
  MC, the warp is approximately identity. Only the disagreement near
  theta=0 gets significantly remapped.

- Cost: small distortion in middle-theta regions where the GAN was
  already accurate.

- Benefit: sharp peak at 0 appears correctly.
```



