"""
Conditional WGAN-GP for collimator photon transport.

Trains a conditional generator that takes an incoming photon + random noise
and produces a plausible outgoing photon, matching the Monte Carlo distribution.

One GAN per pass class:
  direct  — run now with 100M dataset
  xray    — run after 1B dataset is ready
  scatter — run after 1B dataset is ready

Usage:
    uv run python train_gan.py --data /media/storage/nabdullah/postprocessed_100M.npy --class direct
    uv run python train_gan.py --data /media/storage/nabdullah/postprocessed_1B.npy --class xray
    uv run python train_gan.py --data /media/storage/nabdullah/postprocessed_1B.npy --class scatter
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


IN_COLS          = [0, 1, 2, 3, 4]
OUT_COLS         = [5, 6, 7, 8, 9]
OUT_COLS_SPATIAL = [5, 6, 7, 8]        # x, y, theta, phi — no energy
OUT_E_COL        = 9
IS_SECONDARY_COL = 10                  # 1 if outgoing photon is secondary (Pb K X-ray)
IN_E             = 4
OUT_E            = 9
OUT_NAMES_ALL    = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]

DIRECT_REL_TOL = 0.05


def get_class_indices(data, class_name):
    out_e        = data[:, OUT_E]
    in_e         = data[:, IN_E]
    is_secondary = data[:, IS_SECONDARY_COL] > 0

    passed  = out_e > 0
    xray    = passed & is_secondary
    direct  = passed & ~is_secondary & (np.abs(out_e - in_e) / np.maximum(in_e, 1e-6) < DIRECT_REL_TOL)
    scatter = passed & ~is_secondary & ~direct

    mapping = {"direct": direct, "xray": xray, "scatter": scatter}
    if class_name not in mapping:
        raise ValueError(f"Unknown class '{class_name}'. Choose from: direct, xray, scatter")
    return np.flatnonzero(mapping[class_name])


def standardize_from_train(x_train, *others):
    mean = x_train.mean(axis=0)
    std  = x_train.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    result = [(x_train - mean) / std]
    result.extend((x - mean) / std for x in others)
    return mean, std, result


def split_indices(indices, rng, train_frac=0.70, val_frac=0.15):
    indices = np.array(indices, copy=True)
    rng.shuffle(indices)
    n       = len(indices)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def build_mlp(in_dim, out_dim, hidden_dims, dropout=0.0):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.2))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_dim=16, hidden_dims=(256, 256, 256, 256), out_dim=5):
        super().__init__()
        self.net = build_mlp(5 + z_dim, out_dim, hidden_dims)

    def forward(self, x_in, z):
        return self.net(torch.cat([x_in, z], dim=1))


class Critic(nn.Module):
    def __init__(self, hidden_dims=(256, 256, 256, 256), dropout=0.1, in_dim=10):
        super().__init__()
        self.net = build_mlp(in_dim, 1, hidden_dims, dropout=dropout)

    def forward(self, x_in, x_out):
        return self.net(torch.cat([x_in, x_out], dim=1)).squeeze(1)


def build_energy_pmf(energies, n_bins=500):
    """Measure empirical energy distribution as a discrete PMF."""
    counts, edges = np.histogram(energies, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask    = counts > 0
    centers = centers[mask].astype(np.float32)
    probs   = (counts[mask] / counts[mask].sum()).astype(np.float32)
    return centers, probs


def sample_energy(pmf_centers, pmf_probs, n):
    """Sample n energy values from the empirical PMF."""
    idx = np.random.choice(len(pmf_centers), size=n, p=pmf_probs)
    return pmf_centers[idx]


def gradient_penalty(critic, x_in, real_out, fake_out, device):
    """
    WGAN-GP gradient penalty.
    Interpolates between real and fake outgoing photons and penalises
    the critic if its gradient norm deviates from 1.
    """
    batch_size = real_out.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = (alpha * real_out + (1 - alpha) * fake_out).requires_grad_(True)

    score = critic(x_in, interpolated)
    grad  = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=shuffle, drop_last=True,
    )


def make_regression_plots(y_true, y_pred, out_file, title):
    names = OUT_NAMES_ALL
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.hist(y_true[:, i], bins=50, alpha=0.55, label="Monte Carlo", density=True)
        ax.hist(y_pred[:, i], bins=50, alpha=0.55, label="GAN", density=True)
        ax.set_title(name)
    axes[-1].axis("off")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def evaluate_generator(generator, x_test, y_test_full, y_mean, y_std, z_dim, device,
                       n_samples=5, fix_energy=False, pmf_centers=None, pmf_probs=None):
    """
    Generate n_samples outgoing photons per test input.
    If fix_energy: spatial output from GAN, energy sampled from empirical PMF.
    y_test_full always has 5 columns for MAE comparison.
    """
    generator.eval()
    all_preds = []
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    n   = len(x_t)

    with torch.no_grad():
        for _ in range(n_samples):
            z      = torch.randn(n, z_dim, device=device)
            pred_s = generator(x_t, z).cpu().numpy()
            pred   = pred_s * y_std + y_mean  # denorm spatial (4 or 5 cols)

            if fix_energy:
                e = sample_energy(pmf_centers, pmf_probs, n).reshape(-1, 1)
                pred = np.concatenate([pred, e], axis=1)  # now 5 cols

            all_preds.append(pred)

    mean_pred = np.mean(all_preds, axis=0)
    mae = np.mean(np.abs(mean_pred - y_test_full), axis=0)

    generator.train()
    return mae, all_preds[0]


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng    = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.data}")
    data = np.load(args.data, mmap_mode="r")

    print(f"Selecting class: {args.cls}")
    idx = get_class_indices(data, args.cls)
    print(f"  {args.cls} photons: {len(idx):,}")

    train_idx, val_idx, test_idx = split_indices(idx, rng)

    x_train = np.asarray(data[train_idx][:, IN_COLS],  dtype=np.float32)
    x_val   = np.asarray(data[val_idx][:, IN_COLS],    dtype=np.float32)
    x_test  = np.asarray(data[test_idx][:, IN_COLS],   dtype=np.float32)
    y_train = np.asarray(data[train_idx][:, OUT_COLS], dtype=np.float32)
    y_val   = np.asarray(data[val_idx][:, OUT_COLS],   dtype=np.float32)
    y_test  = np.asarray(data[test_idx][:, OUT_COLS],  dtype=np.float32)

    fix_energy  = args.fix_energy
    pmf_centers = pmf_probs = None

    if fix_energy:
        # keep full y_test for MAE comparison, but train only on spatial outputs
        e_train     = np.asarray(data[train_idx][:, OUT_E_COL], dtype=np.float32)
        pmf_centers, pmf_probs = build_energy_pmf(e_train)
        print(f"  Energy PMF built from {len(e_train):,} samples, {len(pmf_centers)} bins")
        y_train = np.asarray(data[train_idx][:, OUT_COLS_SPATIAL], dtype=np.float32)
        y_val   = np.asarray(data[val_idx][:, OUT_COLS_SPATIAL],   dtype=np.float32)
        # y_test keeps all 5 cols for evaluation
        y_test  = np.asarray(data[test_idx][:, OUT_COLS], dtype=np.float32)
        out_dim  = 4
        crit_dim = 9   # 5 incoming + 4 spatial
    else:
        out_dim  = 5
        crit_dim = 10

    x_mean, x_std, (x_train_s, x_val_s, x_test_s) = standardize_from_train(x_train, x_val, x_test)
    if fix_energy:
        y_mean, y_std, (y_train_s, y_val_s) = standardize_from_train(y_train, y_val)
    else:
        y_mean, y_std, (y_train_s, y_val_s, _) = standardize_from_train(y_train, y_val, y_test)

    train_loader = make_loader(x_train_s, y_train_s, args.batch_size, shuffle=True)
    val_loader   = make_loader(x_val_s,   y_val_s,   args.batch_size, shuffle=False)

    hidden_dims = tuple(args.hidden_dims)
    G = Generator(z_dim=args.z_dim, hidden_dims=hidden_dims, out_dim=out_dim).to(device)
    C = Critic(hidden_dims=hidden_dims, dropout=0.1, in_dim=crit_dim).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.0, 0.9))

    print(f"Device:      {device}")
    print(f"Generator:   {sum(p.numel() for p in G.parameters()):,} params")
    print(f"Critic:      {sum(p.numel() for p in C.parameters()):,} params")
    print(f"z_dim:       {args.z_dim}")
    print(f"Train rows:  {len(train_idx):,}")
    print(f"Output dir:  {out_dir}")

    history = []
    best_val_w  = float("inf")
    best_G_state = None

    for epoch in range(1, args.epochs + 1):
        G.train(); C.train()
        epoch_c_loss = 0.0
        epoch_g_loss = 0.0
        n_batches    = 0

        for x_real, y_real in train_loader:
            x_real = x_real.to(device)
            y_real = y_real.to(device)
            batch  = x_real.size(0)

            # --- train critic n_critic steps per generator step ---
            for _ in range(args.n_critic):
                z      = torch.randn(batch, args.z_dim, device=device)
                y_fake = G(x_real, z).detach()

                c_real = C(x_real, y_real)
                c_fake = C(x_real, y_fake)
                gp     = gradient_penalty(C, x_real, y_real, y_fake, device)
                c_loss = c_fake.mean() - c_real.mean() + args.gp_lambda * gp

                opt_C.zero_grad()
                c_loss.backward()
                opt_C.step()

            # --- train generator ---
            z      = torch.randn(batch, args.z_dim, device=device)
            y_fake = G(x_real, z)
            g_loss = -C(x_real, y_fake).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_c_loss += c_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches    += 1

        epoch_c_loss /= n_batches
        epoch_g_loss /= n_batches

        # wasserstein estimate on val set (lower = generator better)
        C.eval(); G.eval()
        val_w = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                z        = torch.randn(x_v.size(0), args.z_dim, device=device)
                y_fake_v = G(x_v, z)
                val_w   += (C(x_v, y_fake_v).mean() - C(x_v, y_v).mean()).item()
        val_w /= len(val_loader)
        C.train(); G.train()

        history.append({"epoch": epoch, "c_loss": epoch_c_loss, "g_loss": epoch_g_loss, "val_w": val_w})
        print(f"epoch {epoch:03d}: c_loss={epoch_c_loss:.4f}, g_loss={epoch_g_loss:.4f}, val_w={val_w:.4f}")

        if val_w < best_val_w:
            best_val_w   = val_w
            best_G_state = {k: v.cpu().clone() for k, v in G.state_dict().items()}

        if epoch % args.plot_every == 0:
            G.load_state_dict(best_G_state)
            mae, y_sample = evaluate_generator(G, x_test_s, y_test, y_mean, y_std, args.z_dim, device,
                                               fix_energy=fix_energy, pmf_centers=pmf_centers, pmf_probs=pmf_probs)
            make_regression_plots(y_test, y_sample, out_dir / f"gan_histograms_epoch{epoch:03d}.png", f"GAN {args.cls} — epoch {epoch}")
            mae_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(OUT_NAMES_ALL, mae))
            print(f"  [eval] MAE: {mae_str}")

    # final evaluation with best generator
    G.load_state_dict(best_G_state)
    mae, y_sample = evaluate_generator(G, x_test_s, y_test, y_mean, y_std, args.z_dim, device, n_samples=10,
                                       fix_energy=fix_energy, pmf_centers=pmf_centers, pmf_probs=pmf_probs)
    make_regression_plots(y_test, y_sample, out_dir / "gan_histograms_final.png", f"GAN {args.cls} — final")

    checkpoint = {
        "generator_state": best_G_state,
        "z_dim": args.z_dim,
        "hidden_dims": list(hidden_dims),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "input_columns": IN_COLS,
        "output_columns": OUT_COLS_SPATIAL if fix_energy else OUT_COLS,
        "class": args.cls,
        "fix_energy": fix_energy,
        "history": history,
    }
    if fix_energy:
        checkpoint["pmf_centers"] = pmf_centers
        checkpoint["pmf_probs"]   = pmf_probs
    torch.save(checkpoint, out_dir / "generator.pt")

    report = {
        "class": args.cls,
        "fix_energy": fix_energy,
        "data_file": args.data,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "z_dim": args.z_dim,
        "hidden_dims": list(hidden_dims),
        "best_val_w": float(best_val_w),
        "final_mae": {n: float(v) for n, v in zip(OUT_NAMES_ALL, mae)},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"Class: {args.cls}")
    print(f"Fix energy: {fix_energy}")
    print(f"Best val Wasserstein: {best_val_w:.4f}")
    print("Final MAE (avg over 10 samples):")
    for n, v in zip(OUT_NAMES_ALL, mae):
        print(f"  {n}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="postprocessed_data.npy")
    parser.add_argument("--class",      dest="cls", default="direct", choices=["direct", "xray", "scatter"])
    parser.add_argument("--out-dir",    default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--z-dim",      type=int,   default=16)
    parser.add_argument("--n-critic",   type=int,   default=5,
                        help="Critic steps per generator step (WGAN standard is 5)")
    parser.add_argument("--gp-lambda",  type=float, default=10.0,
                        help="Gradient penalty weight (WGAN-GP standard is 10)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--plot-every", type=int,   default=20,
                        help="Save histogram plots every N epochs")
    parser.add_argument("--fix-energy", action="store_true",
                        help="For xray: learn only spatial outputs, sample energy from empirical PMF")
    args = parser.parse_args()

    if args.out_dir is None:
        suffix = "_fixE" if args.fix_energy else ""
        args.out_dir = f"ml_artifacts/gan_{args.cls}{suffix}"

    train(args)


if __name__ == "__main__":
    main()
