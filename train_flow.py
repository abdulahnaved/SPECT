"""
Conditional Flow Matching for collimator photon transport.

  - sample t ~ U(0, 1), x_0 ~ N(0, I), x_1 = real outgoing photon
  - interpolate x_t = (1 - t) * x_0 + t * x_1   (linear path)
  - target velocity is v_t = x_1 - x_0          (constant along the path)
  - velocity net predicts v_hat from (incoming, x_t, t)
  - loss = MSE(v_hat, v_t)
  - inference = solve dx/dt = v_hat(x, t, incoming) from t=0 to t=1
                with a small ODE solver (Euler / RK4) and x(0) ~ N(0, I)

Usage:
    uv run python train_flow.py --data postprocessed.npy --class direct
    uv run python train_flow.py --data xray.npy --class xray --fix-energy --warp-phi
    uv run python train_flow.py --data scatter.npy --class scatter --warp-phi
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
OUT_E_COL        = 9
OUT_PHI_COL      = 8
OUT_THETA_COL    = 7
IS_SECONDARY_COL = 10
IN_E             = 4
OUT_E            = 9
OUT_NAMES_ALL    = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]
OUT_X_IDX, OUT_Y_IDX, OUT_THETA_IDX, OUT_PHI_IDX, OUT_E_IDX = 0, 1, 2, 3, 4

DIRECT_REL_TOL = 0.05


def get_class_indices(data, class_name):
    out_e = data[:, OUT_E]
    in_e  = data[:, IN_E]
    passed = out_e > 0

    if data.shape[1] > IS_SECONDARY_COL:
        # v2 data: physics-correct TrackID classification
        is_secondary = data[:, IS_SECONDARY_COL] > 0
        xray    = passed & is_secondary
        direct  = passed & ~is_secondary & (np.abs(out_e - in_e) / np.maximum(in_e, 1e-6) < DIRECT_REL_TOL)
        scatter = passed & ~is_secondary & ~direct
    else:
        # legacy data (10 cols): fall back to energy-threshold classification
        xray    = passed & (out_e >= 70.0) & (out_e <= 90.0) & (in_e > 88.0)
        direct  = passed & ~xray & (np.abs(out_e - in_e) / np.maximum(in_e, 1e-6) < DIRECT_REL_TOL)
        scatter = passed & ~xray & ~direct

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
        layers.append(nn.SiLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar t in [0, 1] — gives the velocity net
    a richer notion of time than a single scalar."""
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t: shape (B,) in [0, 1]
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class VelocityNet(nn.Module):
    """
    Predicts the velocity field v(x_t, t | incoming) used by the ODE solver.

    Input:  incoming photon (5) + current x_t (out_dim) + time embedding (t_dim)
    Output: velocity in output space (out_dim)
    """
    def __init__(self, out_dim=5, hidden_dims=(256, 256, 256, 256), t_dim=64):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.net = build_mlp(5 + out_dim + t_dim, out_dim, hidden_dims)

    def forward(self, x_in, x_t, t):
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x_in, x_t, t_emb], dim=1))


def build_pmf(values, n_bins=500):
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask    = counts > 0
    centers = centers[mask].astype(np.float32)
    probs   = (counts[mask] / counts[mask].sum()).astype(np.float32)
    return centers, probs


def sample_pmf(pmf_centers, pmf_probs, n):
    idx = np.random.choice(len(pmf_centers), size=n, p=pmf_probs)
    return pmf_centers[idx]


def build_cdf(values, n_bins=2000):
    sorted_vals = np.sort(values.astype(np.float64))
    n = len(sorted_vals)
    q = (np.arange(n) + 0.5) / n
    if n > n_bins:
        step = n // n_bins
        sorted_vals = sorted_vals[::step]
        q = q[::step]
    return sorted_vals.astype(np.float32), q.astype(np.float32)


def warp_to_cdf(values, target_sorted, target_q):
    """Map values through target CDF, preserving rank ordering."""
    ranks    = np.argsort(np.argsort(values))
    quantile = (ranks + 0.5) / len(values)
    return np.interp(quantile, target_q, target_sorted).astype(values.dtype)


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=shuffle, drop_last=True,
    )


def make_regression_plots(y_true, y_pred, out_file, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, name in enumerate(OUT_NAMES_ALL):
        ax = axes[i]
        ax.hist(y_true[:, i], bins=50, alpha=0.55, label="Monte Carlo", density=True)
        ax.hist(y_pred[:, i], bins=50, alpha=0.55, label="Flow", density=True)
        ax.set_title(name)
    axes[-1].axis("off")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


@torch.no_grad()
def sample_flow(net, x_in, out_dim, n_steps, device):
    """
    Solve dx/dt = net(x_in, x, t) from t=0 to t=1 with RK4.
    Initial state x(0) ~ N(0, I) in standardized space.
    Returns x(1) — generated outgoing photon in standardized space.
    """
    n = x_in.shape[0]
    x = torch.randn(n, out_dim, device=device)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t0 = torch.full((n,), step * dt,         device=device)
        t1 = torch.full((n,), step * dt + dt/2,  device=device)
        t2 = torch.full((n,), step * dt + dt/2,  device=device)
        t3 = torch.full((n,), step * dt + dt,    device=device)

        k1 = net(x_in, x,                      t0)
        k2 = net(x_in, x + dt * k1 / 2,        t1)
        k3 = net(x_in, x + dt * k2 / 2,        t2)
        k4 = net(x_in, x + dt * k3,            t3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return x


def evaluate_flow(net, x_test, y_test_full, y_mean, y_std, out_dim, device,
                  n_samples=5, n_steps=50, fixed_pmfs=None, learn_indices=None,
                  warp_cdfs=None):
    """
    For each test input, draw n_samples outgoing photons from the flow.
    Fixed outputs come from PMF sampling; learned outputs come from the flow.
    Optionally warp specified columns through an empirical CDF.
    y_test_full always has 5 columns for MAE comparison.
    """
    if fixed_pmfs is None:
        fixed_pmfs = {}
    if learn_indices is None:
        learn_indices = list(range(5))
    if warp_cdfs is None:
        warp_cdfs = {}

    net.eval()
    all_preds = []
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    n   = len(x_t)

    for _ in range(n_samples):
        pred_s = sample_flow(net, x_t, out_dim, n_steps, device).cpu().numpy()
        pred   = pred_s * y_std + y_mean

        if fixed_pmfs:
            full = np.zeros((n, 5), dtype=np.float32)
            for j, i in enumerate(learn_indices):
                full[:, i] = pred[:, j]
            for i, (centers, probs) in fixed_pmfs.items():
                full[:, i] = sample_pmf(centers, probs, n)
            pred = full

        for i, (target_sorted, target_q) in warp_cdfs.items():
            pred[:, i] = warp_to_cdf(pred[:, i], target_sorted, target_q)

        all_preds.append(pred)

    mean_pred = np.mean(all_preds, axis=0)
    mae = np.mean(np.abs(mean_pred - y_test_full), axis=0)

    net.train()
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

    # --- build fixed PMFs for any outputs not learned by the flow ---
    fixed_pmfs = {}

    if args.fix_energy:
        e_train = np.asarray(data[train_idx][:, OUT_E_COL], dtype=np.float32)
        fixed_pmfs[OUT_E_IDX] = build_pmf(e_train)
        print(f"  Energy PMF: {len(e_train):,} samples, {len(fixed_pmfs[OUT_E_IDX][0])} bins")

    if args.fix_phi:
        phi_train = np.asarray(data[train_idx][:, OUT_PHI_COL], dtype=np.float32)
        fixed_pmfs[OUT_PHI_IDX] = build_pmf(phi_train)
        print(f"  Phi PMF:    {len(phi_train):,} samples, {len(fixed_pmfs[OUT_PHI_IDX][0])} bins")

    # --- build CDFs for any outputs we'll warp at inference time ---
    warp_cdfs = {}
    if args.warp_phi and not args.fix_phi:
        phi_train_warp = np.asarray(data[train_idx][:, OUT_PHI_COL], dtype=np.float32)
        warp_cdfs[OUT_PHI_IDX] = build_cdf(phi_train_warp)
        print(f"  Phi CDF:    {len(phi_train_warp):,} samples for marginal warping")
    if args.warp_theta:
        theta_train_warp = np.asarray(data[train_idx][:, OUT_THETA_COL], dtype=np.float32)
        warp_cdfs[OUT_THETA_IDX] = build_cdf(theta_train_warp)
        print(f"  Theta CDF:  {len(theta_train_warp):,} samples for marginal warping")

    learn_indices   = [i for i in range(5) if i not in fixed_pmfs]
    learn_data_cols = [OUT_COLS[i] for i in learn_indices]
    out_dim         = len(learn_indices)

    if fixed_pmfs:
        y_train = np.asarray(data[train_idx][:, learn_data_cols], dtype=np.float32)
        y_val   = np.asarray(data[val_idx][:, learn_data_cols],   dtype=np.float32)
        y_test  = np.asarray(data[test_idx][:, OUT_COLS],         dtype=np.float32)  # full 5 cols for MAE

    x_mean, x_std, (x_train_s, x_val_s, x_test_s) = standardize_from_train(x_train, x_val, x_test)
    if fixed_pmfs:
        y_mean, y_std, (y_train_s, y_val_s) = standardize_from_train(y_train, y_val)
    else:
        y_mean, y_std, (y_train_s, y_val_s, _) = standardize_from_train(y_train, y_val, y_test)

    train_loader = make_loader(x_train_s, y_train_s, args.batch_size, shuffle=True)
    val_loader   = make_loader(x_val_s,   y_val_s,   args.batch_size, shuffle=False)

    hidden_dims = tuple(args.hidden_dims)
    net = VelocityNet(out_dim=out_dim, hidden_dims=hidden_dims, t_dim=args.t_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    print(f"Device:        {device}")
    print(f"VelocityNet:   {sum(p.numel() for p in net.parameters()):,} params")
    print(f"Output dim:    {out_dim}")
    print(f"Train rows:    {len(train_idx):,}")
    print(f"Output dir:    {out_dir}")

    history = []
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x_in, x1 in train_loader:
            x_in = x_in.to(device)
            x1   = x1.to(device)
            batch = x_in.size(0)

            # sample noise and time uniformly
            x0 = torch.randn_like(x1)
            t  = torch.rand(batch, device=device)

            # interpolate along the linear path
            x_t = (1.0 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            target_v = x1 - x0

            # predict velocity and regress
            pred_v = net(x_in, x_t, t)
            loss = ((pred_v - target_v) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= n_batches

        # val: same MSE on the velocity regression task
        net.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_in, x1 in val_loader:
                x_in = x_in.to(device)
                x1   = x1.to(device)
                x0   = torch.randn_like(x1)
                t    = torch.rand(x1.size(0), device=device)
                x_t  = (1.0 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
                target_v = x1 - x0
                pred_v   = net(x_in, x_t, t)
                val_loss += ((pred_v - target_v) ** 2).mean().item()
                n_val_batches += 1
        val_loss /= n_val_batches

        sched.step()

        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:03d}: train_loss={epoch_loss:.5f}, val_loss={val_loss:.5f}, lr={opt.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch % args.plot_every == 0:
            net.load_state_dict(best_state)
            mae, y_sample = evaluate_flow(net, x_test_s, y_test, y_mean, y_std, out_dim, device,
                                          n_samples=2, n_steps=args.n_steps,
                                          fixed_pmfs=fixed_pmfs, learn_indices=learn_indices,
                                          warp_cdfs=warp_cdfs)
            make_regression_plots(y_test, y_sample, out_dir / f"flow_histograms_epoch{epoch:03d}.png",
                                  f"Flow {args.cls} — epoch {epoch}")
            mae_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(OUT_NAMES_ALL, mae))
            print(f"  [eval] MAE: {mae_str}")

    net.load_state_dict(best_state)
    mae, y_sample = evaluate_flow(net, x_test_s, y_test, y_mean, y_std, out_dim, device,
                                  n_samples=10, n_steps=args.n_steps,
                                  fixed_pmfs=fixed_pmfs, learn_indices=learn_indices,
                                  warp_cdfs=warp_cdfs)
    make_regression_plots(y_test, y_sample, out_dir / "flow_histograms_final.png",
                          f"Flow {args.cls} — final")

    checkpoint = {
        "net_state": best_state,
        "hidden_dims": list(hidden_dims),
        "t_dim": args.t_dim,
        "out_dim": out_dim,
        "n_steps": args.n_steps,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "input_columns": IN_COLS,
        "output_columns": learn_data_cols,
        "learn_indices": learn_indices,
        "fixed_pmfs": {i: (c.tolist(), p.tolist()) for i, (c, p) in fixed_pmfs.items()},
        "class": args.cls,
        "history": history,
    }
    torch.save(checkpoint, out_dir / "flow.pt")

    report = {
        "class": args.cls,
        "fix_energy": args.fix_energy,
        "fix_phi": args.fix_phi,
        "warp_phi": args.warp_phi,
        "warp_theta": args.warp_theta,
        "data_file": args.data,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "t_dim": args.t_dim,
        "n_steps": args.n_steps,
        "hidden_dims": list(hidden_dims),
        "best_val_loss": float(best_val_loss),
        "final_mae": {n: float(v) for n, v in zip(OUT_NAMES_ALL, mae)},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"Class: {args.cls}")
    print(f"Fix energy: {args.fix_energy}  Fix phi: {args.fix_phi}  Warp phi: {args.warp_phi}  Warp theta: {args.warp_theta}")
    print(f"Best val loss: {best_val_loss:.5f}")
    print("Final MAE (avg over 10 samples):")
    for n, v in zip(OUT_NAMES_ALL, mae):
        print(f"  {n}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="postprocessed_data.npy")
    parser.add_argument("--class",       dest="cls", default="direct", choices=["direct", "xray", "scatter"])
    parser.add_argument("--out-dir",     default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--t-dim",       type=int,   default=64)
    parser.add_argument("--n-steps",     type=int,   default=50,
                        help="ODE solver steps at inference")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--plot-every",  type=int,   default=20)
    parser.add_argument("--fix-energy",  action="store_true")
    parser.add_argument("--fix-phi",     action="store_true")
    parser.add_argument("--warp-phi",    action="store_true")
    parser.add_argument("--warp-theta",  action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        suffix = ("_fixE" if args.fix_energy else "") + ("_fixP" if args.fix_phi else "") + ("_warpP" if args.warp_phi else "") + ("_warpT" if args.warp_theta else "")
        args.out_dir = f"ml_artifacts/flow_{args.cls}{suffix}"

    train(args)


if __name__ == "__main__":
    main()
