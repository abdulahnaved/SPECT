"""
Train neural networks on postprocessed collimator data.

Two models:
  1. Classifier: incoming photon -> pass probability.
  2. Regressor:  incoming photon -> outgoing photon values (passed photons only).

Usage:
    uv run python train_prototype.py
    uv run python train_prototype.py --hidden-dims 256 256 256 256 --dropout 0.1 --batchnorm
    uv run python train_prototype.py --hidden-dims 512 512 512 512 512 --dropout 0.2 --batchnorm
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


IN_COLS = [0, 1, 2, 3, 4]
OUT_COLS = [5, 6, 7, 8, 9]
OUT_E = 9


def build_mlp(in_dim, out_dim, hidden_dims, dropout=0.0, batchnorm=False):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, hidden_dims=(256, 256, 256, 256), dropout=0.0, batchnorm=False):
        super().__init__()
        self.net = build_mlp(5, 1, hidden_dims, dropout=dropout, batchnorm=batchnorm)

    def forward(self, x):
        return self.net(x).squeeze(1)


class Regressor(nn.Module):
    def __init__(self, hidden_dims=(256, 256, 256, 256), dropout=0.0, batchnorm=False):
        super().__init__()
        self.net = build_mlp(5, 5, hidden_dims, dropout=dropout, batchnorm=batchnorm)

    def forward(self, x):
        return self.net(x)


def split_indices(indices, rng, train_frac=0.70, val_frac=0.15):
    indices = np.array(indices, copy=True)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def standardize_from_train(x_train, *others):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    result = [(x_train - mean) / std]
    result.extend((x - mean) / std for x in others)
    return mean, std, result


def make_loader(x, y, batch_size, shuffle):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def binary_metrics(logits, labels):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy().astype(np.int64)
    pred = probs >= 0.5

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_classifier(data, passed_idx, failed_idx, out_dir, args, device, rng):
    hidden_dims = tuple(args.hidden_dims)
    n_pos = len(passed_idx)
    n_neg = min(len(failed_idx), n_pos * args.negative_ratio)
    sampled_failed = rng.choice(failed_idx, size=n_neg, replace=False)

    pos_train, pos_val, pos_test = split_indices(passed_idx, rng)
    neg_train, neg_val, neg_test = split_indices(sampled_failed, rng)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    x_train = np.asarray(data[train_idx][:, IN_COLS], dtype=np.float32)
    x_val = np.asarray(data[val_idx][:, IN_COLS], dtype=np.float32)
    x_test = np.asarray(data[test_idx][:, IN_COLS], dtype=np.float32)
    y_train = (np.asarray(data[train_idx][:, OUT_E]) > 0).astype(np.float32)
    y_val = (np.asarray(data[val_idx][:, OUT_E]) > 0).astype(np.float32)
    y_test = (np.asarray(data[test_idx][:, OUT_E]) > 0).astype(np.float32)

    x_mean, x_std, (x_train, x_val, x_test) = standardize_from_train(x_train, x_val, x_test)

    model = Classifier(hidden_dims=hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * len(xb)
                logits_all.append(logits.cpu())
                labels_all.append(yb.cpu())
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        metrics = binary_metrics(torch.cat(logits_all), torch.cat(labels_all))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        print(
            f"classifier epoch {epoch:03d}: "
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if args.early_stopping > 0 and patience_count >= args.early_stopping:
                print(f"  early stopping at epoch {epoch} (patience={args.early_stopping})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)
    logits_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits_all.append(model(xb.to(device)).cpu())
            labels_all.append(yb)
    test_metrics = binary_metrics(torch.cat(logits_all), torch.cat(labels_all))

    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_dims": list(hidden_dims),
            "dropout": args.dropout,
            "batchnorm": args.batchnorm,
            "x_mean": x_mean,
            "x_std": x_std,
            "input_columns": IN_COLS,
            "history": history,
            "test_metrics": test_metrics,
        },
        out_dir / "classifier.pt",
    )

    return {
        "positive_rows": int(n_pos),
        "negative_rows_sampled": int(n_neg),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "test_metrics": test_metrics,
    }


def train_regressor(data, passed_idx, out_dir, args, device, rng):
    hidden_dims = tuple(args.hidden_dims)
    train_idx, val_idx, test_idx = split_indices(passed_idx, rng)

    x_train = np.asarray(data[train_idx][:, IN_COLS], dtype=np.float32)
    x_val = np.asarray(data[val_idx][:, IN_COLS], dtype=np.float32)
    x_test = np.asarray(data[test_idx][:, IN_COLS], dtype=np.float32)
    y_train = np.asarray(data[train_idx][:, OUT_COLS], dtype=np.float32)
    y_val = np.asarray(data[val_idx][:, OUT_COLS], dtype=np.float32)
    y_test = np.asarray(data[test_idx][:, OUT_COLS], dtype=np.float32)

    x_mean, x_std, (x_train, x_val, x_test) = standardize_from_train(x_train, x_val, x_test)
    y_mean, y_std, (y_train_s, y_val_s, y_test_s) = standardize_from_train(y_train, y_val, y_test)

    model = Regressor(hidden_dims=hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    train_loader = make_loader(x_train, y_train_s, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val_s, args.batch_size, shuffle=False)

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, args.regressor_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"regressor epoch {epoch:03d}: "
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if args.early_stopping > 0 and patience_count >= args.early_stopping:
                print(f"  early stopping at epoch {epoch} (patience={args.early_stopping})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = make_loader(x_test, y_test_s, args.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_s = model(xb.to(device)).cpu().numpy()
            preds.append(pred_s)
    y_pred = np.concatenate(preds, axis=0) * y_std + y_mean

    mae = np.mean(np.abs(y_pred - y_test), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))

    make_regression_plots(y_test, y_pred, out_dir / "regressor_histograms.png")

    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_dims": list(hidden_dims),
            "dropout": args.dropout,
            "batchnorm": args.batchnorm,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "input_columns": IN_COLS,
            "output_columns": OUT_COLS,
            "history": history,
            "mae": mae,
            "rmse": rmse,
        },
        out_dir / "regressor.pt",
    )

    return {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
    }


def make_regression_plots(y_true, y_pred, out_file):
    names = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.hist(y_true[:, i], bins=40, alpha=0.55, label="Monte Carlo", density=True)
        ax.hist(y_pred[:, i], bins=40, alpha=0.55, label="NN", density=True)
        ax.set_title(name)
    axes[-1].axis("off")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="postprocessed_data.npy")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Defaults to ml_artifacts/hN_wW[_bn][_doD] based on config.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=40)
    parser.add_argument("--regressor-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256],
        help="Hidden layer sizes for both models. E.g. --hidden-dims 256 256 256 256"
    )
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability (0 = disabled).")
    parser.add_argument("--batchnorm", action="store_true",
                        help="Add BatchNorm1d after each hidden layer.")
    parser.add_argument("--early-stopping", type=int, default=15,
                        help="Stop if val loss does not improve for this many epochs. 0 = disabled.")
    args = parser.parse_args()

    if args.out_dir is None:
        depth = len(args.hidden_dims)
        width = args.hidden_dims[0]
        tag = f"h{depth}_w{width}"
        if args.batchnorm:
            tag += "_bn"
        if args.dropout > 0:
            tag += f"_do{int(args.dropout * 100):02d}"
        args.out_dir = f"ml_artifacts/{tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading {args.data}")
    data = np.load(args.data, mmap_mode="r")
    passed_mask = data[:, OUT_E] > 0
    passed_idx = np.flatnonzero(passed_mask)
    failed_idx = np.flatnonzero(~passed_mask)

    n_params_cls = sum(
        p.numel() for p in Classifier(
            hidden_dims=args.hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm
        ).parameters()
    )
    n_params_reg = sum(
        p.numel() for p in Regressor(
            hidden_dims=args.hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm
        ).parameters()
    )

    print(f"Rows: {len(data)}")
    print(f"Passed: {len(passed_idx)}")
    print(f"Failed/lost: {len(failed_idx)}")
    print(f"Transmission: {len(passed_idx) / len(data):.6f}")
    print(f"Device: {device}")
    print(f"Architecture: hidden_dims={args.hidden_dims}, batchnorm={args.batchnorm}, dropout={args.dropout}")
    print(f"Classifier params: {n_params_cls:,}")
    print(f"Regressor params:  {n_params_reg:,}")
    print(f"Output dir: {out_dir}")

    report = {
        "data_file": args.data,
        "rows": int(len(data)),
        "passed": int(len(passed_idx)),
        "failed_or_lost": int(len(failed_idx)),
        "transmission": float(len(passed_idx) / len(data)),
        "architecture": {
            "hidden_dims": args.hidden_dims,
            "batchnorm": args.batchnorm,
            "dropout": args.dropout,
            "classifier_params": n_params_cls,
            "regressor_params": n_params_reg,
        },
    }

    print("\nTraining classifier")
    report["classifier"] = train_classifier(data, passed_idx, failed_idx, out_dir, args, device, rng)

    print("\nTraining outgoing-value regressor")
    report["regressor"] = train_regressor(data, passed_idx, out_dir, args, device, rng)

    report_file = out_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_file}")
    print(f"Saved models: {out_dir / 'classifier.pt'}, {out_dir / 'regressor.pt'}")
    print(f"Saved plot: {out_dir / 'regressor_histograms.png'}")

    cls_m = report["classifier"]["test_metrics"]
    reg_m = report["regressor"]
    print("\n=== Summary ===")
    print(f"Classifier balanced_acc: {cls_m['balanced_accuracy']:.4f}  recall: {cls_m['recall']:.4f}  precision: {cls_m['precision']:.4f}")
    names = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]
    for name, mae in zip(names, reg_m["mae"]):
        print(f"  {name} MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
