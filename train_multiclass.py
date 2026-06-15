"""
Multi-class collimator transport model.

Classes:
  0  BLOCKED  photon did not exit
  1  DIRECT   primary photon exited with energy close to incoming (within 5%)
  2  XRAY     outgoing photon is a secondary (TrackID > 1) — Pb fluorescence
  3  SCATTER  primary photon exited after losing significant energy (Compton)

One 4-class classifier is trained on all photons.
One regressor per pass class (DIRECT, XRAY, SCATTER) trained only on that class.

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


IN_COLS  = [0, 1, 2, 3, 4]
OUT_COLS = [5, 6, 7, 8, 9]
IN_E  = 4
OUT_E = 9
IS_SECONDARY_COL = 10

CLASS_NAMES = ["blocked", "direct", "xray", "scatter"]
BLOCKED = 0
DIRECT  = 1
XRAY    = 2
SCATTER = 3

DIRECT_REL_TOL = 0.05  # |out_E - in_E| / in_E < 5% => direct


def assign_classes(data):
    in_e         = data[:, IN_E]
    out_e        = data[:, OUT_E]
    is_secondary = data[:, IS_SECONDARY_COL] > 0

    labels = np.full(len(data), BLOCKED, dtype=np.int64)

    passed  = out_e > 0
    xray    = passed & is_secondary
    direct  = passed & ~is_secondary & (np.abs(out_e - in_e) / np.maximum(in_e, 1e-6) < DIRECT_REL_TOL)
    scatter = passed & ~is_secondary & ~direct

    labels[direct]  = DIRECT
    labels[xray]    = XRAY
    labels[scatter] = SCATTER

    return labels


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


class MultiClassifier(nn.Module):
    def __init__(self, n_classes=4, hidden_dims=(256, 256, 256, 256), dropout=0.0, batchnorm=False):
        super().__init__()
        self.net = build_mlp(5, n_classes, hidden_dims, dropout=dropout, batchnorm=batchnorm)

    def forward(self, x):
        return self.net(x)


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
    n_val   = int(n * val_frac)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def standardize_from_train(x_train, *others):
    mean = x_train.mean(axis=0)
    std  = x_train.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    result = [(x_train - mean) / std]
    result.extend((x - mean) / std for x in others)
    return mean, std, result


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y)),
        batch_size=batch_size, shuffle=shuffle
    )


def train_classifier(data, labels, out_dir, args, device, rng):
    hidden_dims = tuple(args.hidden_dims)

    pass_idx    = np.flatnonzero(labels != BLOCKED)
    blocked_idx = np.flatnonzero(labels == BLOCKED)
    n_sample_blocked = min(len(blocked_idx), len(pass_idx) * args.negative_ratio)
    sampled_blocked  = rng.choice(blocked_idx, size=n_sample_blocked, replace=False)
    all_idx = np.concatenate([pass_idx, sampled_blocked])
    rng.shuffle(all_idx)

    per_class_splits = {}
    for c in range(4):
        cidx = all_idx[labels[all_idx] == c]
        if len(cidx) == 0:
            per_class_splits[c] = (np.array([]), np.array([]), np.array([]))
        else:
            per_class_splits[c] = split_indices(cidx, rng)

    train_idx = np.concatenate([per_class_splits[c][0] for c in range(4)])
    val_idx   = np.concatenate([per_class_splits[c][1] for c in range(4)])
    test_idx  = np.concatenate([per_class_splits[c][2] for c in range(4)])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    x_train = np.asarray(data[train_idx][:, IN_COLS], dtype=np.float32)
    x_val   = np.asarray(data[val_idx][:, IN_COLS],   dtype=np.float32)
    x_test  = np.asarray(data[test_idx][:, IN_COLS],  dtype=np.float32)
    y_train = labels[train_idx]
    y_val   = labels[val_idx]
    y_test  = labels[test_idx]

    x_mean, x_std, (x_train, x_val, x_test) = standardize_from_train(x_train, x_val, x_test)

    # class weights: inverse frequency over the sampled training set
    train_counts = np.bincount(y_train, minlength=4).astype(np.float32)
    train_counts = np.where(train_counts == 0, 1.0, train_counts)
    weights = torch.tensor(1.0 / train_counts, dtype=torch.float32).to(device)

    model     = MultiClassifier(hidden_dims=hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    loss_fn   = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(x_val,   y_val,   args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state    = None
    patience_count = 0
    history = []

    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"classifier epoch {epoch:03d}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if args.early_stopping > 0 and patience_count >= args.early_stopping:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # test metrics
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(yb.numpy())
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    overall_acc = float((all_preds == all_labels).mean())
    per_class_acc = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == all_labels[mask]).mean())

    # confusion matrix
    conf = np.zeros((4, 4), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf[true, pred] += 1

    torch.save({
        "model_state": model.state_dict(),
        "hidden_dims": list(hidden_dims),
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "x_mean": x_mean,
        "x_std": x_std,
        "input_columns": IN_COLS,
        "class_names": CLASS_NAMES,
        "history": history,
    }, out_dir / "classifier.pt")

    return {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf.tolist(),
    }


def train_one_regressor(data, idx, class_name, out_dir, args, device, rng):
    hidden_dims = tuple(args.hidden_dims)
    train_idx, val_idx, test_idx = split_indices(idx, rng)

    x_train = np.asarray(data[train_idx][:, IN_COLS], dtype=np.float32)
    x_val   = np.asarray(data[val_idx][:, IN_COLS],   dtype=np.float32)
    x_test  = np.asarray(data[test_idx][:, IN_COLS],  dtype=np.float32)
    y_train = np.asarray(data[train_idx][:, OUT_COLS], dtype=np.float32)
    y_val   = np.asarray(data[val_idx][:, OUT_COLS],   dtype=np.float32)
    y_test  = np.asarray(data[test_idx][:, OUT_COLS],  dtype=np.float32)

    x_mean, x_std, (x_train, x_val, x_test) = standardize_from_train(x_train, x_val, x_test)
    y_mean, y_std, (y_train_s, y_val_s, y_test_s) = standardize_from_train(y_train, y_val, y_test)

    model     = Regressor(hidden_dims=hidden_dims, dropout=args.dropout, batchnorm=args.batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    loss_fn   = nn.MSELoss()

    train_loader = make_loader(x_train, y_train_s, args.batch_size, shuffle=True)
    val_loader   = make_loader(x_val,   y_val_s,   args.batch_size, shuffle=False)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, args.regressor_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"  [{class_name}] epoch {epoch:03d}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if args.early_stopping > 0 and patience_count >= args.early_stopping:
                print(f"  [{class_name}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    test_loader = make_loader(x_test, y_test_s, args.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds) * y_std + y_mean

    mae  = np.mean(np.abs(y_pred - y_test), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))

    make_regression_plots(y_test, y_pred, out_dir / f"regressor_{class_name}_histograms.png", class_name)

    torch.save({
        "model_state": model.state_dict(),
        "hidden_dims": list(hidden_dims),
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "input_columns": IN_COLS,
        "output_columns": OUT_COLS,
        "class_name": class_name,
        "mae": mae, "rmse": rmse,
    }, out_dir / f"regressor_{class_name}.pt")

    return {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
    }


def make_regression_plots(y_true, y_pred, out_file, title):
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
    fig.suptitle(f"Regressor: {title}")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="postprocessed_data.npy")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--classifier-epochs", type=int, default=40)
    parser.add_argument("--regressor-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--early-stopping", type=int, default=15)
    args = parser.parse_args()

    if args.out_dir is None:
        depth = len(args.hidden_dims)
        width = args.hidden_dims[0]
        tag = f"multiclass_h{depth}_w{width}"
        if args.batchnorm:
            tag += "_bn"
        if args.dropout > 0:
            tag += f"_do{int(args.dropout * 100):02d}"
        args.out_dir = f"ml_artifacts/{tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng    = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.data}")
    data   = np.load(args.data, mmap_mode="r")
    labels = assign_classes(data)

    class_counts = np.bincount(labels, minlength=4)
    print(f"\nClass distribution:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s}: {class_counts[c]:>10,}  ({100*class_counts[c]/len(data):.4f}%)")
    print(f"  {'total':10s}: {len(data):>10,}")
    print(f"\nDevice: {device}")
    print(f"Architecture: hidden_dims={args.hidden_dims}, batchnorm={args.batchnorm}, dropout={args.dropout}")
    print(f"Output dir: {out_dir}")

    report = {
        "data_file": args.data,
        "class_counts": {name: int(class_counts[c]) for c, name in enumerate(CLASS_NAMES)},
        "architecture": {
            "hidden_dims": args.hidden_dims,
            "batchnorm": args.batchnorm,
            "dropout": args.dropout,
        },
    }

    print("\n--- Training multi-class classifier ---")
    report["classifier"] = train_classifier(data, labels, out_dir, args, device, rng)

    print("\n--- Training per-class regressors ---")
    report["regressors"] = {}
    for c, name in enumerate(CLASS_NAMES):
        if c == BLOCKED:
            continue
        idx = np.flatnonzero(labels == c)
        if len(idx) < 100:
            print(f"  [{name}] skipping — only {len(idx)} samples")
            continue
        print(f"\nRegressor: {name} ({len(idx):,} photons)")
        report["regressors"][name] = train_one_regressor(data, idx, name, out_dir, args, device, rng)

    report_file = out_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    cls = report["classifier"]
    print(f"Classifier overall accuracy: {cls['overall_accuracy']:.4f}")
    print(f"Per-class accuracy:")
    for name, acc in cls["per_class_accuracy"].items():
        print(f"  {name:10s}: {acc:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':10s} " + "  ".join(f"{n:>10s}" for n in CLASS_NAMES))
    for i, row in enumerate(cls["confusion_matrix"]):
        print(f"  {CLASS_NAMES[i]:10s} " + "  ".join(f"{v:>10,}" for v in row))

    out_names = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]
    for name, reg in report["regressors"].items():
        print(f"\n{name} regressor MAE:")
        for col, mae in zip(out_names, reg["mae"]):
            print(f"  {col}: {mae:.4f}")


if __name__ == "__main__":
    main()
