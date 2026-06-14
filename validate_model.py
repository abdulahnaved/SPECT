"""
Validate the prototype models beyond one global histogram.

This script is meant to run on the server, where the large processed dataset
already lives. It produces:
  - classifier pass-rate checks in input bins
  - outgoing-value histograms in incoming-energy bins
  - outgoing-value histograms in incoming-angle bins
  - predicted-vs-Monte-Carlo scatter plots for the regressor
  - a compact JSON summary

"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_prototype import Classifier, Regressor, IN_COLS, OUT_COLS, OUT_E


IN_NAMES = ["in_x", "in_y", "in_theta", "in_phi", "in_E"]
OUT_NAMES = ["out_x", "out_y", "out_theta", "out_phi", "out_E"]


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(model_cls, checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    model = model_cls().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def sample_indices(indices, max_count, rng):
    indices = np.asarray(indices)
    if len(indices) <= max_count:
        return indices
    return rng.choice(indices, size=max_count, replace=False)


def predict_classifier(model, checkpoint, x, device, batch_size):
    x = np.asarray(x, dtype=np.float32)
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def predict_regressor(model, checkpoint, x, device, batch_size):
    x = np.asarray(x, dtype=np.float32)
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            pred_s = model(xb).cpu().numpy()
            preds.append(pred_s)
    pred_s = np.concatenate(preds, axis=0)
    return pred_s * checkpoint["y_std"] + checkpoint["y_mean"]


def prior_correct_balanced_probability(p_balanced, real_prior):
    """Convert balanced-training probability to approximate real-prior probability."""
    eps = 1e-6
    p_balanced = np.clip(p_balanced, eps, 1.0 - eps)
    balanced_odds = p_balanced / (1.0 - p_balanced)
    real_odds = balanced_odds * real_prior / (1.0 - real_prior)
    return real_odds / (1.0 + real_odds)


def binned_rates(values, labels, predicted, bins):
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (values >= low) & (values < high)
        count = int(mask.sum())
        if count == 0:
            actual_rate = float("nan")
            predicted_rate = float("nan")
        else:
            actual_rate = float(labels[mask].mean())
            predicted_rate = float(predicted[mask].mean())
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "count": count,
                "actual_pass_rate": actual_rate,
                "predicted_pass_rate": predicted_rate,
            }
        )
    return rows


def plot_binned_rates(rows, title, xlabel, out_file):
    centers = np.array([(r["low"] + r["high"]) / 2.0 for r in rows])
    actual = np.array([r["actual_pass_rate"] for r in rows])
    predicted = np.array([r["predicted_pass_rate"] for r in rows])
    width = np.array([r["high"] - r["low"] for r in rows]) * 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers - width / 2.0, actual, width=width, label="Monte Carlo", alpha=0.75)
    ax.bar(centers + width / 2.0, predicted, width=width, label="NN calibrated", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pass rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def make_conditional_histograms(x, y_true, y_pred, condition_col, bins, condition_name, out_name, out_file):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    out_idx = OUT_NAMES.index(out_name)
    condition_values = x[:, condition_col]

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        ax = axes[i]
        mask = (condition_values >= low) & (condition_values < high)
        if mask.sum() == 0:
            ax.set_axis_off()
            continue
        ax.hist(y_true[mask, out_idx], bins=45, alpha=0.55, density=True, label="Monte Carlo")
        ax.hist(y_pred[mask, out_idx], bins=45, alpha=0.55, density=True, label="NN")
        ax.set_title(f"{condition_name}: {low:.2g} to {high:.2g} ({mask.sum()})")
    for ax in axes[len(bins) - 1 :]:
        ax.set_axis_off()
    axes[0].legend()
    fig.suptitle(f"{out_name} conditioned on {condition_name}")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def make_scatter_plots(y_true, y_pred, out_file, max_points=20000):
    if len(y_true) > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, name in enumerate(OUT_NAMES):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=2, alpha=0.25)
        low = float(min(y_true[:, i].min(), y_pred[:, i].min()))
        high = float(max(y_true[:, i].max(), y_pred[:, i].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Monte Carlo")
        ax.set_ylabel("NN")
    axes[-1].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def regression_summary(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    return {
        name: {"mae": float(mae[i]), "rmse": float(rmse[i])}
        for i, name in enumerate(OUT_NAMES)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Processed .npy data file")
    parser.add_argument("--model-dir", required=True, help="Directory with classifier.pt and regressor.pt")
    parser.add_argument("--out-dir", required=True, help="Directory for validation outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--classifier-sample", type=int, default=2_000_000)
    parser.add_argument("--passed-sample", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    rng = np.random.default_rng(args.seed)
    device = choose_device()

    print(f"Loading data: {args.data}")
    data = np.load(args.data, mmap_mode="r")
    labels_all = data[:, OUT_E] > 0
    passed_idx = np.flatnonzero(labels_all)
    failed_idx = np.flatnonzero(~labels_all)
    real_prior = len(passed_idx) / len(data)

    print(f"Rows: {len(data)}")
    print(f"Passed: {len(passed_idx)}")
    print(f"Transmission: {real_prior:.6g}")
    print(f"Device: {device}")

    classifier, classifier_ckpt = load_model(Classifier, model_dir / "classifier.pt", device)
    regressor, regressor_ckpt = load_model(Regressor, model_dir / "regressor.pt", device)

    classifier_idx = sample_indices(np.arange(len(data)), args.classifier_sample, rng)
    x_classifier = np.asarray(data[classifier_idx][:, IN_COLS], dtype=np.float32)
    y_classifier = np.asarray(data[classifier_idx][:, OUT_E] > 0, dtype=np.float32)
    p_balanced = predict_classifier(
        classifier, classifier_ckpt, x_classifier, device, args.batch_size
    )
    p_real = prior_correct_balanced_probability(p_balanced, real_prior)

    energy_bins = np.array([0, 50, 100, 150, 200, 250.1], dtype=np.float32)
    theta_bins = np.quantile(x_classifier[:, IN_NAMES.index("in_theta")], np.linspace(0, 1, 6))
    radius = np.sqrt(x_classifier[:, 0] ** 2 + x_classifier[:, 1] ** 2)
    radius_bins = np.quantile(radius, np.linspace(0, 1, 6))

    classifier_bins = {
        "incoming_energy": binned_rates(
            x_classifier[:, IN_NAMES.index("in_E")], y_classifier, p_real, energy_bins
        ),
        "incoming_theta": binned_rates(
            x_classifier[:, IN_NAMES.index("in_theta")], y_classifier, p_real, theta_bins
        ),
        "incoming_radius": binned_rates(radius, y_classifier, p_real, radius_bins),
    }

    plot_binned_rates(
        classifier_bins["incoming_energy"],
        "Pass rate by incoming energy",
        "incoming energy (keV)",
        out_dir / "pass_rate_by_energy.png",
    )
    plot_binned_rates(
        classifier_bins["incoming_theta"],
        "Pass rate by incoming theta",
        "incoming theta (rad)",
        out_dir / "pass_rate_by_theta.png",
    )
    plot_binned_rates(
        classifier_bins["incoming_radius"],
        "Pass rate by incoming radius",
        "incoming radius (mm)",
        out_dir / "pass_rate_by_radius.png",
    )

    reg_idx = sample_indices(passed_idx, args.passed_sample, rng)
    x_reg = np.asarray(data[reg_idx][:, IN_COLS], dtype=np.float32)
    y_true = np.asarray(data[reg_idx][:, OUT_COLS], dtype=np.float32)
    y_pred = predict_regressor(regressor, regressor_ckpt, x_reg, device, args.batch_size)

    make_conditional_histograms(
        x_reg,
        y_true,
        y_pred,
        IN_NAMES.index("in_E"),
        energy_bins,
        "incoming energy (keV)",
        "out_E",
        out_dir / "out_energy_by_incoming_energy.png",
    )
    reg_theta_bins = np.quantile(x_reg[:, IN_NAMES.index("in_theta")], np.linspace(0, 1, 6))
    make_conditional_histograms(
        x_reg,
        y_true,
        y_pred,
        IN_NAMES.index("in_theta"),
        reg_theta_bins,
        "incoming theta (rad)",
        "out_theta",
        out_dir / "out_theta_by_incoming_theta.png",
    )
    make_scatter_plots(y_true, y_pred, out_dir / "regressor_predicted_vs_mc.png")

    report = {
        "data_file": args.data,
        "model_dir": args.model_dir,
        "rows": int(len(data)),
        "passed": int(len(passed_idx)),
        "failed_or_lost": int(len(failed_idx)),
        "transmission": float(real_prior),
        "classifier_sample": int(len(classifier_idx)),
        "passed_sample": int(len(reg_idx)),
        "classifier_bins": classifier_bins,
        "regressor": regression_summary(y_true, y_pred),
        "outputs": [
            "pass_rate_by_energy.png",
            "pass_rate_by_theta.png",
            "pass_rate_by_radius.png",
            "out_energy_by_incoming_energy.png",
            "out_theta_by_incoming_theta.png",
            "regressor_predicted_vs_mc.png",
        ],
    }

    report_file = out_dir / "validation_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation report: {report_file}")
    for output in report["outputs"]:
        print(f"Saved plot: {out_dir / output}")


if __name__ == "__main__":
    main()
