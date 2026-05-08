"""
Plot Exercise 2.15 results.

The script creates figures for the report:
  - NMSE comparison between single-dataset and mixed-dataset training
  - rho comparison
  - NMSE improvement from mixed training
  - optional training loss curves if history CSV files exist
  - CSI reconstruction examples from saved TensorFlow CsiNet weights
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exercise_2_15_datasets import DATASET_SPECS, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, load_ht  # noqa: E402


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_label(name: str) -> str:
    return name.replace("D", "D").replace("_", "\n", 1).replace("_", " ")


def plot_metric(rows: list[dict[str, str]], metric: str, ylabel: str, output: Path) -> None:
    datasets = [spec.name for spec in DATASET_SPECS]
    single = {r["test_dataset"]: float(r[metric]) for r in rows if r["train_type"] == "single_dataset"}
    mixed = {r["test_dataset"]: float(r[metric]) for r in rows if r["train_type"] == "mixed_datasets"}

    x = np.arange(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, [single[d] for d in datasets], width, label="Single-dataset training")
    ax.bar(x + width / 2, [mixed[d] for d in datasets], width, label="Mixed-dataset training")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(d) for d in datasets], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_improvement(rows: list[dict[str, str]], output: Path) -> None:
    datasets = [spec.name for spec in DATASET_SPECS]
    single = {r["test_dataset"]: float(r["nmse_db"]) for r in rows if r["train_type"] == "single_dataset"}
    mixed = {r["test_dataset"]: float(r["nmse_db"]) for r in rows if r["train_type"] == "mixed_datasets"}
    improvement = [single[d] - mixed[d] for d in datasets]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(np.arange(len(datasets)), improvement, color="#3A7D44")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([short_label(d) for d in datasets], fontsize=9)
    ax.set_ylabel("NMSE improvement (dB)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, improvement):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_histories(result_dir: Path, figure_dir: Path) -> None:
    history_files = sorted(result_dir.glob("history_CsiNet_*_dim*_epochs*.csv"))
    if not history_files:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for path in history_files:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        loss = [float(r["loss"]) for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        label = path.stem.replace("history_CsiNet_", "")
        ax.plot(epochs, loss, label=f"{label} train")
        ax.plot(epochs, val_loss, linestyle="--", label=f"{label} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss_curves.png", dpi=220)
    plt.close(fig)


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return None
    return tf


def plot_reconstruction_examples(args: argparse.Namespace, figure_dir: Path) -> None:
    tf = import_tensorflow()
    if tf is None:
        return

    from scripts.run_exercise_2_15_tf import build_csinet, reshape_ht

    model_path = ROOT / "saved_model" / f"CsiNet_mixed_all_dim{args.encoded_dim}.weights.h5"
    if not model_path.exists():
        return

    model = build_csinet(tf, args.encoded_dim)
    model.load_weights(model_path)

    data_dir = ROOT / args.data_dir
    for dataset in args.reconstruction_datasets:
        x_flat = load_ht(data_dir, dataset, "test")[: args.examples]
        x = reshape_ht(x_flat)
        x_hat = model.predict(x, batch_size=args.examples, verbose=0)

        fig, axes = plt.subplots(2, args.examples, figsize=(2.2 * args.examples, 4.4))
        for idx in range(args.examples):
            original = np.abs((x[idx, :, :, 0] - 0.5) + 1j * (x[idx, :, :, 1] - 0.5))
            reconstructed = np.abs((x_hat[idx, :, :, 0] - 0.5) + 1j * (x_hat[idx, :, :, 1] - 0.5))
            axes[0, idx].imshow(original.T, cmap="gray", origin="lower")
            axes[1, idx].imshow(reconstructed.T, cmap="gray", origin="lower")
            axes[0, idx].set_title(f"Sample {idx + 1}", fontsize=9)
            axes[0, idx].axis("off")
            axes[1, idx].axis("off")
        axes[0, 0].set_ylabel("Original", fontsize=10)
        axes[1, 0].set_ylabel("Mixed CsiNet", fontsize=10)
        fig.suptitle(dataset, fontsize=12)
        fig.tight_layout()
        fig.savefig(figure_dir / f"reconstruction_{dataset}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="result/exercise_2_15_csinet_results.csv")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--figure-dir", default="result/figures")
    parser.add_argument("--data-dir", default="data/cost2100_official")
    parser.add_argument("--encoded-dim", type=int, default=512)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--reconstruction-datasets", nargs="*", default=["D1_indoor_uniform", "D5_outdoor_uniform", "D6_outdoor_clustered"])
    args = parser.parse_args()

    figure_dir = ROOT / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = read_results(ROOT / args.results)
    plot_metric(rows, "nmse_db", "NMSE (dB), lower is better", figure_dir / "nmse_comparison.png")
    plot_metric(rows, "rho", "Correlation coefficient rho, higher is better", figure_dir / "rho_comparison.png")
    plot_improvement(rows, figure_dir / "nmse_improvement.png")
    plot_histories(ROOT / args.result_dir, figure_dir)
    plot_reconstruction_examples(args, figure_dir)
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
