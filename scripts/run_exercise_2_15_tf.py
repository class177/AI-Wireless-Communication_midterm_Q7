"""
TensorFlow CsiNet runner for Exercise 2.15.

Run this script in a TensorFlow-compatible Python environment, for example
Python 3.10 or 3.11 with tensorflow installed.  It consumes the datasets
exported from the official COST2100 MATLAB workflow.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exercise_2_15_datasets import (  # noqa: E402
    DATASET_SPECS,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    load_hf_test,
    load_ht,
    mixed_ht,
)


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is not installed in this Python environment. "
            "Use Python 3.10/3.11 and install tensorflow in the "
            "`csinet_tf` environment."
        ) from exc
    return tf


def build_csinet(tf, encoded_dim: int, residual_num: int = 2):
    layers = tf.keras.layers
    image_tensor = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    def common(y):
        y = layers.BatchNormalization(axis=-1)(y)
        return layers.LeakyReLU()(y)

    def residual_block(y):
        shortcut = y
        y = layers.Conv2D(8, (3, 3), padding="same", data_format="channels_last")(y)
        y = common(y)
        y = layers.Conv2D(16, (3, 3), padding="same", data_format="channels_last")(y)
        y = common(y)
        y = layers.Conv2D(2, (3, 3), padding="same", data_format="channels_last")(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Add()([shortcut, y])
        return layers.LeakyReLU()(y)

    x = layers.Conv2D(2, (3, 3), padding="same", data_format="channels_last")(image_tensor)
    x = common(x)
    x = layers.Reshape((IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,))(x)
    encoded = layers.Dense(encoded_dim, activation="linear")(x)
    x = layers.Dense(IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH, activation="linear")(encoded)
    x = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))(x)
    for _ in range(residual_num):
        x = residual_block(x)
    output = layers.Conv2D(2, (3, 3), activation="sigmoid", padding="same", data_format="channels_last")(x)
    model = tf.keras.Model(inputs=image_tensor, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def reshape_ht(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).transpose(0, 2, 3, 1).astype(np.float32)


def evaluate_reconstruction(x_test: np.ndarray, x_hat: np.ndarray, hf_test: np.ndarray) -> tuple[float, float]:
    x_test_c = x_test[:, :, :, 0].reshape(len(x_test), -1) - 0.5
    x_test_c = x_test_c + 1j * (x_test[:, :, :, 1].reshape(len(x_test), -1) - 0.5)
    x_hat_c = x_hat[:, :, :, 0].reshape(len(x_hat), -1) - 0.5
    x_hat_c = x_hat_c + 1j * (x_hat[:, :, :, 1].reshape(len(x_hat), -1) - 0.5)

    power = np.sum(np.abs(x_test_c) ** 2, axis=1)
    mse = np.sum(np.abs(x_test_c - x_hat_c) ** 2, axis=1)
    nmse_db = 10.0 * math.log10(float(np.mean(mse / np.maximum(power, 1e-12))))

    x_hat_ad = x_hat_c.reshape(len(x_hat_c), IMG_HEIGHT, IMG_WIDTH)
    x_hat_f = np.fft.fft(
        np.concatenate((x_hat_ad, np.zeros((len(x_hat_ad), IMG_HEIGHT, 257 - IMG_WIDTH))), axis=2),
        axis=2,
    )[:, :, :125]
    n1 = np.sqrt(np.real(np.sum(np.conj(hf_test) * hf_test, axis=1))).astype("float64")
    n2 = np.sqrt(np.real(np.sum(np.conj(x_hat_f) * x_hat_f, axis=1))).astype("float64")
    aa = np.abs(np.sum(np.conj(hf_test) * x_hat_f, axis=1))
    rho = float(np.mean(np.mean(aa / np.maximum(n1 * n2, 1e-12), axis=1)))
    return nmse_db, rho


def train_and_evaluate(args: argparse.Namespace) -> Path:
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = ROOT / args.data_dir
    result_dir = ROOT / args.result_dir
    model_dir = ROOT / "saved_model"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [spec.name for spec in DATASET_SPECS]
    training_sets = [
        ("single_dataset", args.baseline_dataset, load_ht(data_dir, args.baseline_dataset, "train"), load_ht(data_dir, args.baseline_dataset, "val")),
        ("mixed_datasets", "mixed_all", mixed_ht(data_dir, dataset_names, "train", args.mix_limit), mixed_ht(data_dir, dataset_names, "val", args.val_limit)),
    ]

    rows = []
    for train_type, train_name, x_train_flat, x_val_flat in training_sets:
        x_train = reshape_ht(x_train_flat)
        x_val = reshape_ht(x_val_flat)
        model = build_csinet(tf, args.encoded_dim, args.residual_num)
        started = time.time()
        history = model.fit(
            x_train,
            x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            validation_data=(x_val, x_val),
            verbose=2,
        )
        train_seconds = time.time() - started

        history_output = result_dir / f"history_CsiNet_{train_name}_dim{args.encoded_dim}_epochs{args.epochs}.csv"
        with history_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "val_loss"])
            for idx, (loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"]), start=1):
                writer.writerow([idx, loss, val_loss])

        model.save_weights(model_dir / f"CsiNet_{train_name}_dim{args.encoded_dim}.weights.h5")

        for test_name in dataset_names:
            x_test = reshape_ht(load_ht(data_dir, test_name, "test"))
            hf_test = load_hf_test(data_dir, test_name)
            started = time.time()
            x_hat = model.predict(x_test, batch_size=args.batch_size, verbose=0)
            infer_seconds = (time.time() - started) / len(x_test)
            nmse_db, rho = evaluate_reconstruction(x_test, x_hat, hf_test)
            rows.append(
                {
                    "model": "CsiNet",
                    "train_type": train_type,
                    "train_dataset": train_name,
                    "test_dataset": test_name,
                    "encoded_dim": args.encoded_dim,
                    "epochs": args.epochs,
                    "train_samples": len(x_train),
                    "test_samples": len(x_test),
                    "nmse_db": round(nmse_db, 4),
                    "rho": round(rho, 6),
                    "train_seconds": round(train_seconds, 3),
                    "infer_seconds_per_sample": f"{infer_seconds:.8f}",
                }
            )

    output = result_dir / "exercise_2_15_csinet_results.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cost2100_official")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--encoded-dim", type=int, default=512)
    parser.add_argument("--residual-num", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--baseline-dataset", default="D1_indoor_uniform")
    parser.add_argument("--mix-limit", type=int, default=2500)
    parser.add_argument("--val-limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=535100)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
