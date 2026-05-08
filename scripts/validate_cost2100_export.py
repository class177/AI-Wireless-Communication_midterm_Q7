"""
Validate official COST2100 exports before running CsiNet.

Expected layout:

data/cost2100_official/<dataset>/DATA_Htrain.mat       key: HT
data/cost2100_official/<dataset>/DATA_Hval.mat         key: HT
data/cost2100_official/<dataset>/DATA_Htest.mat        key: HT
data/cost2100_official/<dataset>/DATA_HtestF_all.mat   key: HF_all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io as sio


DATASETS = (
    "D1_indoor_uniform",
    "D2_indoor_center",
    "D3_indoor_edge",
    "D4_indoor_ring",
    "D5_outdoor_uniform",
    "D6_outdoor_clustered",
)


def check_mat(path: Path, key: str, expected_rank: int) -> tuple[int, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    mat = sio.loadmat(path)
    if key not in mat:
        raise KeyError(f"{path} does not contain key {key!r}")
    shape = mat[key].shape
    if len(shape) != expected_rank:
        raise ValueError(f"{path}:{key} has shape {shape}, expected rank {expected_rank}")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cost2100_official")
    args = parser.parse_args()

    root = Path(args.data_dir)
    for dataset in DATASETS:
        dataset_dir = root / dataset
        train_shape = check_mat(dataset_dir / "DATA_Htrain.mat", "HT", 2)
        val_shape = check_mat(dataset_dir / "DATA_Hval.mat", "HT", 2)
        test_shape = check_mat(dataset_dir / "DATA_Htest.mat", "HT", 2)
        hf_shape = check_mat(dataset_dir / "DATA_HtestF_all.mat", "HF_all", 3)

        if train_shape[1] != 2048 or val_shape[1] != 2048 or test_shape[1] != 2048:
            raise ValueError(f"{dataset}: HT must have 2048 features, got {train_shape}, {val_shape}, {test_shape}")
        if hf_shape[1:] != (32, 125):
            raise ValueError(f"{dataset}: HF_all must have shape [samples, 32, 125], got {hf_shape}")

        print(
            f"{dataset}: train={train_shape}, val={val_shape}, "
            f"test={test_shape}, HF_all={hf_shape}"
        )

    print("All COST2100 exports look compatible with the CsiNet pipeline.")


if __name__ == "__main__":
    main()

