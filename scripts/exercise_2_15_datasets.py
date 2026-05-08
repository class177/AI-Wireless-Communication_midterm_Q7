"""
Dataset utilities for Exercise 2.15 official COST2100 exports.

The official MATLAB workflow writes `.mat` files with the following keys:

  HT     : normalized angular-delay CSI, shape [samples, 2048]
  HF_all : complex frequency-domain CSI, shape [samples, 32, 125]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    environment: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("D1_indoor_uniform", "IndoorHall_5GHz"),
    DatasetSpec("D2_indoor_center", "IndoorHall_5GHz"),
    DatasetSpec("D3_indoor_edge", "IndoorHall_5GHz"),
    DatasetSpec("D4_indoor_ring", "IndoorHall_5GHz"),
    DatasetSpec("D5_outdoor_uniform", "SemiUrban_300MHz"),
    DatasetSpec("D6_outdoor_clustered", "SemiUrban_300MHz"),
)


def load_ht(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    mat = sio.loadmat(data_dir / dataset / f"DATA_H{split}.mat")
    return mat["HT"].astype(np.float32)


def load_hf_test(data_dir: Path, dataset: str) -> np.ndarray:
    mat = sio.loadmat(data_dir / dataset / "DATA_HtestF_all.mat")
    return mat["HF_all"]


def mixed_ht(data_dir: Path, datasets: Iterable[str], split: str, limit_per_dataset: int | None = None) -> np.ndarray:
    arrays = []
    for name in datasets:
        arr = load_ht(data_dir, name, split)
        if limit_per_dataset is not None:
            arr = arr[:limit_per_dataset]
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)
