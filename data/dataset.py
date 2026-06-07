"""
Time series datasets for long-term forecasting benchmarks.

Matches the standard preprocessing used by PatchTST/iTransformer/SAMformer
which the PSformer paper compares against (Appendix A.2):
  - Look-back window T = 512 (set in paper)
  - Sliding window with stride 1
  - Z-score normalization fitted on TRAIN split only
  - Standard splits per dataset (Table 1 in paper)
  - test loader: drop_last=False (paper explicitly notes this, Appendix A.2)

Supported datasets:
  ETTh1, ETTh2 :  7 vars, hourly,  splits (8545, 2881, 2881)   [12mo train, 4mo val, 4mo test]
  ETTm1, ETTm2 :  7 vars, 15-min,  splits (34465, 11521, 11521)
  Weather      :  21 vars, 10-min, splits (36792, 5271, 10540)  [70/10/20]
  Electricity  :  321 vars, hourly, splits (18317, 2633, 5261)  [70/10/20]
  Exchange     :  8 vars, daily,   splits (5120, 665, 1422)     [70/10/20]
  Traffic      :  862 vars, hourly, splits (12185, 1757, 3509)  [70/10/20]

Note: split lengths in Table 1 are *number of windows*, not raw rows. The raw-row
splits below follow PatchTST/iTransformer conventions (which the paper says it
matches in Appendix A.2). The number-of-windows reported in Table 1 equals
(split_length - look_back - horizon + 1) summed appropriately for ETT, and
(split_length - look_back + 1) for the others if a sample is one (input, target)
pair. We verify this matches Table 1 in tests.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Column name in each CSV that is treated as the "target" for univariate-tasks;
# PSformer paper uses MULTIVARIATE forecasting, so we keep all columns. This is
# only documented here in case we want univariate ablations later.
DATASET_TARGET_COL = {
    "ETTh1": "OT", "ETTh2": "OT", "ETTm1": "OT", "ETTm2": "OT",
    "Weather": "OT", "Electricity": "OT", "Exchange": "OT", "Traffic": "OT",
}


def _ett_borders(name: str) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """ETT splits use the canonical fixed boundaries from Informer/Autoformer."""
    if name in ("ETTh1", "ETTh2"):
        # Hourly: 12 months train, 4 months val, 4 months test
        train = (0, 12 * 30 * 24)                          # 8640
        val   = (12 * 30 * 24, 16 * 30 * 24)               # 8640..11520
        test  = (16 * 30 * 24, 20 * 30 * 24)               # 11520..14400
    elif name in ("ETTm1", "ETTm2"):
        # 15-min: 12 months train, 4 months val, 4 months test (×4 freq)
        train = (0, 12 * 30 * 24 * 4)                      # 34560
        val   = (12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test  = (16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    else:
        raise ValueError(name)
    return train, val, test


def _ratio_borders(n: int, ratios=(0.7, 0.1, 0.2)
                   ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """70/10/20 splits used by Weather, Electricity, Exchange, Traffic."""
    n_train = int(n * ratios[0])
    n_test = int(n * ratios[2])
    n_val = n - n_train - n_test
    train = (0, n_train)
    val   = (n_train, n_train + n_val)
    test  = (n_train + n_val, n)
    return train, val, test


class TSForecastDataset(Dataset):
    """
    Sliding-window dataset for multivariate long-term forecasting.

    Each item is a tuple (x, y) where:
      x: (M, L) lookback window of length L (default 512)
      y: (M, H) target horizon of length H

    The boundaries are configured so that:
      For split 's' ∈ {train, val, test}, indices used for windows lie in
      [s_start, s_end - 1]. To predict from (x, y), we need x to span
      [i, i+L) and y to span [i+L, i+L+H), so the last valid i is
      s_end - L - H. The train split's lookback is allowed to extend back
      into earlier data ONLY in val/test (so that models conditioning on
      the most recent L points of the train+val region can predict the
      first val/test target). This is the standard PatchTST/iTransformer
      convention.
    """

    def __init__(self, root_path: str, dataset_name: str, split: str,
                 lookback: int = 512, horizon: int = 96,
                 scale_stats=None):
        assert split in ("train", "val", "test"), split
        self.dataset_name = dataset_name
        self.split = split
        self.L = lookback
        self.H = horizon

        df = self._read_csv(root_path, dataset_name)
        self.M = df.shape[1]  # number of variables
        data = df.values.astype(np.float32)  # shape (T_total, M)

        # Compute split borders
        if dataset_name.startswith("ETT"):
            tr, va, te = _ett_borders(dataset_name)
        else:
            tr, va, te = _ratio_borders(len(data))
        self.borders = {"train": tr, "val": va, "test": te}

        # Standardize using TRAIN-split statistics only (z-score per variable)
        if scale_stats is None:
            train_slice = data[tr[0]:tr[1]]
            self.mean = train_slice.mean(axis=0)            # (M,)
            self.std = train_slice.std(axis=0) + 1e-8       # (M,)
        else:
            self.mean, self.std = scale_stats
        data = (data - self.mean) / self.std

        # For val/test, the lookback is allowed to extend BACK into the
        # previous split (to forecast the very first target in the split).
        # i.e. we shift the window-start by -L from the split boundary.
        s_start, s_end = self.borders[split]
        if split != "train":
            s_start = max(0, s_start - self.L)
        # Last index where (i, i+L) is the lookback and (i+L, i+L+H) is target
        last_i = s_end - self.L - self.H
        # Number of usable starting indices
        self.num_windows = last_i - s_start + 1
        self.start = s_start
        self.data = data  # full normalized array, indexed by absolute t

        if self.num_windows <= 0:
            raise ValueError(
                f"Not enough rows in '{split}' for L={self.L}, H={self.H}: "
                f"{self.num_windows}"
            )

    @staticmethod
    def _read_csv(root_path: str, dataset_name: str) -> pd.DataFrame:
        filename_map = {
            "ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv",
            "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
            "Weather": "weather.csv",
            "Electricity": "electricity.csv",
            "Exchange": "exchange_rate.csv",
            "Traffic": "traffic.csv",
        }
        if dataset_name not in filename_map:
            raise ValueError(f"Unknown dataset {dataset_name}")
        path = os.path.join(root_path, filename_map[dataset_name])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {path}. Use scripts/download_data.sh to fetch."
            )
        df = pd.read_csv(path)
        # Drop the date column if present
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        return df

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        i = self.start + idx
        # x: (L, M) -> transpose to (M, L) to match PSformer's R^{M x L} convention
        x = self.data[i:i + self.L]                          # (L, M)
        y = self.data[i + self.L:i + self.L + self.H]        # (H, M)
        x = torch.from_numpy(x.T.copy())                     # (M, L)
        y = torch.from_numpy(y.T.copy())                     # (M, H)
        return x, y

    def get_scale_stats(self):
        return (self.mean, self.std)


def make_loaders(root_path: str, dataset_name: str, lookback: int, horizon: int,
                 batch_size: int, num_workers: int = 4):
    """Factory: creates train/val/test DataLoaders matching the paper's setup."""
    from torch.utils.data import DataLoader

    train_ds = TSForecastDataset(root_path, dataset_name, "train", lookback, horizon)
    stats = train_ds.get_scale_stats()
    val_ds = TSForecastDataset(root_path, dataset_name, "val", lookback, horizon,
                               scale_stats=stats)
    test_ds = TSForecastDataset(root_path, dataset_name, "test", lookback, horizon,
                                scale_stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, pin_memory=True)
    # IMPORTANT: paper explicitly notes drop_last=False for test (Appendix A.2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.M
