"""
Inference script for a trained PSformer.

Two modes:

1) Test-set evaluation (default):
     python -m experiments.predict --ckpt ./checkpoints/psformer_ETTh1_H96.pt \
         --data_root ./datasets

   Loads the checkpoint, re-creates the test loader using the config saved
   inside the checkpoint, and reports MSE / MAE matching the training run.

2) Forecast on a custom (B, M, L) lookback window:
     python -m experiments.predict --ckpt CKPT --input_npy lookback.npy \
         --output_npy forecast.npy

   `lookback.npy` must contain a numpy array of shape (B, M, L) where M and
   L match the values stored in the checkpoint config. The output is the
   forecast in the SAME normalized scale as the model's training data.
   To convert back to real units, multiply by std and add mean (saved in
   the checkpoint as 'scaler' if present; otherwise compute from your data).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from data import make_loaders
from models import PSformer


def load_model(ckpt_path: str, device: str):
    sd = torch.load(ckpt_path, map_location=device)
    cfg = sd["config"]
    model = PSformer(
        num_vars=cfg["num_vars"],
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
        segments=cfg["segments"],
        num_encoders=cfg["num_encoders"],
        revin_stat_window=cfg["revin_stat_window"],
        revin_affine=cfg["revin_affine"],
    ).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()
    return model, cfg, sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pt file")
    parser.add_argument("--data_root", default="./datasets")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_npy", default=None,
                        help="path to .npy with shape (B, M, L) for forecasting")
    parser.add_argument("--output_npy", default=None,
                        help="if set together with --input_npy, save predictions there")
    args = parser.parse_args()

    model, cfg, sd = load_model(args.ckpt, args.device)
    print(f"Loaded {args.ckpt}")
    print(f"  trained on {cfg['dataset']}, H={cfg['horizon']}")
    if "test_mse" in sd:
        print(f"  ckpt-recorded test MSE={sd['test_mse']:.4f} MAE={sd['test_mae']:.4f}")

    # Custom-input forecast
    if args.input_npy is not None:
        x_np = np.load(args.input_npy)
        if x_np.ndim != 3:
            raise ValueError(f"--input_npy must be 3D (B,M,L); got shape {x_np.shape}")
        if x_np.shape[1] != cfg["num_vars"] or x_np.shape[2] != cfg["lookback"]:
            raise ValueError(
                f"input shape {x_np.shape} doesn't match checkpoint "
                f"(num_vars={cfg['num_vars']}, lookback={cfg['lookback']})"
            )
        x = torch.from_numpy(x_np.astype(np.float32)).to(args.device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        print(f"Forecast shape: {pred.shape}  (B, M, F={cfg['horizon']})")
        if args.output_npy:
            np.save(args.output_npy, pred)
            print(f"Saved predictions -> {args.output_npy}")
        return

    # Default: evaluate on test split using same loader settings as training
    _, _, test_loader, _ = make_loaders(
        root_path=args.data_root,
        dataset_name=cfg["dataset"],
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
        batch_size=cfg["batch_size"],
        num_workers=0,
    )
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    total_mse, total_mae, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            pred = model(x)
            n = y.numel()
            total_mse += mse_fn(pred, y).item() * n
            total_mae += mae_fn(pred, y).item() * n
            total_n += n
    print(f"\nTest MSE: {total_mse / total_n:.4f}")
    print(f"Test MAE: {total_mae / total_n:.4f}")


if __name__ == "__main__":
    main()
