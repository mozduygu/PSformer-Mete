"""
Train PSformer on a single (dataset, horizon) configuration.

Usage:
    python -m experiments.train --dataset ETTh1 --horizon 96 \
        --data_root ./datasets --ckpt_dir ./checkpoints

This script reproduces one cell of paper Table 12. It uses the exact
hyperparameters from Appendix A.3 + Table 11 (looked up via utils.configs).

Optional flags:
    --epochs N          override the 300-epoch default (e.g., for quick tests)
    --num_workers K     dataloader workers (default 4)
    --device cuda|cpu
    --no_sam            disable SAM (use plain Adam) -- for debugging
    --log_every N       print metrics every N batches
    --eval_only PATH    skip training, just evaluate a checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Make 'psformer' package importable when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from data import make_loaders
from models import PSformer
from utils import SAM, get_config, append_run


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader, device, criterion_mse, criterion_mae):
    """Compute average MSE and MAE on a loader (no grad)."""
    model.eval()
    total_mse, total_mae, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            n = y.numel()                                # number of scalars
            total_mse += criterion_mse(pred, y).item() * n
            total_mae += criterion_mae(pred, y).item() * n
            total_n += n
    return total_mse / total_n, total_mae / total_n


def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_sam: bool, log_every: int = 0, epoch: int = 0):
    model.train()
    total_loss, total_n = 0.0, 0
    t0 = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_sam:
            # First forward+backward
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss1 = criterion(pred, y)
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward+backward at perturbed weights
            pred = model(x)
            loss2 = criterion(pred, y)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            loss_value = loss1.item()  # report unperturbed loss
        else:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

        n = y.numel()
        total_loss += loss_value * n
        total_n += n

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [epoch {epoch}] batch {batch_idx+1}/{len(loader)} "
                  f"loss={loss_value:.5f} elapsed={elapsed:.1f}s")

    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2",
                                 "Weather", "Electricity", "Exchange", "Traffic"])
    parser.add_argument("--horizon", type=int, required=True,
                        choices=[96, 192, 336, 720])
    parser.add_argument("--data_root", default="./datasets")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override default 300 epochs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_sam", action="store_true",
                        help="disable SAM (use plain Adam) -- debugging only")
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--eval_only", default=None,
                        help="path to checkpoint; skip training and report test metrics")
    parser.add_argument("--ledger", default="./results/runs.jsonl",
                        help="JSONL file to append run results to")
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.horizon)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    print("=" * 72)
    print(f"PSformer training:  dataset={cfg['dataset']}  H={cfg['horizon']}")
    for k in ("num_encoders", "num_vars", "batch_size", "lr", "epochs", "patience",
              "seed", "sam_rho", "revin_stat_window", "revin_affine"):
        print(f"  {k}: {cfg[k]}")
    print("=" * 72)

    set_seed(cfg["seed"])

    # Data
    train_loader, val_loader, test_loader, M = make_loaders(
        root_path=args.data_root,
        dataset_name=cfg["dataset"],
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
        batch_size=cfg["batch_size"],
        num_workers=args.num_workers,
    )
    assert M == cfg["num_vars"], (M, cfg["num_vars"])
    print(f"Loaders ready -- train={len(train_loader)} val={len(val_loader)} "
          f"test={len(test_loader)} batches")

    # Model
    model = PSformer(
        num_vars=cfg["num_vars"],
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
        segments=cfg["segments"],
        num_encoders=cfg["num_encoders"],
        revin_stat_window=cfg["revin_stat_window"],
        revin_affine=cfg["revin_affine"],
    ).to(args.device)
    print(f"Model parameters: full={model.num_parameters():,} "
          f"encoder-only={model.num_parameters(encoder_only=True):,}")

    # Optimizer
    use_sam = (not args.no_sam) and (cfg["sam_rho"] > 0.0)
    if use_sam:
        optimizer = SAM(model.parameters(), torch.optim.Adam,
                        rho=cfg["sam_rho"], lr=cfg["lr"])
        print(f"Using SAM (rho={cfg['sam_rho']}) wrapping Adam (lr={cfg['lr']})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        print(f"Using plain Adam (lr={cfg['lr']})  -- no SAM")

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Eval-only path
    if args.eval_only is not None:
        sd = torch.load(args.eval_only, map_location=args.device)
        model.load_state_dict(sd["model_state_dict"])
        test_mse, test_mae = evaluate(model, test_loader, args.device,
                                      criterion_mse, criterion_mae)
        print(f"\n[eval_only] test MSE: {test_mse:.4f}  test MAE: {test_mae:.4f}")
        return

    # Training loop with early stopping
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir,
                             f"psformer_{cfg['dataset']}_H{cfg['horizon']}.pt")
    best_val_mse = float("inf")
    epochs_since_best = 0
    best_test_mse, best_test_mae = float("inf"), float("inf")
    best_epoch = 0
    last_epoch = 0
    train_t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        last_epoch = epoch
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_mse,
                                     args.device, use_sam, args.log_every, epoch)
        val_mse, val_mae = evaluate(model, val_loader, args.device,
                                    criterion_mse, criterion_mae)
        elapsed = time.time() - t0

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_since_best = 0
            # Evaluate test only when we have a new best val, to save compute
            test_mse, test_mae = evaluate(model, test_loader, args.device,
                                          criterion_mse, criterion_mae)
            best_test_mse, best_test_mae = test_mse, test_mae
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
            }, ckpt_path)
            tag = "  (new best -- saved)"
        else:
            epochs_since_best += 1
            tag = f"  (no improve {epochs_since_best}/{cfg['patience']})"

        print(f"epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={train_loss:.5f}  "
              f"val_mse={val_mse:.5f}  val_mae={val_mae:.5f}  "
              f"({elapsed:.1f}s){tag}")

        if epochs_since_best >= cfg["patience"]:
            print(f"\nEarly stopping triggered after epoch {epoch} "
                  f"(no improvement for {cfg['patience']} epochs).")
            break

    wall_time = time.time() - train_t0

    # Append to results ledger
    record = append_run(
        args.ledger,
        dataset=cfg["dataset"],
        horizon=cfg["horizon"],
        seed=cfg["seed"],
        test_mse=best_test_mse,
        test_mae=best_test_mae,
        best_val_mse=best_val_mse,
        best_epoch=best_epoch,
        stopped_epoch=last_epoch,
        wall_time_sec=wall_time,
        config=cfg,
    )

    print("\n" + "=" * 72)
    print(f"DONE.  Best val MSE: {best_val_mse:.4f}")
    print(f"Test MSE at best val: {best_test_mse:.4f}  "
          f"(paper: {record.get('paper_target_mse')}, "
          f"delta: {record.get('rel_diff_mse_pct')}%)")
    print(f"Test MAE at best val: {best_test_mae:.4f}  "
          f"(paper: {record.get('paper_target_mae')}, "
          f"delta: {record.get('rel_diff_mae_pct')}%)")
    print(f"Best epoch: {best_epoch}/{last_epoch}  Wall time: {wall_time:.1f}s")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Ledger:     {args.ledger}")
    print("=" * 72)


if __name__ == "__main__":
    main()
