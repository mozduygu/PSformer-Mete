"""
Backfill the runs ledger from existing checkpoints + log files.

If you trained before the ledger system existed, this script recovers what it
can from each .pt checkpoint and (optionally) the matching log file. Best/last
epoch and wall-time are best-effort -- they're parsed from the log if present,
otherwise filled with -1.

Usage:
    python -m experiments.backfill_ledger \
        --ckpt_dir ./checkpoints --log_dir . --ledger ./results/runs.jsonl
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils import append_run


CKPT_RE = re.compile(r"psformer_(?P<ds>[A-Za-z]+)_H(?P<h>\d+)\.pt$")
EARLY_RE = re.compile(r"Early stopping triggered after epoch (\d+)")
EPOCH_RE = re.compile(r"epoch\s+(\d+)/\d+\s+train_loss=.*?\(([\d.]+)s\)")
BEST_VAL_RE = re.compile(r"DONE\.\s+Best val MSE:\s*([\d.]+)")


def parse_log(path: str) -> dict:
    """Best-effort extraction of stopped_epoch, total_time, best_epoch from a log."""
    if not path or not os.path.exists(path):
        return {"stopped_epoch": -1, "wall_time_sec": -1.0, "best_epoch": -1}
    text = open(path).read()
    m = EARLY_RE.search(text)
    stopped = int(m.group(1)) if m else -1

    # Total wall time = sum of per-epoch elapsed seconds
    total = 0.0
    last_best_epoch = -1
    for em in EPOCH_RE.finditer(text):
        ep = int(em.group(1))
        total += float(em.group(2))
        # Capture epoch -> "(new best -- saved)" markers
    for line in text.splitlines():
        if "(new best -- saved)" in line:
            mm = re.match(r"epoch\s+(\d+)/", line)
            if mm:
                last_best_epoch = int(mm.group(1))

    return {
        "stopped_epoch": stopped if stopped > 0 else -1,
        "wall_time_sec": round(total, 1) if total > 0 else -1.0,
        "best_epoch": last_best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default=".",
                        help="directory containing logs_<DATASET>_H<H>.log files")
    parser.add_argument("--ledger", default="./results/runs.jsonl")
    args = parser.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "psformer_*.pt")))
    if not ckpts:
        print(f"No checkpoints in {args.ckpt_dir}.")
        return

    print(f"Found {len(ckpts)} checkpoints. Backfilling ledger -> {args.ledger}\n")
    for ckpt_path in ckpts:
        m = CKPT_RE.search(os.path.basename(ckpt_path))
        if not m:
            print(f"  skip (unparseable name): {ckpt_path}")
            continue
        ds, h = m["ds"], int(m["h"])
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = sd["config"]

        log_path = os.path.join(args.log_dir, f"logs_{ds}_H{h}.log")
        log_info = parse_log(log_path)

        record = append_run(
            args.ledger,
            dataset=cfg["dataset"],
            horizon=cfg["horizon"],
            seed=cfg["seed"],
            test_mse=sd["test_mse"],
            test_mae=sd["test_mae"],
            best_val_mse=sd["val_mse"],
            best_epoch=sd.get("epoch", log_info["best_epoch"]),
            stopped_epoch=log_info["stopped_epoch"],
            wall_time_sec=log_info["wall_time_sec"],
            config=cfg,
        )
        print(f"  + {ds:10s} H={h:3d}  test MSE={record['test_mse']:.4f}  "
              f"MAE={record['test_mae']:.4f}  "
              f"ΔMSE={record['rel_diff_mse_pct']}%")

    print(f"\nDone. Run `python -m experiments.report` to view the ledger.")


if __name__ == "__main__":
    main()
