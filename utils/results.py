"""
Results ledger for PSformer reproduction experiments.

Maintains a single JSON file (default: results/runs.jsonl) where every training
run appends one line. This is the canonical record for reporting -- it captures
every configuration that produced every metric, plus environment context.

JSONL format (one JSON object per line) is chosen so:
  - Concurrent runs can append without corrupting prior records
  - Git diffs are clean (one run per line)
  - Tools like `jq` and pandas read it trivially

Example record:
    {
      "timestamp": "2026-04-27T22:30:15",
      "dataset": "ETTh1",
      "horizon": 96,
      "seed": 1,
      "test_mse": 0.3579,
      "test_mae": 0.3872,
      "best_val_mse": 0.6797,
      "best_epoch": 41,
      "stopped_epoch": 71,
      "wall_time_sec": 220.3,
      "config": {...},
      "paper_target_mse": 0.352,
      "paper_target_mae": 0.385,
      "rel_diff_mse_pct": 1.7,
      "rel_diff_mae_pct": 0.5,
      "env": {"torch": "2.1.0+cu124", "device": "NVIDIA GeForce RTX 4090",
              "git_sha": "abc1234", "python": "3.12.x"}
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


# Paper Table 12 reference values for delta computation.
PAPER_TABLE_12 = {
    ("ETTh1", 96):  (0.352, 0.385), ("ETTh1", 192): (0.385, 0.406),
    ("ETTh1", 336): (0.411, 0.424), ("ETTh1", 720): (0.440, 0.456),
    ("ETTh2", 96):  (0.272, 0.337), ("ETTh2", 192): (0.335, 0.379),
    ("ETTh2", 336): (0.356, 0.411), ("ETTh2", 720): (0.389, 0.431),
    ("ETTm1", 96):  (0.282, 0.336), ("ETTm1", 192): (0.321, 0.360),
    ("ETTm1", 336): (0.352, 0.380), ("ETTm1", 720): (0.413, 0.412),
    ("ETTm2", 96):  (0.167, 0.258), ("ETTm2", 192): (0.219, 0.292),
    ("ETTm2", 336): (0.269, 0.325), ("ETTm2", 720): (0.347, 0.376),
    ("Weather", 96):  (0.149, 0.200), ("Weather", 192): (0.193, 0.243),
    ("Weather", 336): (0.245, 0.282), ("Weather", 720): (0.314, 0.332),
    ("Electricity", 96):  (0.133, 0.229), ("Electricity", 192): (0.149, 0.242),
    ("Electricity", 336): (0.164, 0.258), ("Electricity", 720): (0.203, 0.291),
    ("Exchange", 96):  (0.081, 0.197), ("Exchange", 192): (0.179, 0.299),
    ("Exchange", 336): (0.328, 0.412), ("Exchange", 720): (0.842, 0.689),
    ("Traffic", 96):  (0.367, 0.257), ("Traffic", 192): (0.390, 0.272),
    ("Traffic", 336): (0.404, 0.274), ("Traffic", 720): (0.439, 0.294),
}


def _git_sha() -> str:
    """Best-effort git short SHA; returns 'no-git' if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "no-git"


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def append_run(
    ledger_path: str,
    *,
    dataset: str,
    horizon: int,
    seed: int,
    test_mse: float,
    test_mae: float,
    best_val_mse: float,
    best_epoch: int,
    stopped_epoch: int,
    wall_time_sec: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append one record to the ledger and return it."""
    paper_mse, paper_mae = PAPER_TABLE_12.get((dataset, horizon), (None, None))
    record: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "test_mse": round(float(test_mse), 6),
        "test_mae": round(float(test_mae), 6),
        "best_val_mse": round(float(best_val_mse), 6),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "wall_time_sec": round(float(wall_time_sec), 1),
        "paper_target_mse": paper_mse,
        "paper_target_mae": paper_mae,
        "rel_diff_mse_pct": (
            round(100.0 * (test_mse - paper_mse) / paper_mse, 2)
            if paper_mse else None
        ),
        "rel_diff_mae_pct": (
            round(100.0 * (test_mae - paper_mae) / paper_mae, 2)
            if paper_mae else None
        ),
        "config": config,
        "env": _env_info(),
    }

    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_runs(ledger_path: str) -> list[dict[str, Any]]:
    """Read all records from the ledger."""
    if not os.path.exists(ledger_path):
        return []
    runs = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs
