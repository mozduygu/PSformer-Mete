"""
Generate a Table-12-shaped reproduction report from the runs ledger.

Usage:
    python -m experiments.report                          # default ledger
    python -m experiments.report --ledger results/runs.jsonl
    python -m experiments.report --markdown               # markdown table

If multiple runs exist for a (dataset, horizon, seed) tuple, we use the most
recent. If multiple seeds exist for a (dataset, horizon), we report mean ± std.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils import load_runs, PAPER_TABLE_12


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2",
            "Weather", "Electricity", "Exchange", "Traffic"]
HORIZONS = [96, 192, 336, 720]


def _agg(records):
    """Average across seeds for one (dataset, horizon) cell."""
    if not records:
        return None
    # Most recent timestamp wins for each unique seed
    by_seed: dict[int, dict] = {}
    for r in records:
        s = r.get("seed", 1)
        if s not in by_seed or r["timestamp"] > by_seed[s]["timestamp"]:
            by_seed[s] = r
    runs = list(by_seed.values())
    mse = [r["test_mse"] for r in runs]
    mae = [r["test_mae"] for r in runs]
    n = len(runs)
    return {
        "n_seeds": n,
        "mse_mean": statistics.mean(mse),
        "mse_std":  statistics.stdev(mse) if n > 1 else 0.0,
        "mae_mean": statistics.mean(mae),
        "mae_std":  statistics.stdev(mae) if n > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="./results/runs.jsonl")
    parser.add_argument("--markdown", action="store_true",
                        help="output a markdown table instead of plain text")
    args = parser.parse_args()

    runs = load_runs(args.ledger)
    if not runs:
        print(f"No runs found at {args.ledger}.")
        return

    # Index runs by (dataset, horizon)
    bucket: dict[tuple, list] = {}
    for r in runs:
        bucket.setdefault((r["dataset"], r["horizon"]), []).append(r)

    if args.markdown:
        _print_markdown(bucket)
    else:
        _print_plain(bucket)

    # Average row at the bottom
    print()
    print("Aggregate across all completed cells:")
    deltas_mse, deltas_mae = [], []
    for key, records in bucket.items():
        if key not in PAPER_TABLE_12:
            continue
        agg = _agg(records)
        paper_mse, paper_mae = PAPER_TABLE_12[key]
        deltas_mse.append(100 * (agg["mse_mean"] - paper_mse) / paper_mse)
        deltas_mae.append(100 * (agg["mae_mean"] - paper_mae) / paper_mae)
    if deltas_mse:
        print(f"  mean MSE delta vs paper: {statistics.mean(deltas_mse):+.2f}%")
        print(f"  mean MAE delta vs paper: {statistics.mean(deltas_mae):+.2f}%")
        print(f"  cells reproduced:        {len(deltas_mse)}/32")


def _print_plain(bucket):
    print(f"\n{'Dataset':<12} {'H':>4}   {'Paper MSE':>10} {'Our MSE':>9} "
          f"{'ΔMSE%':>7}   {'Paper MAE':>10} {'Our MAE':>9} {'ΔMAE%':>7}   {'#seeds':>7}")
    print("-" * 92)
    for ds in DATASETS:
        for h in HORIZONS:
            key = (ds, h)
            recs = bucket.get(key, [])
            if not recs:
                continue
            agg = _agg(recs)
            paper_mse, paper_mae = PAPER_TABLE_12[key]
            d_mse = 100 * (agg["mse_mean"] - paper_mse) / paper_mse
            d_mae = 100 * (agg["mae_mean"] - paper_mae) / paper_mae
            seed_tag = f"{agg['n_seeds']}" + (
                f" (±{agg['mse_std']:.4f})" if agg['n_seeds'] > 1 else ""
            )
            print(f"{ds:<12} {h:>4}   "
                  f"{paper_mse:>10.3f} {agg['mse_mean']:>9.4f} {d_mse:>+6.2f}%   "
                  f"{paper_mae:>10.3f} {agg['mae_mean']:>9.4f} {d_mae:>+6.2f}%   "
                  f"{seed_tag:>7}")


def _print_markdown(bucket):
    print()
    print("| Dataset | H | Paper MSE | Our MSE | Δ MSE | Paper MAE | Our MAE | Δ MAE |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for h in HORIZONS:
            key = (ds, h)
            recs = bucket.get(key, [])
            if not recs:
                continue
            agg = _agg(recs)
            paper_mse, paper_mae = PAPER_TABLE_12[key]
            d_mse = 100 * (agg["mse_mean"] - paper_mse) / paper_mse
            d_mae = 100 * (agg["mae_mean"] - paper_mae) / paper_mae
            print(f"| {ds} | {h} | {paper_mse:.3f} | **{agg['mse_mean']:.4f}** "
                  f"| {d_mse:+.2f}% | {paper_mae:.3f} | **{agg['mae_mean']:.4f}** "
                  f"| {d_mae:+.2f}% |")


if __name__ == "__main__":
    main()
