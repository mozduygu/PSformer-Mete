# CLAUDE.md

Guidance for future Claude Code sessions working in this repository.

## Project Goal

This repository implements and reproduces selected results from a scientific
paper (**PSformer**, arXiv:2411.01419v2 — multivariate long-term time-series
forecasting). There is **no official code**; this is an independent
reproduction. The goal is to improve the current implementation and make the
repository clear, runnable, and reproducible.

Useful context already written:
- `notes/implementation_audit.md` — full paper-vs-code audit, mismatch table,
  known bugs, and a phased improvement roadmap. **Read this first.**
- `papers/PSformer_*.pdf` — the paper (Table 12 = target results; Table 8 =
  param counts; App. A.3 + Table 11 = hyperparameters).

Current status (see audit for detail): architecture is faithful (Table-8 param
counts match exactly); **only ETTh1 is trained** so far, reproducing within ~2%
MSE. The other 7 datasets are not yet run, and 4 of 8 dataset CSVs are missing.

## README Requirements

`README.md` must always contain:
- paper reference **and link**,
- brief paper/method description,
- the result obtained by *this* implementation **alongside** the original paper
  table/plot/figure being compared (name the table — e.g. "Table 12"),
- code structure,
- install and run instructions.

Update `README.md` whenever results, commands, or code structure change.

## Development Rules

- Do **not** rewrite the whole repository.
- Preserve the current implementation unless a change is necessary.
- Make small, focused changes.
- Do **not** overclaim results. If the obtained result differs from the paper,
  report it honestly (state the % delta and which paper table it's compared to).
- Keep installation and run commands up to date with the actual scripts.

## Result Tracking

- Save obtained results under `results/` when possible. The training driver
  already appends to `results/runs.jsonl` via `utils/results.py`; render with
  `python -m experiments.report --markdown`.
- Keep the comparison with the original paper result explicit and clear.
- Always mention which paper table/plot/figure is being compared (results →
  Table 12; param counts → Table 8).

## Safe Editing

- Before modifying code, explain what will be changed and why.
- After modifying code, summarize the change.
- Run the smallest relevant check after changes, e.g.:
  - `python -m experiments.test_param_count` (architecture sanity vs Table 8),
  - a short training smoke test:
    `python -m experiments.train --dataset ETTh1 --horizon 96 --epochs 2`.

## Quick Reference

- Train one cell: `python -m experiments.train --dataset <D> --horizon <H>`
  (config auto-looked-up from `utils/configs.py`).
- Datasets present: ETTh1/ETTh2/ETTm1/ETTm2. Missing:
  weather/electricity/exchange/traffic (`scripts/download_data.sh` helps).
- Hyperparameters live in `utils/configs.py` (do not hardcode elsewhere).
