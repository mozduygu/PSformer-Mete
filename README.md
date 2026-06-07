# PSformer Reproduction

An **independent** PyTorch reproduction of PSformer. The original authors did
**not** release code, so this repository is a from-scratch implementation based
solely on the paper.

## Paper Reference

> **PSformer: Parameter-efficient Transformer with Segment Attention for Time
> Series Forecasting**
> Yanlong Wang, Jian Xu, Fei Ma, Shao-Lun Huang, Danny Dongning Sun, Xiao-Ping Zhang.
> arXiv:2411.01419v2 (Feb 2025).
> Link: https://arxiv.org/abs/2411.01419

## Method Summary

PSformer is a transformer for **multivariate long-term time-series forecasting**.
Given a lookback window `X ∈ R^{M×L}` (M variables, L time steps) it predicts the
next `F` steps. Two ideas drive its parameter efficiency:

- **Parameter-Shared Block (PS Block).** An FFN-like block of three `N×N` linear
  layers with residual connections, `(GeLU(X·W1)·W2 + X)·W3`. The *same* block
  `W^S` is reused at three sites inside each encoder, sharply reducing parameters.
- **Spatial-Temporal Segment Attention (SegAtt).** The input is split into `N`
  non-overlapping patches of size `P` (`L = P·N`); patches at the *same position*
  across all `M` variables are merged into a **segment** of length `C = M·P`,
  giving `X ∈ R^{C×N}`. Attention is applied across the `C` dimension, so a single
  mechanism captures both cross-channel and local temporal structure. Each encoder
  is `(Attn2(ReLU(Attn1(X))) + X)·W^S` with `Q=K=V=PSBlock(X)`.

RevIN normalization wraps the model input/output, and training uses the SAM
(Sharpness-Aware Minimization) optimizer. No positional encoding is used.

## Reproduced Results

**Comparison target: paper Table 12** (test MSE/MAE, lookback `L=512`).
Numbers below are from this implementation's trained checkpoints
(`checkpoints/psformer_ETTh1_H*.pt`, seed 1).

| Dataset | Horizon | Paper MSE (Table 12) | This impl. MSE | Δ MSE | Status |
|---|---|---|---|---|---|
| ETTh1 | 96  | 0.352 | 0.358 | +1.7% | Reproduced (~2%) |
| ETTh1 | 192 | 0.385 | 0.393 | +2.1% | Reproduced (~2%) |
| ETTh1 | 336 | 0.411 | 0.420 | +2.1% | Reproduced (~2%) |
| ETTh1 | 720 | 0.440 | 0.446 | +1.4% | Reproduced (~2%) |
| ETTh2 | all | 0.272 / 0.335 / 0.356 / 0.389 | — | — | Not yet reproduced |
| ETTm1 | all | 0.282 / 0.321 / 0.352 / 0.413 | — | — | Not yet reproduced |
| ETTm2 | all | 0.167 / 0.219 / 0.269 / 0.347 | — | — | Not yet reproduced |
| Weather | all | 0.149 / 0.193 / 0.245 / 0.314 | — | — | Not yet reproduced (CSV missing) |
| Electricity | all | 0.133 / 0.149 / 0.164 / 0.203 | — | — | Not yet reproduced (CSV missing) |
| Exchange | all | 0.081 / 0.179 / 0.328 / 0.842 | — | — | Not yet reproduced (CSV missing) |
| Traffic | all | 0.367 / 0.390 / 0.404 / 0.439 | — | — | Not yet reproduced (CSV missing) |

For reference, MAE on the reproduced cells: 0.387 / 0.409 / 0.426 / 0.462
(paper Table 12: 0.385 / 0.406 / 0.424 / 0.456).

So far **only ETTh1 has been trained**, and it reproduces the paper's Table 12
MSE within roughly **2%**. The remaining datasets have not been trained yet and
are marked accordingly — their numbers are **not** invented here.

**Parameter-count sanity check (paper Table 8).** The architecture's trainable
parameter counts match Table 8 **exactly** for every checked row
(ETTh1 / Weather / Traffic, all horizons). Verify with:

```bash
python -m experiments.test_param_count
```

## Code Structure

```
psformer/
  models/
    psformer.py        PSBlock, SegmentAttention, PSformerEncoder, PSformer
    revin.py           Reversible Instance Normalization (RevIN)
  data/
    dataset.py         sliding-window dataset + make_loaders()
  utils/
    sam.py             SAM optimizer wrapper
    configs.py         per-(dataset, horizon) hyperparameters from the paper
    results.py         JSONL results ledger + paper Table-12 reference values
  experiments/
    train.py           training driver (one dataset×horizon per run)
    predict.py         evaluate a checkpoint / forecast from a .npy window
    report.py          render a Table-12-shaped comparison from the ledger
    backfill_ledger.py rebuild the ledger from existing checkpoints/logs
    test_param_count.py parameter-count sanity check vs Table 8
  scripts/
    download_data.sh   dataset download helper (ETT auto; others manual)
  datasets/            input CSVs (ETTh1/h2/m1/m2 present; others must be added)
  checkpoints/         trained model checkpoints
  notes/               implementation audit and reproduction notes
  papers/              the PSformer paper PDF
```

## Installation

Requires **Python 3.10+** (the code uses `X | None` type-hint syntax). The only
third-party dependencies are `torch`, `numpy`, and `pandas` (see
`requirements.txt`).

### Option A — one-step setup script (recommended)

This creates a virtual environment named `psformer-env` and installs everything
into it:

```bash
bash scripts/setup_env.sh
```

Then **activate** the environment (do this in every new shell before running):

```bash
source psformer-env/bin/activate
```

Verify it works, and run anything from the "Running" section below:

```bash
python -m experiments.test_param_count
```

When you are done, deactivate with `deactivate`.

### Option B — manual

```bash
python -m venv psformer-env
source psformer-env/bin/activate
pip install -r requirements.txt        # or: pip install torch numpy pandas
```

A CUDA-capable GPU is used automatically if available; otherwise the code falls
back to CPU. The dataset download helper (`scripts/download_data.sh`) additionally
uses the system tools `bash` and `curl`.

This dependency set is sufficient to run the full pipeline — parameter-count
check, training, evaluation, custom forecasting, and reporting (all verified
end-to-end inside `psformer-env`).

## Running

**1. Parameter-count sanity check (no data needed):**

```bash
python -m experiments.test_param_count
```

**2. (Optional) Download datasets.** The ETT CSVs are already in `datasets/`.
To fetch them fresh, or to get instructions for the larger datasets:

```bash
bash scripts/download_data.sh ./datasets
```

The four ETT files download automatically; `weather.csv`, `electricity.csv`,
`exchange_rate.csv`, and `traffic.csv` must be obtained manually (the script
prints the links).

**3. Train ETTh1** (all hyperparameters are looked up automatically from
`utils/configs.py`):

```bash
python -m experiments.train --dataset ETTh1 --horizon 96 \
    --data_root ./datasets --ckpt_dir ./checkpoints
```

Repeat with `--horizon 192 / 336 / 720`. Other datasets use the same command
(e.g. `--dataset ETTm1 --horizon 96`) once their CSV is present.

**4. Generate a markdown comparison report** from the results ledger:

```bash
python -m experiments.report --markdown
```

**5. Evaluate a saved checkpoint:**

```bash
python -m experiments.predict --ckpt ./checkpoints/psformer_ETTh1_H96.pt
```

## Current Limitations

- **Only ETTh1 has been trained so far.** It reproduces paper Table 12 MSE within
  ~2%. ETTh2 / ETTm1 / ETTm2 are trainable now (data present) but have not been
  run; Weather / Electricity / Exchange / Traffic additionally require downloading
  their CSVs first.
- **Full reproduction requires downloading the remaining datasets** and training
  all 32 (dataset × horizon) cells.
- This is an **independent reproduction** with no official reference code. Results
  should be interpreted as such, and small deviations from the paper are expected
  (see `notes/implementation_audit.md` for known gaps and a roadmap).
