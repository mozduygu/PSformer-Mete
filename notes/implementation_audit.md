# PSformer Implementation Audit

Audit date: 2026-06-07
Paper: *PSformer: Parameter-efficient Transformer with Segment Attention for Time
Series Forecasting* — Wang et al., arXiv:2411.01419v2 (Feb 2025).
Repo: independent reproduction (no official code released).

---

## 1. Paper objective and target results

**Objective.** Long-term *multivariate* time-series forecasting with a
parameter-efficient transformer. Two core ideas:

1. **Parameter Sharing (PS Block).** Three FC layers `W1, W2, W3 ∈ R^{N×N}`
   with residual connections (an FFN-like block, Eq. 1–3). The *same* PS Block
   `W^S = {W1, W2, W3}` is reused at three sites inside one encoder (two segment
   attentions + final fusion), drastically reducing parameter count.
2. **Spatial-Temporal Segment Attention (SegAtt).** Patches at the *same
   position* across all `M` variables are merged into a "segment" of length
   `C = M·P`. Attention is applied across the `C` dimension so it captures both
   cross-channel and local temporal structure.

**Pipeline (Fig. 1/2).** `X ∈ R^{M×L}` → RevIN(norm) → patch+cross-channel merge
→ `R^{C×N}` (`C=M·P`, `N=L/P`) → `n×` PSformer Encoder → inverse merge → `R^{M×L}`
→ Linear `L→F` (channel-independent) → RevIN(denorm) → `R^{M×F}`. No positional
encoding (§A.6).

**Encoder (Eq. 8).** `X_out = ( Attn2( ReLU( Attn1(X_in) ) ) + X_in ) · W^S`,
where `Attn(X) = Softmax( (X W^S)(X W^S)^T / √d_k ) (X W^S)`, i.e. `Q=K=V=PSBlock(X)`,
`d_k = N`.

**Training recipe (App. A.3, Table 11).** `L=512`, `N=32`, MSE loss, Adam,
LR `1e-4` constant, **SAM** optimizer (per-dataset `ρ` in Table 11), batch 16
(Traffic 8), 300 epochs, patience 30, **seed 1**. Encoders: 1 for
ETTh1/ETTh2/ETTm2/Exchange, 3 for ETTm1/Weather/Electricity/Traffic. Exchange
uses RevIN stat window 16 (§B.7). `drop_last=False` for the test loader (§A.2).

**Target results (Table 12, MSE/MAE; the reproduction goal).**

| Dataset | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| ETTh1 | 0.352/0.385 | 0.385/0.406 | 0.411/0.424 | 0.440/0.456 |
| ETTh2 | 0.272/0.337 | 0.335/0.379 | 0.356/0.411 | 0.389/0.431 |
| ETTm1 | 0.282/0.336 | 0.321/0.360 | 0.352/0.380 | 0.413/0.412 |
| ETTm2 | 0.167/0.258 | 0.219/0.292 | 0.269/0.325 | 0.347/0.376 |
| Weather | 0.149/0.200 | 0.193/0.243 | 0.245/0.282 | 0.314/0.332 |
| Electricity | 0.133/0.229 | 0.149/0.242 | 0.164/0.258 | 0.203/0.291 |
| Exchange | 0.081/0.197 | 0.179/0.299 | 0.328/0.412 | 0.842/0.689 |
| Traffic | 0.367/0.257 | 0.390/0.272 | 0.404/0.274 | 0.439/0.294 |

Headline claim: best MSE on 7/8 datasets, with parameter counts ~1/1000 of
TSMixer/ModernTCN (Table 8).

---

## 2. What the current repo already implements

- **Full model** (`models/psformer.py`): `PSBlock`, `SegmentAttention`,
  `PSformerEncoder`, `PSformer` — faithful to Eq. 1–8 and Fig. 1/2.
- **RevIN** (`models/revin.py`) with optional affine and optional stat window.
- **Datasets/loaders** (`data/dataset.py`): sliding-window dataset for all 8
  benchmarks, train-only z-score scaling, ETT fixed borders + 70/10/20 ratio
  borders, `drop_last=False` test loader.
- **SAM optimizer** (`utils/sam.py`): Foret-style two-step SAM wrapping Adam.
- **Config system** (`utils/configs.py`): per-(dataset, horizon) hyperparameters
  transcribed from App. A.3 + Table 11.
- **Training driver** (`experiments/train.py`): SAM/plain loop, early stopping,
  best-on-val checkpointing, ledger append, eval-only mode.
- **Inference** (`experiments/predict.py`): test-set eval + custom `.npy` forecast.
- **Results infra**: `utils/results.py` (JSONL ledger with paper deltas),
  `experiments/report.py` (Table-12-shaped report), `experiments/backfill_ledger.py`.
- **Param-count test** (`experiments/test_param_count.py`) — **runs and passes**:
  every Table-8 row matches exactly (verified this session, torch 2.6.0+cu124).
- **Data download helper** (`scripts/download_data.sh`).

**Reproduction status so far** (only ETTh1 has been trained; checkpoints +
logs present):

| Cfg | Our MSE/MAE | Paper MSE/MAE | ΔMSE | ΔMAE |
|---|---|---|---|---|
| ETTh1-96 | 0.3579 / 0.3872 | 0.352 / 0.385 | +1.7% | +0.6% |
| ETTh1-192 | 0.3931 / 0.4087 | 0.385 / 0.406 | +2.1% | +0.7% |
| ETTh1-336 | 0.4196 / 0.4260 | 0.411 / 0.424 | +2.1% | +0.5% |
| ETTh1-720 | 0.4460 / 0.4617 | 0.440 / 0.456 | +1.4% | +1.2% |

ETTh1 reproduces within ~2% MSE / ~1% MAE — close. The remaining 7 datasets are
not yet run. (The large val MSE of ~0.9 visible in logs is **expected** — it
matches the paper's own Figure 4 val curves for ETTh1; not a bug.)

---

## 3. Current repository structure

```
psformer/
  models/
    psformer.py        PSBlock, SegmentAttention, PSformerEncoder, PSformer
    revin.py           Reversible Instance Normalization
    __init__.py
  data/
    dataset.py         TSForecastDataset + make_loaders()
    __init__.py
  utils/
    sam.py             SAM optimizer wrapper
    configs.py         per-(dataset,horizon) hyperparameters
    results.py         JSONL ledger + paper Table-12 reference
    __init__.py
  experiments/
    train.py           training driver
    predict.py         eval / inference
    report.py          Table-12-shaped report from ledger
    backfill_ledger.py recover ledger from checkpoints+logs
    test_param_count.py param-count sanity vs Table 8
    __init__.py
  scripts/
    download_data.sh   dataset fetch helper (ETT auto; rest manual)
  datasets/            ETTh1/h2/m1/m2 CSV  (weather/electricity/exchange/traffic MISSING)
  checkpoints/         psformer_ETTh1_H{96,192,336,720}.pt
  logs_ETTh1_H{192,336,720}.log
  papers/<paper>.pdf
  README.md
  (no results/, no requirements.txt, no tests/, no .gitignore)
```

---

## 4. Main entry points / scripts

| Command | Purpose |
|---|---|
| `python -m experiments.train --dataset D --horizon H` | Train one cell of Table 12 (auto-looks-up config) |
| `python -m experiments.predict --ckpt CKPT` | Re-evaluate checkpoint on test, or forecast from `.npy` |
| `python -m experiments.test_param_count` | Verify param counts vs Table 8 (passes) |
| `python -m experiments.report [--markdown]` | Render reproduction table from ledger |
| `python -m experiments.backfill_ledger` | Rebuild ledger from existing checkpoints/logs |
| `bash scripts/download_data.sh ./datasets` | Fetch ETT CSVs; prints manual steps for the rest |

There is **no single sweep driver** that runs all 32 (dataset×horizon) cells.

---

## 5. Model architecture currently implemented

`models/psformer.py`:

- **PSBlock** (`forward`): `out = W3( W2(GeLU(W1 x)) + x )`. Three
  `nn.Linear(N, N, bias=True)`. Operates on the last dim `N`; identity on
  leading dims. Bias inferred from Table-8 counts (3·(N²+N)=3168 for N=32). ✓
- **SegmentAttention**: `qkv = ps_block(x)` (B,C,N); `scores = qkv·qkvᵀ·(1/√N)`
  → softmax over keys (dim=-1) → `out = attn·qkv`. Single head, no extra
  projection. Scale uses `d_k = N`. ✓
- **PSformerEncoder**: one shared `ps_block`; `a1=attn1(x,ps); a1=ReLU(a1);
  a2=attn2(a1,ps); out=ps(a2+x)`. Matches Eq. 8 (the trailing `·W^S` is the
  *full* PS Block). ✓
- **PSformer**: RevIN(norm) → `_patch` (B,M,L)→(B,C,N) → encoder stack →
  `_unpatch` → `head = Linear(L,F)` applied per variable → RevIN(denorm).
  `_patch`/`_unpatch` are exact inverses; time order within L is preserved, so
  the channel-independent head sees chronologically-ordered lookback. ✓

**Verified faithful**: patch/merge ordering, `d_k=N` scaling, full-PSBlock fusion,
channel-independent `L→F` head, no positional encoding, GeLU (block) + ReLU
(between attentions). Param counts match Table 8 exactly for ETTh1/Weather/Traffic.

---

## 6. Dataset / preprocessing currently implemented

`data/dataset.py`:

- Reads CSV, drops `date` column, keeps all `M` channels (multivariate).
- **ETT borders**: fixed Informer/Autoformer months (train 0–8640, val/test in
  4-month blocks). **Others**: 70/10/20 ratio split.
- **Scaling**: z-score fit on **train slice only**, applied to all splits. ✓
- **Windowing**: sliding stride-1; val/test start is pulled back by `L` so the
  first val/test target is predictable (standard PatchTST convention). Window
  count = `s_end − L − H − s_start + 1`.
- `make_loaders`: train shuffle + `drop_last=True`; **val `drop_last=True`**;
  test `drop_last=False` (matches §A.2). Returns `M`.

Item shape: `x:(M,L)`, `y:(M,H)` (transposed to PSformer's `R^{M×L}` convention).

**Only ETTh1/h2/m1/m2 CSVs are present.** weather/electricity/exchange/traffic
are absent → those datasets cannot currently be trained.

---

## 7. Training loop currently implemented

`experiments/train.py`:

- `set_seed`: seeds numpy + torch (CPU+CUDA) only.
- **SAM path**: forward→`loss1.backward()`→`first_step`→forward→`loss2.backward()`
  →`second_step`. Reports unperturbed `loss1`. **Plain path** when `--no_sam` or
  `ρ=0`.
- Early stopping on **val MSE**, patience 30. Saves checkpoint on each new-best
  val; evaluates test only at new-best (test used purely for reporting, not
  selection — good practice).
- Appends a ledger record with paper deltas; prints summary.
- LR schedule: **constant** (no scheduler) — matches App. A.3. ✓

---

## 8. Loss functions currently implemented

- **Training**: `nn.MSELoss()` on RevIN-denormalized output vs target, both in
  the train-z-scored space (standard PatchTST/SAMformer convention; metrics are
  reported in the scaled space). ✓ matches paper ("Loss = MSE").
- **Eval**: MSE (`nn.MSELoss`) and MAE (`nn.L1Loss`), accumulated weighted by
  element count → exact global mean over (samples × M × H). ✓

---

## 9. Evaluation metrics currently implemented

- Test/val **MSE** and **MAE**, computed by `evaluate()` (numel-weighted mean,
  correct regardless of last-batch size).
- `predict.py` recomputes the same on test.
- `results.py` stores metrics + relative-% deltas vs Table 12; `report.py`
  aggregates across seeds (mean ± std) and prints per-cell deltas + a 32-cell
  reproduction summary.

---

## 10. Config / hyperparameter system currently implemented

`utils/configs.py` → `get_config(dataset, horizon)` returns a dict with:
`lookback=512`, `segments=32`, `num_encoders` (per dataset), `num_vars`,
`batch_size` (16, Traffic 8), `lr=1e-4`, `epochs=300`, `patience=30`, `seed=1`,
`sam_rho` (Table 11), `revin_stat_window` (Exchange=16 else None),
`revin_affine=False`. All transcribed from App. A.3 + Table 11. Spot-checked
against the paper: correct.

---

## 11. Paper vs implementation mismatch table

| Aspect | Paper | Implementation | Verdict |
|---|---|---|---|
| PS Block (Eq. 3) | `(GeLU(XW1)W2 + X)W3`, `W∈R^{N×N}` | identical, bias on | ✅ match |
| Param sharing | one `W^S` reused 3× per encoder; distinct across encoders | identical | ✅ match |
| SegAtt `Q=K=V` | shared PS-Block non-linear projection | `qkv=ps_block(x)` | ✅ match |
| Attention axis | across `C` (`QKᵀ∈R^{C×C}`) | softmax over C, contraction over N | ✅ match |
| Scale `d_k` | `N` (N↔d_model) | `1/√N` | ✅ match (per README rationale) |
| Encoder Eq. 8 | `(Attn2(ReLU(Attn1))+X)·W^S` (full PS Block) | identical | ✅ match |
| Heads | single (implied) | single head | ✅ plausible |
| Positional enc. | none | none | ✅ match |
| RevIN | in/out, no affine (per Table-8 counts) | affine=False default | ✅ match |
| Head | `W^F∈R^{L×F}`, channel-independent | `Linear(L,F)` shared over M | ✅ match |
| Loss | MSE | MSE | ✅ match |
| Optimizer | Adam + SAM (Foret), per-ds ρ | SAM(Adam), ρ from Table 11 | ✅ match |
| LR schedule | constant 1e-4 | constant, no scheduler | ✅ match |
| Encoders/ds | 1 or 3 per App. A.3 | identical | ✅ match |
| Exchange RevIN window | 16 | 16 | ✅ match |
| Test `drop_last` | False | False | ✅ match |
| **Val `drop_last`** | unspecified (HF/PatchTST use False) | **True** | ⚠️ minor mismatch |
| **Seed/determinism** | "fixed seed 1 for reproducibility" | partial seeding only | ⚠️ weak |
| **Splits actually run** | all 8 datasets | **only ETTh1** | ❌ incomplete |
| **Data present** | all 8 | **ETT only** | ❌ incomplete |
| Param counts (Table 8) | exact | exact (test passes) | ✅ match |

---

## 12. Missing components

1. **4 of 8 datasets' CSVs** (weather, electricity, exchange_rate, traffic) are
   not downloaded → those benchmarks cannot run.
2. **7 of 8 datasets never trained** — only ETTh1 has checkpoints/logs. No
   ETTh2/ETTm1/ETTm2 runs even though their data is present.
3. **No sweep/orchestration script** to run all 32 cells and assemble Table 12.
4. **No `results/runs.jsonl`** — the ledger was never created (logs exist but
   `backfill_ledger.py` was not run; `report.py` therefore prints nothing).
5. **No `requirements.txt` / environment pinning** (torch/numpy/pandas versions).
6. **No unit tests** for patch↔unpatch round-trip, RevIN invertibility, or
   loader window counts (the dataset docstring promises a test that doesn't exist).
7. **No `.gitignore`** — `__pycache__/` and `checkpoints/` are tracked.
8. **No multi-seed support exercised** (paper uses seed 1 only, but variance is
   unknown; report.py supports it but no runs use it).

---

## 13. Bugs or suspicious implementation choices

**Likely-impactful**

- **B1 — Validation loader uses `drop_last=True`** (`dataset.py:198`). This drops
  the last partial val batch, slightly perturbing the val MSE used for *model
  selection* and early stopping. Faithful reproductions (HF/PatchTST) use
  `drop_last=False` on val. Low risk but free to fix; could nudge which epoch is
  chosen. Test loader is correctly `False`.
- **B2 — Incomplete determinism** (`train.py:40-43`). `set_seed` seeds numpy +
  torch CPU/CUDA but **not** Python `random`, **not** `cudnn.deterministic`/
  `benchmark`, and **not** DataLoader workers (no `worker_init_fn`/`generator`).
  With `num_workers=4` + `shuffle=True`, run-to-run and machine-to-machine
  results are not bit-reproducible, contradicting the paper's "fixed seed 1 for
  reproducibility." This is a plausible contributor to the residual ~2% gap.

**Minor / cosmetic**

- **B3 — RevIN denorm divisor** (`revin.py:85`): divides by
  `affine_weight + eps*eps` (=1e-10) rather than the conventional
  `affine_weight + eps`. Dead path while `affine=False`, but wrong if affine is
  ever enabled.
- **B4 — RevIN class default `affine=True`** while PSformer always passes
  `False`. A footgun: instantiating RevIN directly would silently add affine
  params and change Table-8 counts.
- **B5 — `report.py`/`train.py` default `--ledger ./results/runs.jsonl`** but
  nothing creates `results/` until first append; combined with #4 the reporting
  path is currently empty.
- **B6 — `num_workers=4` hard default** in train but `0` in predict — fine, but
  inconsistent and interacts with B2.

**Checked and found CORRECT (not bugs), to avoid re-litigating**

- `d_k=N` scaling, patch/merge ordering, full-PSBlock fusion, channel-independent
  head, MSE-in-scaled-space, numel-weighted eval averaging, val/test window
  pull-back by `L`, non-persistent RevIN buffers (correctly recomputed per
  forward at eval), SAM two-step logic. The large val MSE (~0.9 on ETTh1)
  **matches the paper's own Figure 4** and is not a defect.

---

## 14. Suggested improvement roadmap

**Phase 0 — Reproduce the rest of the table (highest value, no code risk)**
1. Download the 4 missing CSVs (`scripts/download_data.sh` + Tsinghua/HF bundle);
   verify `M` per Table 1 (Weather 21, Electricity 321, Exchange 8, Traffic 862).
2. Run ETTh2/ETTm1/ETTm2 now (data already present) — fast, validates the
   1-vs-3-encoder configs and SAM ρ values beyond ETTh1.
3. Add a **sweep driver** (shell or python) over the 32 cells; backfill/append to
   `results/runs.jsonl`; render with `report.py --markdown`.

**Phase 1 — Tighten reproducibility (targets the ~2% gap)**
4. Harden `set_seed`: seed `random`, set `torch.backends.cudnn.deterministic=True`,
   `benchmark=False`, optionally `torch.use_deterministic_algorithms(True)`, and
   pass a seeded `generator` + `worker_init_fn` to the train DataLoader (B2).
5. Set val `drop_last=False` (B1) so model selection matches the test protocol.
6. Pin the environment (`requirements.txt`) and record library/GPU in the ledger
   (already partly captured by `results.py:_env_info`).

**Phase 2 — Robustness & confidence**
7. Add `tests/`: patch↔unpatch round-trip (assert `unpatch(patch(x))==x`), RevIN
   norm/denorm invertibility, loader window-count vs Table 1, and re-wire
   `test_param_count` as a pytest. Fixes the dataset docstring's unmet promise.
8. Fix RevIN denorm divisor (B3) and make the class default `affine=False` (B4)
   to match how PSformer uses it.
9. Multi-seed runs (e.g. seeds 1,2,3) on ETT to quantify variance and decide
   whether the residual gap is within noise — the cleanest way to declare
   "reproduced."

**Phase 3 — If gap persists after Phase 1–2**
10. Cross-check ambiguous points against SAMformer's public code (paper says it
    mirrors SAMformer): exact SAM placement, whether SAM perturbs only the
    attention params, RevIN eps, and Adam betas. Try `cudnn` off and confirm the
    ETTh1 numbers tighten toward 0.352/0.385.
11. Sanity-plot forecasts/attention maps (Fig. 5/6 analog) to confirm SegAtt is
    learning structured maps, not degenerate uniform attention.

**Done-criteria.** All 32 cells in `results/runs.jsonl`; `report.py` mean |ΔMSE|
within a few % of Table 12; param-count test green; reproducibility test
(same seed → same metric) passing.
