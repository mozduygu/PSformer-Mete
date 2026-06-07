"""
Parameter-count sanity test against PSformer paper Table 8.

We cannot run the model in this sandbox (no torch installed), but we can
compute the parameter counts by hand using the architecture spec. This file
both documents the expected counts and provides a runnable check the user can
execute on their machine to verify the implementation matches the paper
BEFORE training.

Usage on the user's machine:
    python -m experiments.test_param_count
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PSformer


# Paper Table 8: (dataset, horizon) -> (full_params, encoder_only_params)
# Note: encoder_only excludes the linear mapping head and (since paper uses
# no affine) RevIN parameters.
PAPER_TABLE_8 = {
    ("ETTh1",   96):  (52_416, 3_168),
    ("ETTh1",  192):  (101_664, 3_168),
    ("ETTh1",  336):  (175_536, 3_168),
    ("ETTh1",  720):  (372_528, 3_168),
    # Weather: 21 variables -> patch_size 16 -> C = 21*16 = 336.
    # 3 encoders, so encoder-only = 3 * 3,168 = 9,504.
    ("Weather", 96):  (58_752, 9_504),
    ("Weather", 192): (108_000, 9_504),
    ("Weather", 336): (181_872, 9_504),
    ("Weather", 720): (378_864, 9_504),
    # Traffic: 862 variables. 3 encoders.
    # Same encoder count (because encoder params depend only on N=32, not M)
    ("Traffic", 96):  (58_752, 9_504),
    ("Traffic", 192): (108_000, 9_504),
    ("Traffic", 336): (181_872, 9_504),
    ("Traffic", 720): (378_864, 9_504),
}

# Per-dataset architecture from Appendix A.3
DATASET_CONFIG = {
    "ETTh1":       {"M": 7,   "n_enc": 1},
    "ETTh2":       {"M": 7,   "n_enc": 1},
    "ETTm1":       {"M": 7,   "n_enc": 3},
    "ETTm2":       {"M": 7,   "n_enc": 1},
    "Weather":     {"M": 21,  "n_enc": 3},
    "Electricity": {"M": 321, "n_enc": 3},
    "Exchange":    {"M": 8,   "n_enc": 1},
    "Traffic":     {"M": 862, "n_enc": 3},
}


def main():
    print(f"{'Dataset':<10} {'H':>4} {'Full (got)':>12} {'Full (paper)':>14}"
          f" {'Enc (got)':>10} {'Enc (paper)':>12}  {'Match?'}")
    print("-" * 80)

    all_ok = True
    for (dataset, horizon), (full_paper, enc_paper) in PAPER_TABLE_8.items():
        cfg = DATASET_CONFIG[dataset]
        model = PSformer(
            num_vars=cfg["M"],
            lookback=512,
            horizon=horizon,
            segments=32,
            num_encoders=cfg["n_enc"],
            revin_affine=False,
        )
        got_full = model.num_parameters(encoder_only=False)
        got_enc = model.num_parameters(encoder_only=True)

        match = (got_full == full_paper) and (got_enc == enc_paper)
        all_ok = all_ok and match
        marker = "OK" if match else "DIFF"
        print(f"{dataset:<10} {horizon:>4} {got_full:>12,} {full_paper:>14,}"
              f" {got_enc:>10,} {enc_paper:>12,}  {marker}")

    print()
    if all_ok:
        print("All parameter counts match Table 8.")
        return 0
    else:
        print("Some parameter counts DO NOT match Table 8 -- inspect above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
