"""
Per-(dataset, horizon) hyperparameters for PSformer.

These are the EXACT values reported in the paper. Together with Appendix A.3
and Table 11, they fully specify each experiment.

  Appendix A.3:
    L = 512, N = 32, schedule = "constant", LR = 1e-4, batch = 16
    (batch = 8 for Traffic), epochs = 300, patience = 30, seed = 1.

    Encoders per dataset:
      ETTh1, ETTh2, ETTm2, Exchange  -> 1
      ETTm1, Weather, Electricity, Traffic -> 3

  Table 11 (SAM rho values):
    Listed below, indexed by (dataset, horizon).

  Appendix B.7:
    Exchange uses RevIN with stat_window=16 (not 512).
"""

# Number of encoders per dataset (Appendix A.3)
NUM_ENCODERS = {
    "ETTh1":       1,
    "ETTh2":       1,
    "ETTm1":       3,
    "ETTm2":       1,
    "Weather":     3,
    "Electricity": 3,
    "Exchange":    1,
    "Traffic":     3,
}

# Number of variables (channels) per dataset (Table 1)
NUM_VARS = {
    "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
    "Weather": 21, "Electricity": 321, "Exchange": 8, "Traffic": 862,
}

# Batch size per dataset (Appendix A.3: 16 default, 8 for Traffic)
BATCH_SIZE = {
    "ETTh1": 16, "ETTh2": 16, "ETTm1": 16, "ETTm2": 16,
    "Weather": 16, "Electricity": 16, "Exchange": 16, "Traffic": 8,
}

# RevIN statistics window: None means use full lookback; integer is fixed window.
# Appendix B.7: Exchange uses 16.
REVIN_STAT_WINDOW = {
    "ETTh1": None, "ETTh2": None, "ETTm1": None, "ETTm2": None,
    "Weather": None, "Electricity": None, "Exchange": 16, "Traffic": None,
}

# SAM rho per (dataset, horizon), from Table 11
SAM_RHO = {
    ("ETTh1", 96): 0.6,  ("ETTh1", 192): 0.8,  ("ETTh1", 336): 0.9,  ("ETTh1", 720): 0.6,
    ("ETTh2", 96): 0.1,  ("ETTh2", 192): 0.0,  ("ETTh2", 336): 0.6,  ("ETTh2", 720): 0.5,
    ("ETTm1", 96): 0.4,  ("ETTm1", 192): 0.4,  ("ETTm1", 336): 0.4,  ("ETTm1", 720): 0.4,
    ("ETTm2", 96): 0.0,  ("ETTm2", 192): 0.2,  ("ETTm2", 336): 0.3,  ("ETTm2", 720): 0.3,
    ("Electricity", 96): 0.0, ("Electricity", 192): 0.1,
    ("Electricity", 336): 0.1, ("Electricity", 720): 0.1,
    ("Exchange", 96): 0.2, ("Exchange", 192): 0.1, ("Exchange", 336): 0.2, ("Exchange", 720): 0.2,
    ("Traffic", 96): 0.1, ("Traffic", 192): 0.1, ("Traffic", 336): 0.2, ("Traffic", 720): 0.3,
    ("Weather", 96): 0.1, ("Weather", 192): 0.1, ("Weather", 336): 0.2, ("Weather", 720): 0.3,
}


def get_config(dataset: str, horizon: int) -> dict:
    """Return all hyperparameters needed to reproduce a single (dataset, H) run."""
    if dataset not in NUM_ENCODERS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if (dataset, horizon) not in SAM_RHO:
        raise ValueError(
            f"No SAM rho specified for ({dataset}, H={horizon}). "
            f"Valid horizons are 96, 192, 336, 720."
        )
    return {
        "dataset":           dataset,
        "horizon":           horizon,
        "lookback":          512,
        "segments":          32,
        "num_encoders":      NUM_ENCODERS[dataset],
        "num_vars":          NUM_VARS[dataset],
        "batch_size":        BATCH_SIZE[dataset],
        "lr":                1e-4,
        "epochs":            300,
        "patience":          30,
        "seed":              1,
        "sam_rho":           SAM_RHO[(dataset, horizon)],
        "revin_stat_window": REVIN_STAT_WINDOW[dataset],
        "revin_affine":      False,
    }
