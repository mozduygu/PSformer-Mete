"""
RevIN: Reversible Instance Normalization for Time Series Forecasting.
Kim et al., ICLR 2022.

Used by PSformer at both input (normalize) and output (denormalize) of the model.
For Exchange dataset, paper uses a smaller lookback window (16) for computing
statistics due to non-stationarity (see Appendix B.7).
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    For input X of shape (B, M, L) with M variables and L time steps:
      - 'norm' mode: subtract per-instance per-variable mean, divide by std,
                     optionally apply learnable affine (gamma, beta).
      - 'denorm' mode: invert the transform using stored statistics.

    The statistics are computed over the time dimension. If `stat_window` is
    provided, statistics are computed only over the LAST `stat_window` time
    steps (used for Exchange dataset per Appendix B.7).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True,
                 stat_window: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.stat_window = stat_window  # None => use full lookback for stats

        if self.affine:
            # Learnable per-variable scale and shift
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Buffers populated during 'norm' and used during 'denorm'
        self.register_buffer("_mean", torch.zeros(1), persistent=False)
        self.register_buffer("_stdev", torch.ones(1), persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: tensor shaped (B, M, L) for 'norm' or (B, M, F) for 'denorm'.
            mode: 'norm' or 'denorm'.
        """
        if mode == "norm":
            self._compute_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def _compute_statistics(self, x: torch.Tensor) -> None:
        """Compute per-instance, per-variable mean and std along time axis."""
        # x: (B, M, L)
        if self.stat_window is not None and x.shape[-1] > self.stat_window:
            x_for_stats = x[..., -self.stat_window:]
        else:
            x_for_stats = x

        # Keep dim for broadcastable subtraction in normalize/denormalize
        self._mean = x_for_stats.mean(dim=-1, keepdim=True).detach()
        # Use unbiased=False to match common RevIN implementations
        var = x_for_stats.var(dim=-1, keepdim=True, unbiased=False).detach()
        self._stdev = torch.sqrt(var + self.eps)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._stdev
        if self.affine:
            # affine_weight/bias: (M,) -> (1, M, 1) for broadcasting against (B, M, L)
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(-1)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(-1)
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(-1)
            # Avoid divide by zero if affine_weight is initialized/learned to 0
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(-1) + self.eps * self.eps)
        x = x * self._stdev + self._mean
        return x
