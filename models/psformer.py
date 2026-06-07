"""
PSformer model implementation.

Reference:
  Wang et al., "PSformer: Parameter-efficient Transformer with Segment Attention
  for Time Series Forecasting", arXiv:2411.01419v2 (Feb 2025).

Implements exactly the architecture described in §3.2 and Figure 2:

  Pipeline:
    X (B, M, L) -> RevIN(norm)
                -> Patch + cross-channel merge: (B, M, L) -> (B, C, N)
                   where L = P*N and C = M*P, with P the patch length
                -> [PSformer Encoder] x n
                -> Inverse cross-channel merge: (B, C, N) -> (B, M, L)
                -> Linear mapping (L -> F): (B, M, L) -> (B, M, F)
                -> RevIN(denorm)
    Output: (B, M, F)   [F = forecast horizon]

  Each PSformer Encoder uses ONE shared PSBlock W_S across:
      stage-1 SegAtt (which uses W_S three times to produce Q, K, V),
      stage-2 SegAtt (same),
      and a final PSBlock applied after the residual.

  Different encoders have different W_S (paper §3.2 footnote and Table 4).
  Encoder forward (Eq. 8 of the paper):

      X_out = ( Attn2( ReLU( Attn1(X_in) ) ) + X_in ) · W_S

  where Attn_k(X) = Softmax( (X·W_S)(X·W_S)^T / sqrt(d_k) ) · (X·W_S),
  i.e. Q = K = V = X·W_S.   Note d_k = N (paper §3.2).

  No positional encoding (§3, Appendix A.6).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .revin import RevIN


class PSBlock(nn.Module):
    """
    Parameter-Shared Block.

    Implements Eq. (3) of the paper:
        Out = ( GeLU(X · W1) · W2 + X ) · W3

    All three weight matrices are R^{N x N}, and the block transforms a
    tensor of shape (..., N) -> (..., N) -- identity in all preceding dims.
    The block parameters W_S = (W1, W2, W3) are shared across the three
    sites within a single PSformer encoder.

    Bias: enabled by default. We deduced this from the paper's Table 8
    parameter counts: ETTh1 (M=7, L=512, N=32) reports 3,168 encoder
    params for n_enc=1. With biases, one PSBlock has 3*(N*N + N) = 3*1056 = 3168.
    Without biases it would be 3,072 -- so biases are present.
    """

    def __init__(self, n: int, bias: bool = True):
        super().__init__()
        self.n = n
        self.W1 = nn.Linear(n, n, bias=bias)
        self.W2 = nn.Linear(n, n, bias=bias)
        self.W3 = nn.Linear(n, n, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N)
        h = F.gelu(self.W1(x))             # (..., N)
        h = self.W2(h) + x                 # residual after W2 (Eq. 1)
        out = self.W3(h)                   # (..., N)         (Eq. 2)
        return out


class SegmentAttention(nn.Module):
    """
    Spatial-Temporal Segment Attention (SegAtt).

    Given a PSBlock `ps` and an input X of shape (B, C, N):
       Q = K = V = ps(X)            # (B, C, N)
       Attn = Softmax(Q K^T / sqrt(N)) @ V    # attention along the C axis,
                                              # output shape (B, C, N)

    The attention dot-product is taken over N (treating N as the embedding
    dim per the paper's d_model analogy). The softmax/affinity is computed
    across the C dimension. This matches the paper's description in §3.2,
    where they write that attention "primarily applies attention across the
    C dimension".

    Note: a single attention head, no separate Q/K/V projections beyond what
    PSBlock provides. We deliberately do NOT add a Linear out-projection;
    Eq. 8 puts the final W_S after the residual, not inside the attention.
    """

    def __init__(self, n: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n)

    def forward(self, x: torch.Tensor, ps_block: PSBlock) -> torch.Tensor:
        # x: (B, C, N)
        qkv = ps_block(x)                  # (B, C, N), used for all of Q,K,V
        # Attention: scores have shape (B, C, C)
        scores = torch.matmul(qkv, qkv.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, qkv)      # (B, C, N)
        return out


class PSformerEncoder(nn.Module):
    """
    A single PSformer encoder layer.

    All three sub-modules (two SegAtt stages and the final PS block) share the
    same PSBlock instance ps_block. Following Eq. 8:

        x1   = SegAtt_1(x_in)               # uses ps_block to make Q,K,V
        x1a  = ReLU(x1)
        x2   = SegAtt_2(x1a)                # uses ps_block to make Q,K,V
        out  = ps_block( x2 + x_in )        # final PS block, same ps_block
    """

    def __init__(self, n: int):
        super().__init__()
        self.ps_block = PSBlock(n)
        self.attn1 = SegmentAttention(n)
        self.attn2 = SegmentAttention(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        a1 = self.attn1(x, self.ps_block)            # (B, C, N)
        a1 = F.relu(a1)
        a2 = self.attn2(a1, self.ps_block)           # (B, C, N)
        out = self.ps_block(a2 + x)                  # (B, C, N), residual + final PS
        return out


class PSformer(nn.Module):
    """
    PSformer for multivariate long-term time series forecasting.

    Args:
        num_vars (M):     number of variables / channels.
        lookback (L):     input look-back length (paper uses 512).
        horizon (F):      forecast horizon length (96, 192, 336, or 720).
        segments (N):     number of patches (paper uses 32 throughout).
                          Must satisfy lookback % segments == 0; patch_size = L / N.
        num_encoders (n): number of stacked PSformer encoders. Per Appendix A.3:
                          1 for ETTh1/ETTh2/ETTm2/Exchange,
                          3 for ETTm1/Weather/Electricity/Traffic.
        revin_stat_window: optional, use only the last K time-steps to compute
                           RevIN statistics. For Exchange use 16 (Appendix B.7);
                           else None (use full lookback).
        revin_affine:     whether RevIN's per-variable affine is learnable.
                           Default False -- the paper's reported parameter
                           counts (Table 8) match exactly when RevIN has no
                           affine. ETTh1 H=96: 3,168 (encoder) + 49,248 (head)
                           = 52,416 total, matching Table 8 exactly.
    """

    def __init__(self, num_vars: int, lookback: int, horizon: int,
                 segments: int = 32, num_encoders: int = 1,
                 revin_stat_window: int | None = None,
                 revin_affine: bool = False):
        super().__init__()
        if lookback % segments != 0:
            raise ValueError(
                f"lookback ({lookback}) must be divisible by segments ({segments})"
            )
        self.M = num_vars
        self.L = lookback
        self.F = horizon
        self.N = segments
        self.P = lookback // segments        # patch size
        self.C = num_vars * self.P           # segment dim = M * P
        self.num_encoders = num_encoders

        self.revin = RevIN(num_features=num_vars, affine=revin_affine,
                           stat_window=revin_stat_window)

        # Stack of independent encoders (each has its own shared PSBlock weights)
        self.encoders = nn.ModuleList([
            PSformerEncoder(self.N) for _ in range(num_encoders)
        ])

        # Final linear mapping from L -> F (Eq. just below Eq. 8: W_F ∈ R^{L x F})
        # Applied per-variable.
        self.head = nn.Linear(self.L, self.F, bias=True)

    # -----------------------------------------------------------------
    # Patch / inverse-patch helpers
    # -----------------------------------------------------------------
    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, M, L) -> (B, C, N) where C = M*P, N = L/P.

        Step-by-step (matches Figure 1 of the paper):
          1. Reshape (B, M, L) -> (B, M, N, P) by splitting the time axis
             into N non-overlapping patches of size P.
          2. Transpose to (B, N, M, P) to keep "same-position patches across
             variables" together.
          3. Merge the (M, P) dims to form C = M*P.    -> (B, N, C)
          4. Transpose to (B, C, N) so the leading dim used by PSBlock /
             SegAtt is the segment-content C.
        """
        B, M, L = x.shape
        assert M == self.M and L == self.L, (M, L, self.M, self.L)
        # 1) split time -> patches
        x = x.reshape(B, M, self.N, self.P)         # (B, M, N, P)
        # 2) gather same-position patches across variables
        x = x.permute(0, 2, 1, 3).contiguous()      # (B, N, M, P)
        # 3) merge M and P into segment dim C
        x = x.reshape(B, self.N, self.C)            # (B, N, C)
        # 4) put C as the row dim, N as the column (model) dim
        x = x.transpose(1, 2).contiguous()          # (B, C, N)
        return x

    def _unpatch(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of _patch: (B, C, N) -> (B, M, L)."""
        B, C, N = x.shape
        assert C == self.C and N == self.N, (C, N, self.C, self.N)
        # Reverse step 4: -> (B, N, C)
        x = x.transpose(1, 2).contiguous()          # (B, N, C)
        # Reverse step 3: split C back into (M, P) -> (B, N, M, P)
        x = x.reshape(B, self.N, self.M, self.P)
        # Reverse step 2: -> (B, M, N, P)
        x = x.permute(0, 2, 1, 3).contiguous()
        # Reverse step 1: merge (N, P) -> L
        x = x.reshape(B, self.M, self.L)
        return x

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, L)  ->  (B, M, F)
        """
        # 1) RevIN normalize (using full L or stat_window if specified)
        x = self.revin(x, mode="norm")

        # 2) patch + cross-channel merge: (B, M, L) -> (B, C, N)
        h = self._patch(x)

        # 3) Encoder stack
        for enc in self.encoders:
            h = enc(h)                              # (B, C, N)

        # 4) inverse merge: (B, C, N) -> (B, M, L)
        h = self._unpatch(h)

        # 5) head: per-variable Linear(L -> F)
        # h: (B, M, L); apply Linear over the last dim
        out = self.head(h)                          # (B, M, F)

        # 6) RevIN denormalize back to original scale
        out = self.revin(out, mode="denorm")
        return out

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    @torch.no_grad()
    def num_parameters(self, encoder_only: bool = False) -> int:
        if encoder_only:
            return sum(p.numel() for p in self.encoders.parameters())
        return sum(p.numel() for p in self.parameters())
