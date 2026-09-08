"""
Permutation-equivariant broadcast bases for graph-free CPEN ("pe-basis").

Implements the masked, mean-scaled T1->T2 (5 ops), T2->T2 (15 ops), and
T2->T1 (5 ops) broadcast tensors. Every summed ("free") input index carries a
weight w_k, where w is either 1/N_valid (uniform) or the raw energy fractions
z (energy-weights, IRC-safe). Weights are applied per the q_tau counting in
the writeup; masked variants (output-diagonal / off-diagonal restrictions) are
always used.

Each basis op has its own D x D weight matrix; the matrices are applied to the
compact representation (e.g. row sums, traces) before broadcasting, which is
mathematically identical to contracting the full broadcast tensor but far
cheaper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_T1_TO_T2 = 5
N_T2_TO_T2 = 15
N_T2_TO_T1 = 5


def _diag_masks(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eye, off) masks with shape (1, N, N, 1)."""
    eye = torch.eye(n, device=device, dtype=dtype).view(1, n, n, 1)
    return eye, 1.0 - eye


def apply_t1_to_t2(
    s: torch.Tensor,
    w: torch.Tensor,
    weights: nn.ModuleList,
) -> torch.Tensor:
    """
    T1 -> T2 lifting: X'_ij = sum_a (tau_a . s)_ij W_a.

    Parameters
    ----------
    s:
        Activated particle features (batch, N, D).
    w:
        Per-particle weights (batch, N); 1/N_valid or energy fractions z.
    weights:
        Five (D, D) linear maps W_1..W_5.
    """
    batch, n, d = s.shape
    eye, off = _diag_masks(n, s.device, s.dtype)
    g = torch.einsum("bn,bnd->bd", w, s)  # weighted global mean, q=1

    out = weights[0](s).unsqueeze(2) * off            # tau~1: s_i -> row i, off-diag
    out = out + weights[1](s).unsqueeze(1) * off      # tau~2: s_j -> col j, off-diag
    out = out + weights[2](g).view(batch, 1, 1, d) * off   # tau~3: global mean, off-diag
    out = out + weights[3](s).unsqueeze(2) * eye      # tau~4: s_i -> (i,i)
    out = out + weights[4](g).view(batch, 1, 1, d) * eye   # tau~5: global mean -> diag
    return out


def _t2_stats(
    S: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (diag, row_mean, col_mean, trace_mean, global_mean) of S under weights w."""
    diag = torch.diagonal(S, dim1=1, dim2=2).permute(0, 2, 1)  # (B, N, D)
    row = torch.einsum("bl,bild->bid", w, S)   # R_i = sum_l w_l S_il
    col = torch.einsum("bk,bkjd->bjd", w, S)   # C_j = sum_k w_k S_kj
    tr = torch.einsum("bk,bkd->bd", w, diag)   # sum_k w_k S_kk
    glob = torch.einsum("bk,bkd->bd", w, row)  # sum_kl w_k w_l S_kl
    return diag, row, col, tr, glob


def apply_t2_to_t2(
    S: torch.Tensor,
    w: torch.Tensor,
    weights: nn.ModuleList,
) -> torch.Tensor:
    """
    T2 -> T2 update with the 15 masked broadcast tensors.

    *S* is the activated edge tensor (batch, N, N, D); *weights* holds the
    fifteen (D, D) linear maps in the order of the writeup enumeration.
    """
    batch, n, _, d = S.shape
    eye, off = _diag_masks(n, S.device, S.dtype)
    diag, row, col, tr, glob = _t2_stats(S, w)

    out = weights[0](S)                                  # tau~1: identity
    out = out + weights[1](S.transpose(1, 2))            # tau~2: transpose
    out = out + weights[2](diag).unsqueeze(2) * off      # tau~3: d_i -> row i, off-diag
    out = out + weights[3](diag).unsqueeze(1) * off      # tau~4: d_j -> col j, off-diag
    out = out + weights[4](diag).unsqueeze(2) * eye      # tau~5: d_i -> (i,i)
    out = out + weights[5](tr).view(batch, 1, 1, d) * off    # tau~6: trace mean, off-diag
    out = out + weights[6](tr).view(batch, 1, 1, d) * eye    # tau~7: trace mean -> diag
    out = out + weights[7](row).unsqueeze(2) * off       # tau~8: R_i -> row i, off-diag
    out = out + weights[8](col).unsqueeze(2) * off       # tau~9: C_i -> row i, off-diag
    out = out + weights[9](row).unsqueeze(1) * off       # tau~10: R_j -> col j, off-diag
    out = out + weights[10](col).unsqueeze(1) * off      # tau~11: C_j -> col j, off-diag
    out = out + weights[11](row).unsqueeze(2) * eye      # tau~12: R_i -> (i,i)
    out = out + weights[12](col).unsqueeze(2) * eye      # tau~13: C_i -> (i,i)
    out = out + weights[13](glob).view(batch, 1, 1, d) * off  # tau~14: global mean, off-diag
    out = out + weights[14](glob).view(batch, 1, 1, d) * eye  # tau~15: global mean -> diag
    return out


def apply_t2_to_t1(
    S: torch.Tensor,
    w: torch.Tensor,
    weights: nn.ModuleList,
) -> torch.Tensor:
    """
    T2 -> T1 lowering with the 5 broadcast tensors.

    X'_i = W_1 S_ii + W_2 (row mean)_i + W_3 (col mean)_i
         + W_4 (off-diagonal global mean) + W_5 (trace mean).
    """
    diag, row, col, tr, glob = _t2_stats(S, w)
    # tau~4 masks the input diagonal: subtract the weighted diagonal part.
    glob_off = glob - torch.einsum("bk,bk,bkd->bd", w, w, diag)

    out = weights[0](diag) + weights[1](row) + weights[2](col)
    out = out + (weights[3](glob_off) + weights[4](tr)).unsqueeze(1)
    return out
