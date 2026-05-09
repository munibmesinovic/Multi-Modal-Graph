"""Reusable graph + temporal components.

Adapted from DynaGraph's `src/model.py` (DynaGraph release at
``DynaGraph Updated/DynaGraph Model Code eICU + HIRID/DynaGraph/src/model.py``)
with extensions for multi-modal use.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


# ----------------------------------------------------------------------------
# Dynamic Graph Construction (paper §3.2 + DynaGraph original)
# ----------------------------------------------------------------------------

class DynamicGraphConstructor(nn.Module):
    """Per-window learnable adjacency via outer product + top-k sparsification.

    Returns A_t in [0,1]^{N×N} (sigmoid + sparsified, optionally symmetrised).
    Self-loops are zeroed; supports temporal sharing of theta/psi via the
    ``share_across_time`` flag.
    """

    def __init__(self, num_nodes: int, num_windows: int, k: int = 5,
                 share_across_time: bool = False, symmetric: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_windows = num_windows
        self.k = min(k, num_nodes - 1)
        self.symmetric = symmetric

        if share_across_time:
            self.theta = Parameter(torch.empty(1, num_nodes, 1))
            self.psi = Parameter(torch.empty(1, 1, num_nodes))
        else:
            self.theta = Parameter(torch.empty(num_windows, num_nodes, 1))
            self.psi = Parameter(torch.empty(num_windows, 1, num_nodes))
        init.xavier_uniform_(self.theta)
        init.xavier_uniform_(self.psi)

    def forward(self) -> torch.Tensor:
        """Returns (S, N, N) sparse adjacency tensor."""
        # Broadcast in case theta/psi are shared across time
        theta = self.theta.expand(self.num_windows, self.num_nodes, 1)
        psi = self.psi.expand(self.num_windows, 1, self.num_nodes)
        A_raw = torch.matmul(theta, psi)                              # (S, N, N)
        idx = torch.arange(self.num_nodes, device=A_raw.device)
        A_raw = A_raw.clone()
        A_raw[:, idx, idx] = float("-inf")

        A_prob = torch.sigmoid(A_raw)                                  # (S, N, N) ∈ (0,1)
        _, topk = A_prob.topk(self.k, dim=-1)
        mask = torch.zeros_like(A_prob)
        mask.scatter_(2, topk, 1.0)
        A_sparse = A_prob * mask

        if self.symmetric:
            A_sparse = 0.5 * (A_sparse + A_sparse.transpose(1, 2))
        A_sparse[:, idx, idx] = 0.0
        return A_sparse


def apply_ema(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average across time axis (paper §3.2).

    A: (S, N, N) → returns A_smooth same shape, A_smooth[0] = A[0].
    """
    if alpha <= 0.0 or A.shape[0] <= 1:
        return A
    out = [A[0]]
    for t in range(1, A.shape[0]):
        out.append(alpha * out[-1] + (1.0 - alpha) * A[t])
    return torch.stack(out, dim=0)


# ----------------------------------------------------------------------------
# Interpretability Matrix
# ----------------------------------------------------------------------------

class InterpretabilityMatrix(nn.Module):
    """Learnable edge importance E with grad-magnitude smoothing.

    Used as the I^(m) and I^(m→n) matrices in the paper. forward() returns a
    sigmoid-scaled, gradient-aware importance matrix.
    """

    def __init__(self, n_rows: int, n_cols: int | None = None):
        super().__init__()
        n_cols = n_cols or n_rows
        self.E = Parameter(torch.empty(n_rows, n_cols))
        init.uniform_(self.E, 0.0, 1.0)
        self.register_buffer("grad_accum", torch.zeros(n_rows, n_cols))

    def update_from_gradients(self, momentum: float = 0.9):
        if self.E.grad is not None:
            self.grad_accum.mul_(momentum).add_((1.0 - momentum) * self.E.grad.abs())

    def forward(self) -> torch.Tensor:
        norm = self.grad_accum / (self.grad_accum.max() + 1e-8)
        return torch.sigmoid(self.E) * (1.0 + norm)


def normalised_fuse(A: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """G = (A + E) ∘ (I + A) followed by symmetric normalisation.

    Operates on the last 2 dims so it works for both (N, N) and (S, N, N).
    """
    if A.dim() == 2:
        I = torch.eye(A.shape[0], device=A.device)
        G = (A + E) * (I + A)
    else:
        N = A.shape[-1]
        I = torch.eye(N, device=A.device).expand_as(A)
        G = (A + E.unsqueeze(0)) * (I + A)
    deg = G.sum(dim=-1).clamp(min=1e-8)
    deg_inv_sqrt = deg.pow(-0.5)
    G = deg_inv_sqrt.unsqueeze(-1) * G * deg_inv_sqrt.unsqueeze(-2)
    return G


# ----------------------------------------------------------------------------
# GIN layers
# ----------------------------------------------------------------------------

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer with 2-layer MLP."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim), A: (B, N, N) or (N, N) → (B, N, out_dim)."""
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        agg = torch.bmm(A, x)
        out = (1.0 + self.eps) * x + agg
        B, N, D = out.shape
        out = self.mlp(out.reshape(B * N, D)).reshape(B, N, -1)
        return out


class GINEncoder(nn.Module):
    def __init__(self, input_dim: int, dims=(128, 64, 32)):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for d in dims:
            self.layers.append(GINLayer(prev, d))
            prev = d
        self.out_dim = prev

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x, A))
        return x


# ----------------------------------------------------------------------------
# Temporal encoder (BiLSTM per node)
# ----------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Bidirectional LSTM applied independently to each node's time series."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                init.orthogonal_(p)
            elif "bias" in name:
                init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, T) → (B, N, T, 2*H)."""
        B, N, T = x.shape
        x_flat = x.reshape(B * N, T, 1)
        out, _ = self.lstm(x_flat)
        return out.reshape(B, N, T, -1)


# ----------------------------------------------------------------------------
# Soft-assignment graph pooling (paper §3.2 — ICD/Rad → 50 nodes)
# ----------------------------------------------------------------------------

class SoftPool(nn.Module):
    """Two-layer MLP soft-assignment pool: N → K nodes.

    Computes assignment matrix M ∈ R^{N×K} via softmax(MLP(X)), then:
        X' = M^T X        ∈ R^{K × D}
        A' = M^T A M      ∈ R^{K × K}
    """

    def __init__(self, in_dim: int, num_nodes: int, num_clusters: int):
        super().__init__()
        self.assign = nn.Sequential(
            nn.Linear(in_dim, max(in_dim // 2, 32)),
            nn.ReLU(),
            nn.Linear(max(in_dim // 2, 32), num_clusters),
        )
        self.num_clusters = num_clusters

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """X: (B, N, D), A: (B, N, N) or (N, N) → X' (B, K, D), A' (B, K, K)."""
        M = F.softmax(self.assign(X), dim=-1)               # (B, N, K)
        X_pool = torch.bmm(M.transpose(1, 2), X)             # (B, K, D)
        if A.dim() == 2:
            A_b = A.unsqueeze(0).expand(X.shape[0], -1, -1)
        else:
            A_b = A
        A_pool = torch.bmm(torch.bmm(M.transpose(1, 2), A_b), M)  # (B, K, K)
        return X_pool, A_pool, M
