"""Hierarchical cross-modal block fusion (paper §3.3).

Constructs the fused multi-modal adjacency::

        ⎡  A^(T)   W^(T→S)  W^(T→R)  W^(T→C)  ⎤
        ⎢ W^(S→T)  A^(S)    W^(S→R)  W^(S→C)  ⎥
    A^Fused =
        ⎢ W^(R→T)  W^(R→S)  Ã^(R)    W^(R→C)  ⎥
        ⎣ W^(C→T)  W^(C→S)  W^(C→R)  Ã^(C)    ⎦

and the matching block-structured interpretability matrix I^Fused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Parameter, init

from .components import InterpretabilityMatrix


class CrossModalBlock(nn.Module):
    """Holds the W^(m→n) and I^(m→n) for one ordered modality pair.

    The cross-modal adjacency is sigmoid(W_raw) so edges are in [0,1]
    and always positive. An initial bias of -1.0 starts cross-modal
    edges weak (~0.27) so the model must actively learn to strengthen
    cross-modal connections rather than starting at noise.
    """

    def __init__(self, dim_src: int, dim_dst: int):
        super().__init__()
        self.W_raw = Parameter(torch.empty(dim_src, dim_dst))
        init.xavier_uniform_(self.W_raw)
        # Start with weak cross-modal edges (sigmoid(-1) ≈ 0.27)
        self.W_bias = Parameter(torch.full((1,), -1.0))
        self.I = InterpretabilityMatrix(dim_src, dim_dst)
        # Ablation lever: when True, W returns a frozen 1/dim_dst uniform
        # matrix (detached — no gradient flows back through adjacency).
        self.uniform = False

    @property
    def W(self):
        if self.uniform:
            d_src, d_dst = self.W_raw.shape
            u = torch.full((d_src, d_dst), 1.0 / d_dst,
                            device=self.W_raw.device, dtype=self.W_raw.dtype)
            return u.detach()
        return torch.sigmoid(self.W_raw + self.W_bias)


class HierarchicalFusion(nn.Module):
    """Builds A^Fused and I^Fused from intra-modality blocks + cross-modal blocks.

    Args:
        modality_dims: dict mapping modality name → number of nodes after pooling
        modality_keys: order of modalities (defines block layout)
    """

    def __init__(self, modality_dims: dict[str, int], modality_keys: list[str]):
        super().__init__()
        self.modality_keys = modality_keys
        self.modality_dims = modality_dims
        self.total_dim = sum(modality_dims[k] for k in modality_keys)

        # Cross-modal blocks for ordered pairs (m → n) where m != n
        self.cross = nn.ModuleDict()
        for m in modality_keys:
            for n in modality_keys:
                if m == n:
                    continue
                self.cross[f"{m}__{n}"] = CrossModalBlock(modality_dims[m], modality_dims[n])

    def assemble(self, intra: dict[str, torch.Tensor],
                 intra_interp: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Stitch the per-modality A^(m) and I^(m) into block matrices.

        Args:
            intra: {m: (B, dim_m, dim_m)} or {m: (dim_m, dim_m)} adjacency per modality
            intra_interp: same shape as intra, the I^(m) matrices
        Returns:
            A_fused: (B, total_dim, total_dim) or (total_dim, total_dim)
            I_fused: same shape as A_fused
        """
        # Determine batch dim from first intra value
        any_v = next(iter(intra.values()))
        batched = any_v.dim() == 3
        B = any_v.shape[0] if batched else 1
        device = any_v.device

        if batched:
            A_fused = torch.zeros(B, self.total_dim, self.total_dim, device=device)
            I_fused = torch.zeros(B, self.total_dim, self.total_dim, device=device)
        else:
            A_fused = torch.zeros(self.total_dim, self.total_dim, device=device)
            I_fused = torch.zeros(self.total_dim, self.total_dim, device=device)

        # Build offset table for block placement
        offsets = {}
        cursor = 0
        for k in self.modality_keys:
            offsets[k] = cursor
            cursor += self.modality_dims[k]

        # Diagonal blocks (intra-modality)
        for m in self.modality_keys:
            d = self.modality_dims[m]
            o = offsets[m]
            if batched:
                A_fused[:, o:o + d, o:o + d] = intra[m]
                I_fused[:, o:o + d, o:o + d] = intra_interp[m].unsqueeze(0).expand(B, -1, -1)
            else:
                A_fused[o:o + d, o:o + d] = intra[m]
                I_fused[o:o + d, o:o + d] = intra_interp[m]

        # Off-diagonal blocks (cross-modal)
        for m in self.modality_keys:
            for n in self.modality_keys:
                if m == n:
                    continue
                blk = self.cross[f"{m}__{n}"]
                d_m = self.modality_dims[m]
                d_n = self.modality_dims[n]
                o_m = offsets[m]
                o_n = offsets[n]
                if batched:
                    A_fused[:, o_m:o_m + d_m, o_n:o_n + d_n] = blk.W.unsqueeze(0).expand(B, -1, -1)
                    I_fused[:, o_m:o_m + d_m, o_n:o_n + d_n] = blk.I().unsqueeze(0).expand(B, -1, -1)
                else:
                    A_fused[o_m:o_m + d_m, o_n:o_n + d_n] = blk.W
                    I_fused[o_m:o_m + d_m, o_n:o_n + d_n] = blk.I()

        return A_fused, I_fused

    def offsets(self) -> dict[str, tuple[int, int]]:
        """Returns {modality: (start, end)} index ranges in the fused dim."""
        out = {}
        cursor = 0
        for k in self.modality_keys:
            out[k] = (cursor, cursor + self.modality_dims[k])
            cursor += self.modality_dims[k]
        return out
