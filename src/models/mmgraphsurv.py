"""MM-GraphSurv: hierarchical multi-modal graph survival model.

End-to-end implementation of the architecture in
``GraphSurv/sample_paper.tex`` Algorithm 1, with the following corrections
relative to the original (DySurv-era) GraphSurv codebase:

  • Honours the s=6 windowed grid declared in the paper text rather than
    the older 24-step hourly grid.
  • Implements all four modalities (dynamic / static / ICD / radiology),
    not just dynamic + static.
  • Soft-pools high-dim ICD and radiology blocks to 50 nodes via two-layer
    MLP soft-assignment (paper §3.2).
  • Uses the hierarchical block fusion from §3.3 with W^(m→n) cross blocks
    and matching I^(m→n) interpretability matrices.
  • Supports competing risks (E parallel cause-specific MLP heads).
  • Plug-and-play modality drop-out at inference: pass ``drop_modalities``
    to ``forward`` to mask out one or more blocks (Table 6 reproduction).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    DynamicGraphConstructor,
    InterpretabilityMatrix,
    apply_ema,
    normalised_fuse,
    GINEncoder,
    SoftPool,
)
from .fusion import HierarchicalFusion


# ----------------------------------------------------------------------------
# Per-modality intra encoders
# ----------------------------------------------------------------------------

class DynamicModalityEncoder(nn.Module):
    """Builds A^(T) (dynamic graph over time-series features) and feature matrix.

    The features are the raw (B, F_dyn, S) → reshaped per-window to (B, F_dyn, 1)
    used as node features for the GIN. The graph is per-window with EMA smoothing.
    """

    def __init__(self, num_features: int, num_windows: int, k_neighbors: int,
                 alpha_ema: float):
        super().__init__()
        self.num_features = num_features
        self.num_windows = num_windows
        self.alpha_ema = alpha_ema
        self.graph = DynamicGraphConstructor(
            num_nodes=num_features, num_windows=num_windows,
            k=k_neighbors, share_across_time=False, symmetric=True,
        )
        self.interp = InterpretabilityMatrix(num_features)

    def forward(self):
        """Returns A_t per window (S, F, F) and the per-window I matrix (F, F)."""
        A_t = self.graph()                            # (S, F, F)
        A_t = apply_ema(A_t, self.alpha_ema)          # smoothed
        return A_t, self.interp()


class StaticModalityEncoder(nn.Module):
    """Static modality: single intra graph A^(S) replicated across time."""

    def __init__(self, num_features: int, k_neighbors: int):
        super().__init__()
        self.graph = DynamicGraphConstructor(
            num_nodes=num_features, num_windows=1,
            k=k_neighbors, share_across_time=True, symmetric=True,
        )
        self.interp = InterpretabilityMatrix(num_features)

    def forward(self):
        """Returns (F, F) intra adjacency and (F, F) interpretability."""
        A = self.graph().squeeze(0)
        return A, self.interp()


class PoolableModalityEncoder(nn.Module):
    """Encoder for high-dim modalities (ICD, radiology) that get soft-pooled.

    Builds a fully-connected modality graph from cosine similarity (paper §3.2)
    and then applies SoftPool to reduce to ``num_clusters`` nodes.
    """

    def __init__(self, in_dim: int, num_clusters: int, k_neighbors: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_clusters = num_clusters
        self.k_neighbors = k_neighbors
        # Project to a low-dim feature space for similarity computation
        self.proj = nn.Linear(in_dim, max(64, in_dim // 4))
        self.pool = SoftPool(in_dim=self.proj.out_features,
                             num_nodes=in_dim, num_clusters=num_clusters)
        self.interp = InterpretabilityMatrix(num_clusters)

    def forward(self, x_block: torch.Tensor):
        """x_block: (B, in_dim) raw modality features (one vector per patient).

        Returns:
            X_pool: (B, K, D_proj) pooled node features
            A_pool: (B, K, K) pooled adjacency
            I_pool: (K, K) interpretability
        """
        # Each "node" in the pre-pool graph is one feature dimension.
        # Node features are patient-specific: each node i gets
        # x_block[:, i] * proj.weight[i] — the raw feature value scaled
        # by the learned projection for that feature.
        node_emb = self.proj.weight.t()                       # (in_dim, D_proj)
        node_norm = F.normalize(node_emb, dim=-1)
        A_full = torch.matmul(node_norm, node_norm.t())        # (in_dim, in_dim)
        A_full = (A_full + 1.0) / 2.0                          # rescale to [0,1]

        B = x_block.shape[0]
        # Patient-specific node features: scale each node's embedding by
        # the patient's raw feature value for that node/feature dimension.
        x_vals = x_block.unsqueeze(-1)                          # (B, in_dim, 1)
        node_feats = node_emb.unsqueeze(0) * x_vals             # (B, in_dim, D_proj)

        X_pool, A_pool, _ = self.pool(node_feats, A_full)
        return X_pool, A_pool, self.interp()


# ----------------------------------------------------------------------------
# Survival heads
# ----------------------------------------------------------------------------

class CauseSpecificHead(nn.Module):
    """Per-risk MLP that predicts K discrete-time hazards."""

    def __init__(self, in_dim: int, num_durations: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_durations),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# Full MM-GraphSurv model
# ----------------------------------------------------------------------------

class MMGraphSurv(nn.Module):
    def __init__(
        self,
        modality_dims: dict[str, int],         # raw F_m per modality (pre-pool for ICD/RAD)
        modality_keys: list[str],              # order: e.g. ["dynamic", "static", "icd", "rad"]
        pool_dims: dict[str, int],             # {modality: K} for pooled modalities
        num_windows: int,                      # s
        num_risks: int,                        # E
        num_durations: int,                    # K
        k_neighbors: int = 5,
        alpha_ema: float = 0.5,
        gin_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        use_temporal_edges: bool = True,
        cause_specific_proj: bool = False,
        cause_specific_gin: bool = False,
        per_cause_hazard: bool = False,
        modality_types: dict[str, str] | None = None,  # {m: "dynamic"|"static"|"pooled"}
        fusion_mode: str = "graph",                    # "graph" | "mlp_concat" (ablation)
        adjacency_mode: str = "learned",               # "learned" | "uniform_frozen" (ablation)
    ):
        super().__init__()
        self.modality_keys = modality_keys
        self.num_windows = num_windows
        self.num_risks = num_risks
        self.num_durations = num_durations
        self.alpha_ema = alpha_ema
        self.use_temporal_edges = use_temporal_edges
        self.cause_specific_proj = cause_specific_proj
        self.cause_specific_gin = cause_specific_gin
        self.per_cause_hazard = per_cause_hazard
        self.fusion_mode = fusion_mode
        self.adjacency_mode = adjacency_mode

        # Modality-type dict: explicit if provided, else infer from the literal
        # keys (legacy: "dynamic" → dynamic, "static" → static, else pooled).
        if modality_types is None:
            modality_types = {
                m: ("dynamic" if m == "dynamic"
                    else "static" if m == "static"
                    else "pooled")
                for m in modality_keys
            }
        self.modality_types = modality_types

        # Per-modality encoders (chosen by TYPE, not by name)
        self.encoders = nn.ModuleDict()
        post_dims = {}
        for m in modality_keys:
            mt = self.modality_types.get(m, m)
            if mt == "dynamic":
                self.encoders[m] = DynamicModalityEncoder(
                    num_features=modality_dims[m],
                    num_windows=num_windows,
                    k_neighbors=k_neighbors,
                    alpha_ema=alpha_ema,
                )
                post_dims[m] = modality_dims[m]
            elif mt == "static":
                self.encoders[m] = StaticModalityEncoder(
                    num_features=modality_dims[m], k_neighbors=k_neighbors,
                )
                post_dims[m] = modality_dims[m]
            elif mt == "pooled" or m in pool_dims:
                self.encoders[m] = PoolableModalityEncoder(
                    in_dim=modality_dims[m],
                    num_clusters=pool_dims[m],
                    k_neighbors=k_neighbors,
                )
                post_dims[m] = pool_dims[m]
            else:
                raise ValueError(f"Unknown modality {m!r} (type={mt!r}); "
                                 f"expected type ∈ {{dynamic, static, pooled}}")

        self.fusion = HierarchicalFusion(modality_dims=post_dims, modality_keys=modality_keys)

        # GIN encoder(s) operate on the fused graph at every window.
        # If cause_specific_gin: split into shared trunk + per-cause tail layer.
        if cause_specific_gin:
            shared_dims = tuple(gin_dims[:-1])
            tail_dim = gin_dims[-1]
            last_shared_dim = shared_dims[-1] if shared_dims else 1
            self.gin_shared = (
                GINEncoder(input_dim=1, dims=shared_dims) if shared_dims else None
            )
            self.gin_cause = nn.ModuleList([
                GINEncoder(input_dim=last_shared_dim, dims=(tail_dim,))
                for _ in range(num_risks)
            ])
        else:
            self.gin = GINEncoder(input_dim=1, dims=gin_dims)

        # Temporal pool over s windows then heads
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        graph_emb_dim = self.fusion.total_dim * gin_dims[-1]

        if cause_specific_proj:
            # Shared trunk to 256, then per-cause projection to 128.
            # Gives each cause its own 128-dim read instead of all causes
            # sharing a single 128-dim h (Option #1 for competing-risk fix).
            self.proj = nn.Sequential(
                nn.Linear(graph_emb_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cause_proj = nn.ModuleList([
                nn.Sequential(nn.Linear(256, 128), nn.ReLU())
                for _ in range(num_risks)
            ])
        else:
            self.proj = nn.Sequential(
                nn.Linear(graph_emb_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
        self.heads = nn.ModuleList([
            CauseSpecificHead(in_dim=128, num_durations=num_durations, dropout=dropout)
            for _ in range(num_risks)
        ])

        # Ablation head: MLP on concatenated raw modality features (no graph,
        # no GIN). Built regardless of fusion_mode so the state_dict is stable,
        # but only used when fusion_mode == "mlp_concat".
        total_raw = sum(modality_dims[k] for k in modality_keys)
        self.mlp_concat_proj = nn.Sequential(
            nn.Linear(total_raw, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Ablation lever: if adjacency_mode == "uniform_frozen", flip every
        # cross-modal block's uniform flag now so forward returns frozen W.
        if self.adjacency_mode == "uniform_frozen":
            for blk in self.fusion.cross.values():
                blk.uniform = True

    # ------------------------------------------------------------------
    # Slicing helper: take (B, S, F_total) tensor → per-modality blocks
    # ------------------------------------------------------------------

    def _slice_inputs(self, x: torch.Tensor, modality_slices: dict[str, slice]) -> dict[str, torch.Tensor]:
        """x: (B, S, F_total) → {m: (B, S, F_m)}"""
        return {m: x[:, :, sl] for m, sl in modality_slices.items()}


def _offset_for(active_keys: list[str], modality: str,
                modality_dims: dict[str, int]) -> tuple[int, int]:
    """Compute the (lo, hi) column slice of ``modality`` in the concatenated
    per-window node feature tensor for the given active modality order."""
    cursor = 0
    for m in active_keys:
        d = modality_dims[m]
        if m == modality:
            return cursor, cursor + d
        cursor += d
    raise KeyError(modality)


# Re-open MMGraphSurv to add forward (was accidentally closed by the helper above)
def _mmgraphsurv_forward(self, x: torch.Tensor, modality_slices: dict[str, slice],
                          drop_modalities: list[str] | None = None,
                          modality_mask: torch.Tensor | None = None) -> dict:
    """x: (B, S, F_total). modality_slices defines per-modality column ranges.

    Args:
        drop_modalities: global modality ablation — drop these blocks for
            ALL samples in the batch (Table 6 reproduction).
        modality_mask: per-sample (B, num_modalities) float tensor. 0 means
            the corresponding modality is absent for that sample; node
            features for that block are zeroed so no information flows
            from it through the GIN. Used for honest multi-modal training
            when some patients don't have all modalities (Decision D18).

    Returns:
        dict with keys:
            logits: (B, E, K) cause-specific hazards
            h: (B, 128) projected embedding
            intra_interp: {m: (F_m, F_m)} per-modality interpretability matrices
            I_fused: (F_fused, F_fused) last-window block-structured interpretability
            A_fused: (F_fused, F_fused) last-window fused adjacency
            Z: (B, S, F_fused, gin_out) per-window graph embeddings
    """
    B, S, _ = x.shape
    assert S == self.num_windows, f"expected {self.num_windows} windows, got {S}"
    drop_modalities = set(drop_modalities or [])

    active_keys = [m for m in self.modality_keys if m not in drop_modalities]

    slices = self._slice_inputs(x, modality_slices)

    # --------------------------------------------------------------
    # Ablation shortcut: fusion_mode == "mlp_concat" — no graph, no GIN,
    # just an MLP on the raw modality-concat feature vector (mean-pooled
    # across time windows for comparability with the graph path's
    # window-wise processing).
    # --------------------------------------------------------------
    if getattr(self, "fusion_mode", "graph") == "mlp_concat":
        # x: (B, S, F_total) → (B, F_total) via mean-pool across S windows
        x_mean = x.mean(dim=1)
        h = self.mlp_concat_proj(x_mean)                            # (B, 128)
        logits = torch.stack([head(h) for head in self.heads], dim=1)   # (B, E, K)
        # Dummy tensors for the pipe's expected keys so downstream code
        # doesn't explode. They're never consumed in mlp_concat mode.
        zero = torch.zeros(1, device=x.device)
        return {
            "logits": logits, "h": h,
            "intra_interp": {},
            "I_fused": zero, "A_fused": zero, "Z": zero,
        }

    # ---- Per-modality intra graphs ----------------------------------
    intra: dict[str, torch.Tensor] = {}
    intra_interp: dict[str, torch.Tensor] = {}
    node_feats: dict[str, torch.Tensor] = {}    # (B, S, dim_m_post, 1)

    _uniform_adj = getattr(self, "adjacency_mode", "learned") == "uniform_frozen"

    def _to_uniform(A_ref: torch.Tensor) -> torch.Tensor:
        d = A_ref.shape[-1]
        return torch.full_like(A_ref, 1.0 / d).detach()

    for m in active_keys:
        enc = self.encoders[m]
        mt = self.modality_types.get(m, m)
        if mt == "dynamic":
            A_t, I_m = enc()                       # (S, F, F), (F, F)
            if _uniform_adj:
                A_t = _to_uniform(A_t)
            # node features: per-window scalar value of each feature
            # x_dyn (B, S, F) → (B, S, F, 1)
            x_dyn = slices[m]
            feat = x_dyn.unsqueeze(-1)
            intra[m] = A_t                          # (S, F, F)
            intra_interp[m] = I_m
            node_feats[m] = feat
        elif mt == "static":
            A_s, I_s = enc()                        # (F, F), (F, F)
            if _uniform_adj:
                A_s = _to_uniform(A_s)
            # static features replicated across S
            x_stat = slices[m].mean(dim=1)            # (B, F) — average across windows
            feat = x_stat.unsqueeze(1).expand(B, S, -1).unsqueeze(-1)
            intra[m] = A_s.unsqueeze(0).expand(S, -1, -1)
            intra_interp[m] = I_s
            node_feats[m] = feat
        else:
            # Pooled modality: take time-window 0 representation
            x_block = slices[m][:, 0, :]             # (B, F_m_raw)
            X_pool, A_pool, I_p = enc(x_block)       # (B, K, D), (B, K, K), (K, K)
            if _uniform_adj:
                A_pool = _to_uniform(A_pool)
            # Reduce node features to scalar via mean over D for GIN input
            # (alternative: project to 1)
            feat = X_pool.mean(dim=-1, keepdim=True)            # (B, K, 1)
            feat = feat.unsqueeze(1).expand(B, S, -1, -1)        # (B, S, K, 1)
            intra[m] = A_pool.mean(dim=0).unsqueeze(0).expand(S, -1, -1)  # share across batch
            intra_interp[m] = I_p
            node_feats[m] = feat

    # ---- Hierarchical fusion (per window) -----------------------------
    graph_window_embeddings = []
    last_A_fused = None
    last_I_fused = None
    for s in range(S):
        intra_s = {}
        for m in active_keys:
            A = intra[m]
            if A.dim() == 3:
                intra_s[m] = A[s]
            else:
                intra_s[m] = A
        A_fused_s, I_fused_s = self.fusion.assemble(intra_s, intra_interp)
        G = normalised_fuse(A_fused_s, I_fused_s)        # (F_fused, F_fused)
        last_A_fused = A_fused_s
        last_I_fused = I_fused_s

        # Concatenate node features for active modalities at this window
        feats_s = torch.cat([node_feats[m][:, s] for m in active_keys], dim=1)  # (B, F_fused, 1)

        # Per-sample modality mask: zero out node features for modalities
        # that are absent for each patient (Option 1 modality handling).
        if modality_mask is not None:
            for mi, m in enumerate(active_keys):
                key_idx = self.modality_keys.index(m)
                m_active = modality_mask[:, key_idx].view(B, 1, 1)  # (B,1,1)
                lo, hi = _offset_for(active_keys, m, self.fusion.modality_dims)
                feats_s[:, lo:hi, :] = feats_s[:, lo:hi, :] * m_active

        # GIN on the per-window fused graph
        if self.cause_specific_gin:
            z_shared = (
                self.gin_shared(feats_s, G) if self.gin_shared is not None else feats_s
            )
            per_cause_z = [g(z_shared, G) for g in self.gin_cause]
            graph_window_embeddings.append(per_cause_z)   # list[E] of (B, F_fused, tail)
        else:
            z = self.gin(feats_s, G)                      # (B, F_fused, gin_out)
            graph_window_embeddings.append(z)

    # ---- EMA on GIN embeddings → take last → cause-specific head pass ----------
    if self.cause_specific_gin:
        # graph_window_embeddings: list[S] of list[E] of (B, F_fused, tail)
        per_cause_windows = list(zip(*graph_window_embeddings))   # tuple length E
        per_cause_logits = []
        per_cause_h = []
        for e in range(self.num_risks):
            windows_e = list(per_cause_windows[e])
            if self.alpha_ema > 0 and S > 1:
                smoothed = [windows_e[0]]
                for t in range(1, S):
                    smoothed.append(
                        self.alpha_ema * smoothed[-1]
                        + (1.0 - self.alpha_ema) * windows_e[t]
                    )
                Z_last_e = smoothed[-1]
            else:
                Z_last_e = windows_e[-1]
            Z_flat_e = Z_last_e.reshape(B, -1)
            if self.cause_specific_proj:
                shared_trunk = self.proj(Z_flat_e)
                h_e = self.cause_proj[e](shared_trunk)
            else:
                h_e = self.proj(Z_flat_e)
            per_cause_h.append(h_e)
            per_cause_logits.append(self.heads[e](h_e))
        logits = torch.stack(per_cause_logits, dim=1)             # (B, E, K)
        h = torch.stack(per_cause_h, dim=1).mean(dim=1)            # mean for aux losses
        # Z output (for downstream attribution): use cause 0 trajectories
        Z = torch.stack([w[0] for w in graph_window_embeddings], dim=1).detach()
    else:
        Z = torch.stack(graph_window_embeddings, dim=1)    # (B, S, F_fused, gin_out)

        # EMA smooth across windows, then take last step (causal, like LSTM final hidden)
        if self.alpha_ema > 0 and S > 1:
            smoothed = [graph_window_embeddings[0]]
            for t in range(1, S):
                smoothed.append(
                    self.alpha_ema * smoothed[-1]
                    + (1.0 - self.alpha_ema) * graph_window_embeddings[t]
                )
            Z_last = smoothed[-1]                          # (B, F_fused, gin_out)
        else:
            Z_last = graph_window_embeddings[-1]

        Z_flat = Z_last.reshape(B, -1)                     # (B, F_fused*gin_out)
        if self.cause_specific_proj:
            shared = self.proj(Z_flat)                     # (B, 256) shared trunk
            per_cause_h = [cp(shared) for cp in self.cause_proj]   # list of (B, 128)
            logits = torch.stack(
                [head(h_e) for head, h_e in zip(self.heads, per_cause_h)],
                dim=1,
            )                                              # (B, E, K)
            h = shared                                     # use shared trunk for aux losses
        else:
            h = self.proj(Z_flat)                          # (B, 128)
            logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, E, K)

    return {
        "logits": logits,
        "h": h,
        "intra_interp": {m: intra_interp[m].detach() for m in active_keys},
        "I_fused": last_I_fused.detach(),
        "A_fused": last_A_fused.detach(),
        "Z": Z.detach(),
    }


MMGraphSurv.forward = _mmgraphsurv_forward
