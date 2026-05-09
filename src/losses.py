"""Composite MM-GraphSurv loss (paper §3.3).

L_total = α·L_NLL + β·L_rank + λ·L_smooth + γ·L_cif_smooth
where:
  • L_NLL is the cause-specific discrete-time NLL with right-censoring
  • L_rank is a hinge ranking loss on cumulative incidence at observed event times
  • L_smooth is graph node-feature smoothness ‖h_k − h_l‖² over learnt edges
  • L_cif_smooth is graph-regularised CIF smoothness (D36): patients similar
    in the learned graph space should have similar survival curves
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cif(logits: torch.Tensor, per_cause_hazard: bool = False) -> torch.Tensor:
    """Logits (B, E, K) → cumulative incidence (B, E, K).

    Default: flattened softmax across (E*K) — joint PMF parameterisation
    (paper Eq. 6-7). Cross-cause probability mass is coupled.

    ``per_cause_hazard=True``: sigmoid per (e, t) gives independent
    discrete-time cause-specific hazards; CIF derived via
    F_e(t) = Σ_{s≤t} h_{e,s} · S(s-1) with S(t) = Π_{s≤t} Π_e (1 - h_{e,s}).
    Decouples the loss signal across causes.
    """
    if per_cause_hazard:
        return _compute_cif_hazard(logits)
    B, E, K = logits.shape
    flat = logits.reshape(B, E * K)
    pmf = F.softmax(flat, dim=-1).reshape(B, E, K)
    cif = pmf.cumsum(dim=-1)
    return cif


def _compute_cif_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Hazard-parameterisation CIF (see ``compute_cif`` docstring)."""
    B, E, K = logits.shape
    h = torch.sigmoid(logits).clamp(min=1e-8, max=1 - 1e-8)
    not_h = 1.0 - h
    # Prob of no event in bin s across causes: Π_e (1 - h[e, s])
    prod_e = not_h.prod(dim=1)                                     # (B, K)
    # S(t) = Π_{s≤t} prod_e
    S = prod_e.cumprod(dim=-1)                                     # (B, K)
    # S_prev(s) = S(s-1),  S_prev(0) = 1
    S_prev = torch.cat(
        [torch.ones(B, 1, device=h.device, dtype=h.dtype), S[:, :-1]],
        dim=-1,
    )                                                              # (B, K)
    # F_e(t) = cumsum_{s≤t} ( h[e, s] · S_prev(s) )
    incr = h * S_prev.unsqueeze(1)                                 # (B, E, K)
    cif = incr.cumsum(dim=-1)                                      # (B, E, K)
    return cif


def neg_log_likelihood(
    logits: torch.Tensor,           # (B, E, K)
    durations_idx: torch.Tensor,    # (B,)
    events: torch.Tensor,           # (B,)  0=censor, 1..E=cause
    class_weights: torch.Tensor | None = None,
    per_cause_hazard: bool = False,
) -> torch.Tensor:
    """Cause-specific NLL with right-censoring (paper §3.3 Eq. 8).

    ``per_cause_hazard=True`` switches to cause-specific discrete-time
    hazards: each (e, t) cell is an independent Bernoulli whose loss is
    log-likelihood of "event e at time t" given survival so far. Class
    weights multiply only the positive (event) class of each cause's
    BCE — no cross-cause zero-sum. Dispatches to
    ``_neg_log_likelihood_hazard``.
    """
    if per_cause_hazard:
        return _neg_log_likelihood_hazard(
            logits, durations_idx, events, class_weights=class_weights,
        )
    B, E, K = logits.shape
    pmf = F.softmax(logits.reshape(B, E * K), dim=-1).reshape(B, E, K)
    eps = 1e-12

    surv = 1.0 - pmf.sum(dim=1).cumsum(dim=-1).clamp(0, 1 - eps)   # (B, K)
    surv_at_t = surv.gather(1, durations_idx.unsqueeze(1)).squeeze(1)

    # For uncensored patients: log p(event=cause, time=t)
    is_event = (events > 0)
    log_event_term = torch.zeros(B, device=logits.device)
    if is_event.any():
        e_idx = (events[is_event] - 1).long()
        t_idx = durations_idx[is_event]
        cause_pmf = pmf[is_event][torch.arange(is_event.sum(), device=logits.device), e_idx, t_idx]
        if class_weights is not None:
            w = class_weights[events[is_event]]
        else:
            w = torch.ones_like(cause_pmf)
        log_event_term[is_event] = w * torch.log(cause_pmf.clamp(min=eps))

    # For censored patients: log S(t)
    is_cens = ~is_event
    log_cens_term = torch.zeros(B, device=logits.device)
    if is_cens.any():
        log_cens_term[is_cens] = torch.log(surv_at_t[is_cens].clamp(min=eps))

    return -(log_event_term + log_cens_term).mean()


def _neg_log_likelihood_hazard(
    logits: torch.Tensor,           # (B, E, K)
    durations_idx: torch.Tensor,    # (B,)
    events: torch.Tensor,           # (B,)  0=censor, 1..E=cause
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-cause discrete-time hazard NLL.

    Each (e, t) cell is an independent Bernoulli with probability
    h[b, e, t] = σ(logits[b, e, t]).

    For patient i with event e_i at t_i:
      log L = log h[i, e_i-1, t_i]                          (event term)
            + Σ_{e=e_i-1, s<t_i} log(1 - h[i, e, s])         (own-cause survival)
            + Σ_{e'≠e_i-1, s≤t_i} log(1 - h[i, e', s])       (other-cause survival)
    For censored patient i at t_i:
      log L = Σ_{e, s≤t_i} log(1 - h[i, e, s])

    ``class_weights[e]`` (for e = events value) multiplies ONLY the positive
    (event) term for cause e. Survival (negative) terms are unweighted so
    each cause's loss signal is independent — the zero-sum across causes
    that plagues the flat-softmax formulation is removed.
    """
    B, E, K = logits.shape

    # Build target y[i, e, s] = 1 iff patient i had event of cause e+1 at bin s.
    target = torch.zeros_like(logits)
    is_event = events > 0
    ev_idx = torch.where(is_event)[0]
    if ev_idx.numel() > 0:
        ev_cause = (events[ev_idx] - 1).long()
        ev_t = durations_idx[ev_idx]
        target[ev_idx, ev_cause, ev_t] = 1.0

    # Mask[i, e, s] = 1 where the BCE term is observed:
    #   censored: all (e, s≤t_i)
    #   event:    for cause e with s≤t_i (positive at s=t_i, zero elsewhere)
    #             for other causes with s≤t_i (all zero, observed as no-event)
    t = durations_idx.view(B, 1, 1)
    time_range = torch.arange(K, device=logits.device).view(1, 1, K)
    mask = (time_range <= t).to(logits.dtype).expand(B, E, K).contiguous()

    # Per-cause positive-class weighting. class_weights is [censor, c1, ..., cE]
    # so the e-th cause's pos_weight is class_weights[e+1].
    if class_weights is not None:
        pos_weight = class_weights[1:E + 1].to(logits.dtype).view(1, E, 1)
    else:
        pos_weight = None

    # Numerically stable BCE-with-logits, no sigmoid→log pair (avoids NaN
    # when hazards saturate near 0/1 with large pos_weights).
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight,
    )                                                       # (B, E, K)

    # Sum observed terms per patient (full log-likelihood scale).
    per_patient = (mask * bce).sum(dim=(1, 2))              # (B,)

    return per_patient.mean()


def ranking_loss(
    logits: torch.Tensor,           # (B, E, K)
    durations_idx: torch.Tensor,    # (B,)
    events: torch.Tensor,           # (B,)
    sigma: float = 0.1,
    max_pairs: int = 5000,
    per_cause_hazard: bool = False,
    bin_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cause-specific hinge ranking loss (paper §3.3 Eq. 9) — vectorized.

    When ``bin_weights`` is provided (shape (K,) per event-time bin), each
    ranking pair is scaled by bin_weights[anchor_bin]. Weights should be
    normalized so mean ≈ 1 to preserve loss magnitude. Typical construction:
    precompute global train-set inverse-frequency with sqrt smoothing and a
    max-ratio cap (see ``compute_bin_rank_weights``).
    """
    cif = compute_cif(logits, per_cause_hazard=per_cause_hazard)  # (B, E, K)
    B, E, K = cif.shape
    loss = logits.new_zeros(())

    for e in range(E):
        e_label = e + 1
        unc_idx = torch.where(events == e_label)[0]
        n_unc = len(unc_idx)
        if n_unc < 1:
            continue

        n_pairs = min(n_unc * 20, max_pairs)
        i_sel = unc_idx[torch.randint(n_unc, (n_pairs,), device=logits.device)]
        j_sel = torch.randint(B, (n_pairs,), device=logits.device)

        valid = (i_sel != j_sel) & (durations_idx[j_sel] > durations_idx[i_sel])
        if valid.sum() < 1:
            continue

        i_v = i_sel[valid]
        j_v = j_sel[valid]
        t_i = durations_idx[i_v].long()

        cif_i = cif[i_v, e, t_i]
        cif_j = cif[j_v, e, t_i]
        margin = (1.0 - (cif_i - cif_j)) / sigma
        per_pair = F.relu(margin)

        if bin_weights is not None:
            w = bin_weights.to(per_pair.device)[t_i]
            loss = loss + (per_pair * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss = loss + per_pair.mean()

    return loss / max(E, 1)


def compute_bin_rank_weights(
    train_events: torch.Tensor | np.ndarray,
    train_durations_idx: torch.Tensor | np.ndarray,
    num_bins: int,
    smoothing: str = "sqrt",
    max_ratio: float = 3.0,
    event_label: int = 1,
) -> torch.Tensor:
    """Precompute per-bin rank weights from train-set event-time distribution.

    Weight formula:
        raw[k] = 1 / (count_k ** p) where p depends on ``smoothing``:
                 'sqrt'  → p = 0.5 (gentle)
                 'inv'   → p = 1.0 (aggressive, original)
                 'log'   → raw[k] = 1 / log(count_k + e)
        normalize to mean = 1 across bins with any events
        clamp to [1/max_ratio, max_ratio] so no bin dominates or vanishes.
    """
    if isinstance(train_events, torch.Tensor):
        train_events = train_events.cpu().numpy()
    if isinstance(train_durations_idx, torch.Tensor):
        train_durations_idx = train_durations_idx.cpu().numpy()

    mask = train_events == event_label
    counts = np.bincount(train_durations_idx[mask], minlength=num_bins).astype(float)
    counts_safe = np.maximum(counts, 1.0)  # avoid /0 for empty bins

    if smoothing == "sqrt":
        raw = 1.0 / np.sqrt(counts_safe)
    elif smoothing == "log":
        raw = 1.0 / np.log(counts_safe + np.e)
    else:  # 'inv'
        raw = 1.0 / counts_safe

    # Mean over non-empty bins (empty ones still get raw value; clamp will handle)
    nonempty = counts > 0
    if nonempty.sum() > 0:
        raw = raw / raw[nonempty].mean()
    # Clamp so no bin is > max_ratio× or < 1/max_ratio× the mean
    raw = np.clip(raw, 1.0 / max_ratio, max_ratio)
    # Empty bins (no events) get weight 0 — no anchor will ever land there anyway
    raw[~nonempty] = 0.0
    return torch.tensor(raw, dtype=torch.float32)


def smoothness_reg(h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Graph smoothness ‖h_k − h_l‖² weighted by edge strength (paper §3.3)."""
    if A.dim() == 2:
        A = A.unsqueeze(0)
    deg = A.sum(dim=-1)
    L = torch.diag_embed(deg) - A
    Lh = torch.bmm(L, h)
    return (h * Lh).sum(dim=(-2, -1)).mean()


def cif_graph_smoothness(logits: torch.Tensor, per_cause_hazard: bool = False) -> torch.Tensor:
    """Graph-smoothed CIF regularisation (D36)."""
    cif = compute_cif(logits, per_cause_hazard=per_cause_hazard)
    B, E, K = cif.shape
    cif_flat = cif.reshape(B, E * K)
    n_pairs = min(B * 10, 5000)
    i_idx = torch.randint(B, (n_pairs,), device=logits.device)
    j_idx = torch.randint(B, (n_pairs,), device=logits.device)
    valid = i_idx != j_idx
    if valid.sum() < 1:
        return logits.new_zeros(())
    i_v, j_v = i_idx[valid], j_idx[valid]
    diff = (cif_flat[i_v] - cif_flat[j_v]).pow(2).mean(dim=-1)
    return diff.mean()


class MMGraphSurvLoss(nn.Module):
    def __init__(self, alpha_nll: float = 1.0, beta_rank: float = 1.0,
                 lambda_reg: float = 0.1, rank_sigma: float = 0.1,
                 class_weights=None,
                 cif_smooth_weight: float = 0.0,
                 per_cause_hazard: bool = False,
                 bin_stratified_rank: bool = False,
                 num_bins: int | None = None,
                 bin_rank_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha_nll = alpha_nll
        self.beta_rank = beta_rank
        self.lambda_reg = lambda_reg
        self.rank_sigma = rank_sigma
        self.cif_smooth_weight = cif_smooth_weight
        self.per_cause_hazard = per_cause_hazard
        self.bin_stratified_rank = bin_stratified_rank
        self.num_bins = num_bins
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )
        # Precomputed per-bin rank weights (global, train-set-frozen).
        self.register_buffer(
            "bin_rank_weights",
            bin_rank_weights.float() if bin_rank_weights is not None else None,
        )

    def forward(self, output: dict, durations_idx: torch.Tensor, events: torch.Tensor):
        logits = output["logits"]
        h = output.get("h")

        nll = neg_log_likelihood(
            logits, durations_idx, events,
            class_weights=self.class_weights,
            per_cause_hazard=self.per_cause_hazard,
        )
        rank = ranking_loss(
            logits, durations_idx, events, sigma=self.rank_sigma,
            per_cause_hazard=self.per_cause_hazard,
            bin_weights=(self.bin_rank_weights if self.bin_stratified_rank else None),
        )
        smooth = h.pow(2).mean() if h is not None else logits.new_zeros(())

        cif_sm = logits.new_zeros(())
        if self.cif_smooth_weight > 0:
            cif_sm = cif_graph_smoothness(logits, per_cause_hazard=self.per_cause_hazard)

        total = (self.alpha_nll * nll + self.beta_rank * rank
                 + self.lambda_reg * smooth + self.cif_smooth_weight * cif_sm)
        return total, {
            "nll": nll.detach(), "rank": rank.detach(),
            "smooth": smooth.detach(), "cif_sm": cif_sm.detach(),
        }
