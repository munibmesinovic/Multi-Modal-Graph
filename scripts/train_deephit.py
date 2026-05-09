#!/usr/bin/env python3
"""DeepHit baseline for MM-GraphSurv — native competing-risk support.

PyTorch reimplementation of the DeepHit architecture from:
    Lee et al., "DeepHit: A Deep Learning Approach to Survival Analysis
    with Competing Risks" (AAAI 2018).
    Original TensorFlow code: https://github.com/chl8856/DeepHit

DeepHit directly models P(event=k, time=t | x) via a shared + cause-specific
subnetwork with softmax output over (num_risks × K) bins. It handles both
single-risk and competing-risk natively — no cause-specific wrapper needed.

Usage:
    python scripts/run_baselines_deephit.py --dataset eicu
    python scripts/run_baselines_deephit.py --dataset mcmed
    python scripts/run_baselines_deephit.py --dataset all

Outputs:
    results/baselines/deephit_{dataset}.json — Ctd / IBS / IBLL per split
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import patch_scipy_simps  # noqa: E402

patch_scipy_simps()

from pycox.evaluation import EvalSurv  # noqa: E402


# =============================================================================
# DeepHit model (PyTorch)
# =============================================================================

class DeepHit(nn.Module):
    """DeepHit: shared sub-network + cause-specific sub-networks + softmax.

    Output: (B, num_risks, K) joint probability mass function over
    (event type, discrete time bin). For single-risk, num_risks=1.
    """

    def __init__(self, in_features: int, num_risks: int, num_bins: int,
                 shared_dims: list[int] = (256, 128),
                 cs_dims: list[int] = (128,),
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__()
        self.num_risks = num_risks
        self.num_bins = num_bins

        # Shared sub-network
        shared_layers = []
        prev = in_features
        for h in shared_dims:
            shared_layers.append(nn.Linear(prev, h))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(h))
            shared_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            prev = h
        self.shared = nn.Sequential(*shared_layers)
        shared_out_dim = prev

        # Cause-specific sub-networks (one per risk)
        # Each takes shared output + residual from input
        cs_input_dim = shared_out_dim + in_features  # residual connection
        self.cs_nets = nn.ModuleList()
        for _ in range(num_risks):
            cs_layers = []
            prev = cs_input_dim
            for h in cs_dims:
                cs_layers.append(nn.Linear(prev, h))
                if batch_norm:
                    cs_layers.append(nn.BatchNorm1d(h))
                cs_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    cs_layers.append(nn.Dropout(dropout))
                prev = h
            cs_layers.append(nn.Linear(prev, num_bins))
            self.cs_nets.append(nn.Sequential(*cs_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_risks, K) PMF (softmax over all risk×time bins)."""
        shared_out = self.shared(x)
        # Residual: concatenate raw input with shared output
        cs_input = torch.cat([shared_out, x], dim=-1)

        # Each cause-specific network outputs (B, K) logits
        cs_outputs = [net(cs_input) for net in self.cs_nets]
        # Stack to (B, num_risks, K)
        logits = torch.stack(cs_outputs, dim=1)

        # Softmax over all (num_risks × K) bins jointly — this is the DeepHit
        # key insight: the PMF sums to 1 across ALL risks and time bins.
        B = logits.shape[0]
        flat_logits = logits.reshape(B, -1)  # (B, num_risks * K)
        flat_pmf = F.softmax(flat_logits, dim=-1)
        pmf = flat_pmf.reshape(B, self.num_risks, self.num_bins)
        return pmf


# =============================================================================
# DeepHit loss (log-likelihood + ranking)
# =============================================================================

class DeepHitLoss(nn.Module):
    """Combined log-likelihood and ranking loss for DeepHit.

    Args:
        alpha: weight for the ranking loss (default 0.1)
        sigma: bandwidth for the ranking loss kernel (default 0.1)
    """

    def __init__(self, alpha: float = 0.1, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, pmf: torch.Tensor, durations_idx: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        """
        pmf:           (B, E, K) — predicted PMF from DeepHit
        durations_idx: (B,) — discrete time bin index for each patient
        events:        (B,) — 0=censored, 1..E=event type
        """
        B, E, K = pmf.shape

        # --- 1. Log-likelihood loss ---
        # For uncensored (event > 0): -log P(event=k, time=t)
        # For censored (event == 0): -log P(surviving past time t) = -log(1 - CDF(t))
        ll_loss = torch.zeros(B, device=pmf.device)

        uncensored = events > 0
        censored = events == 0

        if uncensored.any():
            # Index into the correct risk and time bin
            risk_idx = (events[uncensored] - 1).long()  # 0-indexed risk
            time_idx = durations_idx[uncensored].long()
            # P(event=k, time=t)
            p_event = pmf[uncensored, risk_idx, time_idx]
            ll_loss[uncensored] = -torch.log(p_event + 1e-8)

        if censored.any():
            # P(surviving past t) = 1 - sum of all CDF values up to t
            time_idx_c = durations_idx[censored].long()
            # CIF for all risks up to time t: sum_{k=1}^{E} sum_{s=0}^{t} pmf[k, s]
            # Use cumsum along time, then sum across risks
            cum_pmf = pmf[censored].cumsum(dim=-1)  # (Bc, E, K)
            # Total CIF at time t = sum across risks of CIF_k(t)
            total_cif = cum_pmf.sum(dim=1)  # (Bc, K)
            # Gather at the censoring time
            cif_at_t = total_cif.gather(1, time_idx_c.unsqueeze(1)).squeeze(1)
            surv_at_t = 1.0 - cif_at_t
            ll_loss[censored] = -torch.log(surv_at_t + 1e-8)

        nll = ll_loss.mean()

        # --- 2. Ranking loss ---
        # For each pair (i, j) where i had an event and t_i < t_j:
        # penalise if the predicted CIF at t_i for j's risk > CIF at t_i for i's risk
        if self.alpha > 0 and uncensored.sum() > 1:
            rank_loss = self._ranking_loss(pmf, durations_idx, events)
        else:
            rank_loss = torch.tensor(0.0, device=pmf.device)

        return nll + self.alpha * rank_loss

    def _ranking_loss(self, pmf: torch.Tensor, durations_idx: torch.Tensor,
                      events: torch.Tensor) -> torch.Tensor:
        """Vectorized pairwise ranking loss — subsampled for large batches."""
        B, E, K = pmf.shape

        # CIF per risk: cumsum of PMF along time
        cif = pmf.cumsum(dim=-1)  # (B, E, K)

        uncensored_idx = torch.where(events > 0)[0]
        n_unc = len(uncensored_idx)
        if n_unc < 2:
            return torch.tensor(0.0, device=pmf.device)

        # Sample pairs (i from uncensored, j from all) vectorised
        n_pairs = min(n_unc * 20, 5000)
        i_sel = uncensored_idx[torch.randint(n_unc, (n_pairs,), device=pmf.device)]
        j_sel = torch.randint(B, (n_pairs,), device=pmf.device)

        # Keep only valid pairs: i != j and t_i < t_j
        valid = (i_sel != j_sel) & (durations_idx[i_sel] < durations_idx[j_sel])
        if valid.sum() < 1:
            return torch.tensor(0.0, device=pmf.device)

        i_v = i_sel[valid]
        j_v = j_sel[valid]
        t_i = durations_idx[i_v].long()
        k_i = (events[i_v] - 1).long()

        # Gather CIF values: cif[i, k_i, t_i] and cif[j, k_i, t_i]
        cif_i = cif[i_v, k_i, t_i]
        cif_j = cif[j_v, k_i, t_i]
        loss = torch.exp(torch.clamp((cif_j - cif_i) / self.sigma, max=10.0)).mean()
        return loss


# =============================================================================
# Dataset registry & data loading
# =============================================================================

DATASETS = {
    "eicu":    {"tensor_name": "eICU",    "num_risks": 1},
    "mimic":   {"tensor_name": "MIMIC",   "num_risks": 1},
    "support": {"tensor_name": "SUPPORT", "num_risks": 1},
    "mcmed":   {"tensor_name": "MCMED",   "num_risks": 2},
    "pbc2":    {"tensor_name": "PBC2",    "num_risks": 2},
    "hirid":       {"tensor_name": "HIRID", "num_risks": 1},
    "hirid_circ":  {"tensor_name": "HIRID", "num_risks": 1},
    "hirid_expanded": {"tensor_name": "HIRID", "num_risks": 1},
}


# DeepHit (Lee et al. 2018) is a static-input model. For large datasets,
# use only static demographics mean-pooled across time to match the original
# paper's design and avoid feature explosion from flattening temporal data.
STATIC_RANGES = {
    "eicu":  slice(0, 62),    # demographics only (62 cols)
    "mimic": slice(35, 71),   # static demographics only (36 cols)
    "mcmed": slice(32, 55),   # static demographics only (23 cols)
    "hirid": slice(18, 35),   # static only (17 cols) — age+sex+height+APACHE (D72)
    "hirid_circ": slice(18, 35),  # same static slice (hirid_circ deephit ckpts trained with dataset_key="hirid")
    "hirid_expanded": slice(56, 73),  # 17 static cols at new layout (56 dynamic + 17 static)
    "support": None,          # all static (14 features, s=1)
    "pbc2": slice(11, 15),    # static only: drug, age, sex, histologic (4 cols)
}


def load_split(processed_dir: Path, name: str, split: str, dataset_key: str = ""):
    """Load preprocessed tensors and return flat (N, F) array + labels."""
    x = np.load(processed_dir / f"x_{split}_{name}.npy")
    with open(processed_dir / f"y_{split}_surv_{name}.p", "rb") as f:
        durations_raw, events = pickle.load(f)
    cuts = np.load(processed_dir / f"cuts_{name}.npy")

    static_range = STATIC_RANGES.get(dataset_key)
    if static_range is not None:
        x_flat = x[:, :, static_range].mean(axis=1).astype(np.float32)
    else:
        n, s, feat = x.shape
        x_flat = x.reshape(n, s * feat).astype(np.float32)

    durations = durations_raw.astype(np.float64)
    events = events.astype(np.int64)

    # MC-MED ships raw 4-class labels {0,1,2,3}; collapse {3 -> 2} per
    # configs/mcmed.yaml:event_collapse so train and eval share the
    # 2-risk problem. Mirrors calibrate.py:_maybe_remap_events.
    if dataset_key == "mcmed":
        events[events == 3] = 2

    # Bin durations into discrete indices using the dataset's cuts
    dur_idx = np.searchsorted(cuts, durations, side="right") - 1
    dur_idx = np.clip(dur_idx, 0, len(cuts) - 1).astype(np.int64)

    return x_flat, durations, events, dur_idx, cuts


def _remove_constant_cols(x_train, x_val, x_test):
    std = x_train.std(axis=0)
    keep = std > 1e-10
    return x_train[:, keep], x_val[:, keep], x_test[:, keep], keep


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_deephit(pmf: np.ndarray, durations: np.ndarray, events: np.ndarray,
                     cuts: np.ndarray, num_risks: int) -> dict:
    """Evaluate DeepHit predictions using pycox EvalSurv.

    pmf: (N, E, K) predicted PMF
    """
    time_index = cuts.astype(float)
    results = {}
    ctds, ibss, iblls = [], [], []

    for r in range(num_risks):
        e_label = r + 1
        # CIF for risk r: cumsum of PMF along time
        cif_r = pmf[:, r, :].cumsum(axis=-1)
        surv_r = 1.0 - cif_r

        # For single-risk: all patients
        # For competing-risk: only patients who are censored or had event r
        if num_risks == 1:
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
        else:
            mask = (events == 0) | (events == e_label)
            if mask.sum() < 10 or (events[mask] == e_label).sum() < 5:
                results[f"risk{e_label}"] = {
                    "ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")
                }
                continue
            events_bin = (events[mask] == e_label).astype(int)

        surv_df = pd.DataFrame(surv_r[mask].T, index=time_index)
        dur_r = durations[mask]

        try:
            ev = EvalSurv(surv_df, dur_r, events_bin, censor_surv="km")
            ctd = ev.concordance_td()
            tg = np.linspace(dur_r.min(), dur_r.max(), 100)
            ibs = ev.integrated_brier_score(tg)
            ibll = ev.integrated_nbll(tg)
        except Exception as exc:
            print(f"  EvalSurv error risk {e_label}: {exc}")
            ctd, ibs, ibll = float("nan"), float("nan"), float("nan")

        results[f"risk{e_label}"] = {
            "ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll)
        }
        if not np.isnan(ctd):
            ctds.append(ctd)
            ibss.append(ibs)
            iblls.append(ibll)

    results["mean"] = {
        "ctd": float(np.mean(ctds)) if ctds else float("nan"),
        "ibs": float(np.mean(ibss)) if ibss else float("nan"),
        "ibll": float(np.mean(iblls)) if iblls else float("nan"),
    }
    return results


# =============================================================================
# Training
# =============================================================================

def train_deephit(x_train, dur_idx_train, evt_train,
                  x_val, dur_idx_val, evt_val,
                  num_risks: int, num_bins: int,
                  shared_dims=(256, 128), cs_dims=(128,),
                  dropout: float = 0.3, lr: float = 1e-3,
                  weight_decay: float = 1e-4, alpha_rank: float = 0.1,
                  epochs: int = 100, patience: int = 15,
                  batch_size: int = 256, device: str = "auto"):
    """Train DeepHit with early stopping on validation loss."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[1]
    model = DeepHit(in_features, num_risks, num_bins,
                    shared_dims=shared_dims, cs_dims=cs_dims,
                    dropout=dropout).to(device)
    criterion = DeepHitLoss(alpha=alpha_rank)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_t = torch.from_numpy(x_train).float().to(device)
    d_t = torch.from_numpy(dur_idx_train).long().to(device)
    e_t = torch.from_numpy(evt_train).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    d_v = torch.from_numpy(dur_idx_val).long().to(device)
    e_v = torch.from_numpy(evt_val).long().to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    n_train = len(x_train)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 8:
                continue
            pmf = model(x_t[idx])
            loss = criterion(pmf, d_t[idx], e_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pmf = model(x_v)
            val_loss = criterion(val_pmf, d_v, e_v).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_train_loss:.4f}/{val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best val loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": float(best_val_loss), "epochs_run": epoch}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepHit baseline for MM-GraphSurv")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    # Hyperparameters matched to chl8856/DeepHit original:
    # lr=1e-4, dropout=0.4 (keep_prob=0.6), batch_size=32/64/128
    parser.add_argument("--shared_dims", type=str, default="100,100")
    parser.add_argument("--cs_dims", type=str, default="100")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha_rank", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    shared_dims = [int(d) for d in args.shared_dims.split(",")]
    cs_dims = [int(d) for d in args.cs_dims.split(",")]

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results_dir = ROOT / "results" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    for ds in targets:
        info = DATASETS[ds]
        tname = info["tensor_name"]
        num_risks = info["num_risks"]
        proc_dir = ROOT / "data" / ds / "processed"

        if not proc_dir.exists():
            print(f"\n>>> Skipping {ds}: {proc_dir} not found")
            continue

        print(f"\n{'=' * 65}")
        print(f"  DeepHit baseline — {ds.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'=' * 65}")

        x_train, dur_train, evt_train, didx_train, cuts = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val, dur_val, evt_val, didx_val, _ = load_split(proc_dir, tname, "val", dataset_key=ds)
        x_test, dur_test, evt_test, didx_test, _ = load_split(proc_dir, tname, "test", dataset_key=ds)
        num_bins = len(cuts)

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
        print(f"  Bins (K): {num_bins}  Risks: {num_risks}")

        x_train, x_val, x_test, keep_mask = _remove_constant_cols(x_train, x_val, x_test)
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} constant features → {x_train.shape[1]} remaining")

        model, train_info = train_deephit(
            x_train, didx_train, evt_train,
            x_val, didx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins,
            shared_dims=shared_dims, cs_dims=cs_dims,
            dropout=args.dropout, lr=args.lr,
            weight_decay=args.weight_decay, alpha_rank=args.alpha_rank,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size,
        )
        device = next(model.parameters()).device

        # Evaluate on val and test
        results = {"model_info": {"type": "DeepHit", **train_info,
                                  "n_features": x_train.shape[1],
                                  "num_risks": num_risks, "num_bins": num_bins}}

        for split_name, x_s, dur_s, evt_s in [
            ("val", x_val, dur_val, evt_val),
            ("test", x_test, dur_test, evt_test),
        ]:
            with torch.no_grad():
                pmf = model(torch.from_numpy(x_s).float().to(device)).cpu().numpy()
            metrics = evaluate_deephit(pmf, dur_s, evt_s, cuts, num_risks)
            results[split_name] = metrics
            m = metrics.get("mean", metrics.get("risk1", {}))
            print(f"  {split_name}: Ctd={m.get('ctd', 'n/a'):.4f}  "
                  f"IBS={m.get('ibs', 'n/a'):.4f}  IBLL={m.get('ibll', 'n/a'):.4f}")

        out_path = results_dir / f"deephit_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print(f"\n{'=' * 65}")
    print("  DeepHit baseline complete")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
