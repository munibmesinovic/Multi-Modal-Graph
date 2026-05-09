#!/usr/bin/env python3
"""Dynamic-DeepHit baseline for MM-GraphSurv.

PyTorch reimplementation of:
    Lee et al., "Dynamic-DeepHit: A Deep Learning Approach for Dynamic
    Survival Analysis with Competing Risks" (IEEE TBME, 2019).
    Original TensorFlow code: https://github.com/chl8856/Dynamic-DeepHit

Key architecture:
    1. RNN (LSTM) with temporal attention over longitudinal time windows
    2. Last measurement concatenated with attention-weighted context vector
    3. Cause-specific FC sub-networks → softmax PMF (same output as DeepHit)
    4. Loss: NLL + ranking + RNN next-step prediction (auxiliary)

This uses the (N, S, F) temporal tensors directly — no flattening.

Usage:
    python scripts/run_baselines_dynamic_deephit.py --dataset eicu
    python scripts/run_baselines_dynamic_deephit.py --dataset all

Outputs:
    results/baselines/dynamic_deephit_{dataset}.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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
# Dynamic-DeepHit model (PyTorch)
# =============================================================================

class TemporalAttention(nn.Module):
    """Attention over RNN hidden states, conditioned on the last hidden state."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, rnn_outputs: torch.Tensor, last_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        rnn_outputs: (B, S-1, H) — hidden states for t=0..S-2
        last_hidden: (B, H) — hidden state at t=S-1
        mask:        (B, S-1) — 1 if time step has data, 0 otherwise

        Returns context: (B, H)
        """
        S_minus_1 = rnn_outputs.shape[1]
        # Expand last_hidden to match sequence length
        last_exp = last_hidden.unsqueeze(1).expand_as(rnn_outputs)  # (B, S-1, H)
        combined = torch.cat([rnn_outputs, last_exp], dim=-1)  # (B, S-1, 2H)

        e = self.attn(combined).squeeze(-1)  # (B, S-1)
        # Mask out padding
        e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=-1)  # (B, S-1)

        context = (a.unsqueeze(-1) * rnn_outputs).sum(dim=1)  # (B, H)
        return context


class DynamicDeepHit(nn.Module):
    """Dynamic-DeepHit: RNN + temporal attention + cause-specific subnetworks.

    Input: (B, S, F) temporal tensor where S = number of time windows.
    Output: (B, num_risks, K) joint PMF over (event type, discrete time bin).
    """

    def __init__(self, in_features: int, num_risks: int, num_bins: int,
                 rnn_hidden: int = 128, rnn_layers: int = 2,
                 cs_hidden: int = 128, cs_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.num_risks = num_risks
        self.num_bins = num_bins
        self.in_features = in_features
        self.rnn_hidden = rnn_hidden

        # Shared RNN over temporal sequence
        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Temporal attention
        self.attention = TemporalAttention(rnn_hidden)

        # RNN prediction head (auxiliary loss: predict next time step)
        self.rnn_pred = nn.Linear(rnn_hidden, in_features)

        # Combining layer: last measurement + context vector → shared representation
        combine_dim = in_features + rnn_hidden  # x_last + context
        self.combine = nn.Sequential(
            nn.Linear(combine_dim, cs_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Cause-specific sub-networks
        self.cs_nets = nn.ModuleList()
        for _ in range(num_risks):
            layers = []
            prev = cs_hidden
            for _ in range(cs_layers):
                layers.extend([
                    nn.Linear(prev, cs_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev = cs_hidden
            layers.append(nn.Linear(prev, num_bins))
            self.cs_nets.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor):
        """
        x: (B, S, F) temporal input
        Returns: pmf (B, num_risks, K), z_pred (B, S-1, F)
        """
        B, S, n_feat = x.shape

        # Run RNN over full sequence
        rnn_out, _ = self.rnn(x)  # (B, S, H)
        rnn_last = rnn_out[:, -1, :]  # (B, H)
        x_last = x[:, -1, :]  # (B, F)

        if S > 1:
            # Split: history (0..S-2) and last (S-1)
            rnn_history = rnn_out[:, :-1, :]  # (B, S-1, H)
            history_mask = (x[:, :-1, :].abs().sum(dim=-1) > 0).float()
            context = self.attention(rnn_history, rnn_last, history_mask)
            z_pred = self.rnn_pred(rnn_history)  # (B, S-1, F)
        else:
            # Degenerate: S=1 (e.g. SUPPORT), no history to attend over
            context = torch.zeros_like(rnn_last)
            z_pred = torch.zeros(B, 0, n_feat, device=x.device)

        # Last measurement + context → combined input
        combined = torch.cat([x_last, context], dim=-1)  # (B, F+H)
        shared = self.combine(combined)  # (B, cs_hidden)

        # Cause-specific outputs
        cs_outputs = [net(shared) for net in self.cs_nets]
        logits = torch.stack(cs_outputs, dim=1)  # (B, E, K)

        # Joint softmax over all (E × K) bins
        flat_logits = logits.reshape(B, -1)
        flat_pmf = F.softmax(flat_logits, dim=-1)
        pmf = flat_pmf.reshape(B, self.num_risks, self.num_bins)

        return pmf, z_pred


# =============================================================================
# Loss
# =============================================================================

class DynamicDeepHitLoss(nn.Module):
    """NLL + ranking + RNN prediction loss."""

    def __init__(self, alpha: float = 0.1, gamma: float = 1.0, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # ranking weight
        self.gamma = gamma  # RNN prediction weight
        self.sigma = sigma

    def forward(self, pmf: torch.Tensor, z_pred: torch.Tensor,
                x: torch.Tensor, durations_idx: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        B, E, K = pmf.shape

        # --- 1. NLL loss (same as DeepHit) ---
        ll_loss = torch.zeros(B, device=pmf.device)

        uncensored = events > 0
        censored = events == 0

        if uncensored.any():
            risk_idx = (events[uncensored] - 1).long()
            time_idx = durations_idx[uncensored].long()
            p_event = pmf[uncensored, risk_idx, time_idx]
            ll_loss[uncensored] = -torch.log(p_event.clamp(min=1e-8))

        if censored.any():
            time_idx_c = durations_idx[censored].long()
            cum_pmf = pmf[censored].cumsum(dim=-1)
            total_cif = cum_pmf.sum(dim=1)
            cif_at_t = total_cif.gather(1, time_idx_c.unsqueeze(1)).squeeze(1)
            surv_at_t = (1.0 - cif_at_t).clamp(min=1e-8)
            ll_loss[censored] = -torch.log(surv_at_t)

        nll = ll_loss.mean()

        # --- 2. Ranking loss (vectorized, same as DeepHit) ---
        rank_loss = torch.tensor(0.0, device=pmf.device)
        if self.alpha > 0 and uncensored.sum() > 1:
            cif = pmf.cumsum(dim=-1)
            unc_idx = torch.where(events > 0)[0]
            n_unc = len(unc_idx)
            n_pairs = min(n_unc * 20, 5000)

            i_sel = unc_idx[torch.randint(n_unc, (n_pairs,), device=pmf.device)]
            j_sel = torch.randint(B, (n_pairs,), device=pmf.device)
            valid = (i_sel != j_sel) & (durations_idx[i_sel] < durations_idx[j_sel])

            if valid.sum() > 0:
                i_v, j_v = i_sel[valid], j_sel[valid]
                t_i = durations_idx[i_v].long()
                k_i = (events[i_v] - 1).long()
                rank_loss = torch.exp(torch.clamp(
                    (cif[j_v, k_i, t_i] - cif[i_v, k_i, t_i]) / self.sigma, max=10.0
                )).mean()

        # --- 3. RNN prediction loss (predict next time step) ---
        # z_pred: (B, S-1, F), x_target: x[:, 1:, :] (the actual next steps)
        # When S=1 (static datasets), z_pred is empty → skip RNN loss
        x_target = x[:, 1:, :]
        if z_pred.numel() > 0:
            rnn_loss = F.mse_loss(z_pred, x_target)
        else:
            rnn_loss = torch.tensor(0.0, device=pmf.device)

        return nll + self.alpha * rank_loss + self.gamma * rnn_loss


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


# Dynamic-DeepHit (Lee et al. 2019) processes temporal sequences + static
# features. For 4-modality datasets, use only dynamic + static features
# (no rad/ICD) matching the original paper's design.
TEMPORAL_RANGES = {
    "eicu":  slice(None),      # 2-modal: all features are dynamic+static
    "mimic": slice(0, 71),     # [0:35 dynamic, 35:71 static] — skip rad/ICD
    "mcmed": slice(0, 55),     # [0:32 dynamic, 32:55 static] — skip rad/ICD
    "support": slice(None),    # all static
    "pbc2": slice(None),       # 2-modal: all features are dynamic+static
    "hirid_expanded": slice(None),  # 2-modal: all 73 cols (56 dyn + 17 static)
}


def load_split(processed_dir: Path, name: str, split: str, dataset_key: str = ""):
    """Load preprocessed tensors — keeps the (N, S, F) shape for RNN."""
    x = np.load(processed_dir / f"x_{split}_{name}.npy").astype(np.float32)
    with open(processed_dir / f"y_{split}_surv_{name}.p", "rb") as f:
        durations_raw, events = pickle.load(f)
    cuts = np.load(processed_dir / f"cuts_{name}.npy")

    # For 4-modality datasets, keep only dynamic + static features
    tr = TEMPORAL_RANGES.get(dataset_key)
    if tr is not None:
        x = x[:, :, tr]

    events = events.astype(np.int64)
    # MC-MED ships raw 4-class labels {0,1,2,3}; collapse {3 -> 2} per
    # configs/mcmed.yaml:event_collapse so train and eval share the
    # 2-risk problem. Mirrors calibrate.py:_maybe_remap_events.
    if dataset_key == "mcmed":
        events[events == 3] = 2

    dur_idx = np.searchsorted(cuts, durations_raw.astype(np.float64), side="right") - 1
    dur_idx = np.clip(dur_idx, 0, len(cuts) - 1).astype(np.int64)

    return x, durations_raw.astype(np.float64), events, dur_idx, cuts


def evaluate_deephit(pmf: np.ndarray, durations: np.ndarray, events: np.ndarray,
                     cuts: np.ndarray, num_risks: int) -> dict:
    """Same eval as DeepHit baseline."""
    time_index = cuts.astype(float)
    results = {}
    ctds, ibss, iblls = [], [], []

    for r in range(num_risks):
        e_label = r + 1
        cif_r = pmf[:, r, :].cumsum(axis=-1)
        surv_r = 1.0 - cif_r

        if num_risks == 1:
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
        else:
            mask = (events == 0) | (events == e_label)
            if mask.sum() < 10 or (events[mask] == e_label).sum() < 5:
                results[f"risk{e_label}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
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

        results[f"risk{e_label}"] = {"ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll)}
        if not np.isnan(ctd):
            ctds.append(ctd); ibss.append(ibs); iblls.append(ibll)

    results["mean"] = {
        "ctd": float(np.mean(ctds)) if ctds else float("nan"),
        "ibs": float(np.mean(ibss)) if ibss else float("nan"),
        "ibll": float(np.mean(iblls)) if iblls else float("nan"),
    }
    return results


# =============================================================================
# Training
# =============================================================================

def train_model(x_train, didx_train, evt_train,
                x_val, didx_val, evt_val,
                num_risks: int, num_bins: int,
                rnn_hidden: int = 128, rnn_layers: int = 2,
                cs_hidden: int = 128,
                dropout: float = 0.3, lr: float = 1e-3,
                weight_decay: float = 1e-4, alpha: float = 0.1,
                gamma: float = 1.0, epochs: int = 100,
                patience: int = 15, batch_size: int = 256,
                device: str = "auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[-1]
    model = DynamicDeepHit(
        in_features, num_risks, num_bins,
        rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
        cs_hidden=cs_hidden, dropout=dropout,
    ).to(device)
    criterion = DynamicDeepHitLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_t = torch.from_numpy(x_train).float().to(device)
    d_t = torch.from_numpy(didx_train).long().to(device)
    e_t = torch.from_numpy(evt_train).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    d_v = torch.from_numpy(didx_val).long().to(device)
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
            optimizer.zero_grad()
            pmf, z_pred = model(x_t[idx])
            loss = criterion(pmf, z_pred, x_t[idx], d_t[idx], e_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pmf, val_z = model(x_v)
            val_loss = criterion(val_pmf, val_z, x_v, d_v, e_v).item()

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
    parser = argparse.ArgumentParser(description="Dynamic-DeepHit baseline")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    # Hyperparameters matched to chl8856/Dynamic-DeepHit original:
    # lr=1e-4, rnn_hidden=100, cs_hidden=100, dropout=0.4, batch_size=32,
    # alpha=1.0(nll), beta=0.1(rank), gamma=1.0(rnn), weight_decay=1e-5
    parser.add_argument("--rnn_hidden", type=int, default=100)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--cs_hidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Ranking loss weight")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="RNN prediction loss weight")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

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
        print(f"  Dynamic-DeepHit — {ds.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'=' * 65}")

        x_train, dur_train, evt_train, didx_train, cuts = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val, dur_val, evt_val, didx_val, _ = load_split(proc_dir, tname, "val", dataset_key=ds)
        x_test, dur_test, evt_test, didx_test, _ = load_split(proc_dir, tname, "test", dataset_key=ds)
        num_bins = len(cuts)

        # Per-feature standardization using training stats (essential for RNN stability)
        mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
        std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        std[std < 1e-8] = 1.0
        x_train = (x_train - mu) / std
        x_val = (x_val - mu) / std
        x_test = (x_test - mu) / std

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
        print(f"  Bins (K): {num_bins}  Risks: {num_risks}  Time steps (S): {x_train.shape[1]}")

        model, train_info = train_model(
            x_train, didx_train, evt_train,
            x_val, didx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins,
            rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
            cs_hidden=args.cs_hidden, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay,
            alpha=args.alpha, gamma=args.gamma,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size,
        )
        device = next(model.parameters()).device

        results = {"model_info": {"type": "DynamicDeepHit", **train_info,
                                  "n_features": x_train.shape[-1],
                                  "num_risks": num_risks, "num_bins": num_bins,
                                  "time_steps": x_train.shape[1]}}

        for split_name, x_s, dur_s, evt_s in [
            ("val", x_val, dur_val, evt_val),
            ("test", x_test, dur_test, evt_test),
        ]:
            with torch.no_grad():
                pmf, _ = model(torch.from_numpy(x_s).float().to(device))
                pmf = pmf.cpu().numpy()
            metrics = evaluate_deephit(pmf, dur_s, evt_s, cuts, num_risks)
            results[split_name] = metrics
            m = metrics.get("mean", metrics.get("risk1", {}))
            print(f"  {split_name}: Ctd={m.get('ctd', 'n/a'):.4f}  "
                  f"IBS={m.get('ibs', 'n/a'):.4f}  IBLL={m.get('ibll', 'n/a'):.4f}")

        out_path = results_dir / f"dynamic_deephit_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print(f"\n{'=' * 65}")
    print("  Dynamic-DeepHit baseline complete")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
