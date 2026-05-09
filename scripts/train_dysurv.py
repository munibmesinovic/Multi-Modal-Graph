#!/usr/bin/env python3
"""DySurv baseline for MM-GraphSurv.

PyTorch reimplementation adapted from:
    Mesinovic et al., "DySurv: Dynamic survival analysis with conditional
    variational inference."
    Original code: https://github.com/munibmesinovic/DySurv

Architecture:
    1. LSTM encoder → VAE latent space (mu, logvar, reparameterize)
    2. Survival head: FC layers → logistic hazard (discrete time bins)
    3. LSTM decoder: reconstruct input from latent (auxiliary)
    4. Loss: NLL logistic hazard + MSE reconstruction + KL divergence

For competing risks: cause-specific survival heads (one per risk).

Usage:
    python scripts/run_baselines_dysurv.py --dataset eicu
    python scripts/run_baselines_dysurv.py --dataset all

Outputs:
    results/baselines/dysurv_{dataset}.json
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
# DySurv model (PyTorch, adapted from munibmesinovic/DySurv)
# =============================================================================

class DySurv(nn.Module):
    """DySurv: LSTM-VAE with logistic hazard survival head.

    Input: (B, S, F) temporal tensor.
    Output: logits (B, K) for single-risk or (B, K, E) for competing-risk,
            plus decoded reconstruction and VAE parameters.
    """

    def __init__(self, in_features: int, encoded_features: int,
                 out_features: int, seq_len: int, num_risks: int = 1):
        super().__init__()
        self.num_risks = num_risks
        self.in_features = in_features
        self.encoded_features = encoded_features

        # --- Encoder: LSTM → FC → (mu, logvar) ---
        self.lstm_enc = nn.LSTM(in_features, in_features, batch_first=True)
        self.fc_enc1 = nn.Linear(in_features, 3 * in_features)
        self.fc_enc2 = nn.Linear(3 * in_features, 5 * in_features)
        self.fc_enc3 = nn.Linear(5 * in_features, 3 * in_features)
        self.fc_mu = nn.Linear(3 * in_features, encoded_features)
        self.fc_logvar = nn.Linear(3 * in_features, encoded_features)

        # --- Survival head(s) ---
        self.surv_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoded_features, 3 * in_features), nn.ReLU(),
                nn.Linear(3 * in_features, 5 * in_features), nn.ReLU(),
                nn.Linear(5 * in_features, 3 * in_features), nn.ReLU(),
                nn.Linear(3 * in_features, out_features),
            )
            for _ in range(num_risks)
        ])

        # --- Decoder: z → LSTM → reconstruct input ---
        self.decoder_lstm = nn.LSTM(encoded_features, 2 * encoded_features, batch_first=True)
        self.decoder_fc1 = nn.Linear(2 * encoded_features, 3 * encoded_features)
        self.decoder_fc2 = nn.Linear(3 * encoded_features, in_features)
        self.seq_len = seq_len

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def encode(self, x: torch.Tensor):
        """(B, S, F) → mu (B, d), logvar (B, d)."""
        out, _ = self.lstm_enc(x)
        # Mean-pool over time
        h = out.mean(dim=1)  # (B, F)
        h = self.relu(self.fc_enc1(h))
        h = self.relu(self.fc_enc2(h))
        h = self.relu(self.fc_enc3(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """(B, d) → (B, S, F) reconstruction."""
        # Repeat z across time steps
        z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, S, d)
        out, _ = self.decoder_lstm(z_seq)  # (B, S, 2d)
        out = self.dropout(self.decoder_fc1(out))
        out = self.decoder_fc2(out)  # (B, S, F)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Survival predictions
        if self.num_risks == 1:
            phi = self.surv_nets[0](z)  # (B, K)
        else:
            phi = torch.stack([net(z) for net in self.surv_nets], dim=-1)  # (B, K, E)

        # Decoder reconstruction
        x_recon = self.decode(z)  # (B, S, F)

        return phi, x_recon, mu, logvar

    def predict_surv(self, x: torch.Tensor):
        """Predict survival function S(t) from logistic hazard."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if self.num_risks == 1:
            phi = self.surv_nets[0](z)  # (B, K)
            hazard = torch.sigmoid(phi)
            surv = (1 - hazard).cumprod(dim=-1)  # (B, K)
            return surv
        else:
            # Per-risk survival
            survs = []
            for net in self.surv_nets:
                phi = net(z)
                hazard = torch.sigmoid(phi)
                surv = (1 - hazard).cumprod(dim=-1)
                survs.append(surv)
            return torch.stack(survs, dim=-1)  # (B, K, E)


# =============================================================================
# Loss
# =============================================================================

class DySurvLoss(nn.Module):
    """DySurv loss: NLL logistic hazard + MSE reconstruction + KL divergence."""

    def __init__(self, alpha_surv: float = 1.0, alpha_ae: float = 1.0,
                 alpha_kl: float = 0.01, num_risks: int = 1,
                 pos_weight: float | None = None):
        super().__init__()
        self.alpha_surv = alpha_surv
        self.alpha_ae = alpha_ae
        self.alpha_kl = alpha_kl
        self.num_risks = num_risks
        # pos_weight: auto-set based on event rate if None
        self.pos_weight = pos_weight

    def forward(self, phi, x_recon, mu, logvar, x_orig,
                durations_idx, events):
        # --- 1. Survival NLL (logistic hazard) ---
        if self.num_risks == 1:
            loss_surv = self._nll_logistic_hazard(phi, durations_idx, events)
        else:
            loss_surv = self._nll_competing_risk(phi, durations_idx, events)

        # --- 2. Reconstruction MSE ---
        loss_ae = F.mse_loss(x_recon, x_orig)

        # --- 3. KL divergence ---
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return self.alpha_surv * loss_surv + self.alpha_ae * loss_ae + self.alpha_kl * loss_kl

    def _nll_logistic_hazard(self, phi, idx_durations, events):
        """NLL for logistic hazard (single-risk).

        Uses pos_weight to upweight event occurrences, matching the original
        DySurv implementation which uses pos_weight=50 to handle class
        imbalance in the discrete hazard BCE.
        """
        y_bce = torch.zeros_like(phi)
        events_f = events.float().view(-1, 1)
        idx = idx_durations.view(-1, 1).clamp(max=phi.shape[1] - 1)
        y_bce.scatter_(1, idx, events_f)

        pw = torch.tensor([self.pos_weight if self.pos_weight else 1.0], device=phi.device)
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, pos_weight=pw, reduction='none')
        loss = bce.cumsum(1).gather(1, idx).view(-1)
        return loss.mean()

    def _nll_competing_risk(self, phi, idx_durations, events):
        """Cause-specific logistic hazard for competing risks."""
        # phi: (B, K, E)
        B, K, E = phi.shape
        total_loss = torch.tensor(0.0, device=phi.device)
        pw = torch.tensor([self.pos_weight if self.pos_weight else 1.0], device=phi.device)

        for r in range(E):
            phi_r = phi[:, :, r]  # (B, K)
            events_r = (events == (r + 1)).float()
            y_bce = torch.zeros_like(phi_r)
            idx = idx_durations.view(-1, 1).clamp(max=K - 1)
            y_bce.scatter_(1, idx, events_r.view(-1, 1))
            bce = F.binary_cross_entropy_with_logits(phi_r, y_bce, pos_weight=pw, reduction='none')
            loss_r = bce.cumsum(1).gather(1, idx).view(-1).mean()
            total_loss = total_loss + loss_r

        return total_loss / E


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


# DySurv (Mesinovic et al. 2024) processes temporal sequences + static features.
# For 4-modality datasets, use only dynamic + static features (no rad/ICD)
# matching the original paper's design.
TEMPORAL_RANGES = {
    "eicu":  slice(None),      # 2-modal: all features are dynamic+static
    "mimic": slice(0, 71),     # [0:35 dynamic, 35:71 static] — skip rad/ICD
    "mcmed": slice(0, 55),     # [0:32 dynamic, 32:55 static] — skip rad/ICD
    "support": slice(None),    # all static
    "pbc2": slice(None),       # 2-modal: all features are dynamic+static
    "hirid_expanded": slice(None),  # 2-modal: all 73 cols (56 dyn + 17 static)
}


def load_split(processed_dir: Path, name: str, split: str, dataset_key: str = ""):
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


def evaluate_surv(surv_np: np.ndarray, durations: np.ndarray, events: np.ndarray,
                  cuts: np.ndarray, num_risks: int) -> dict:
    time_index = cuts.astype(float)
    results = {}
    ctds, ibss, iblls = [], [], []

    for r in range(num_risks):
        e_label = r + 1
        if num_risks == 1:
            surv_r = surv_np  # (N, K)
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
        else:
            surv_r = surv_np[:, :, r]  # (N, K)
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
                num_risks: int, num_bins: int, seq_len: int,
                encoded_features: int = 20, lr: float = 1e-3,
                weight_decay: float = 1e-4, epochs: int = 100,
                patience: int = 15, batch_size: int = 256,
                device: str = "auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[-1]
    model = DySurv(
        in_features, encoded_features, num_bins, seq_len,
        num_risks=num_risks,
    ).to(device)
    # Auto pos_weight: inverse event rate, capped at 20. Only meaningful
    # for low event rate datasets (eICU 4.6%, MIMIC 5.6%).
    event_rate = (evt_train > 0).mean()
    pw = min(max(1.0 / (event_rate + 1e-3), 1.0), 20.0) if event_rate < 0.3 else 1.0
    print(f"  pos_weight={pw:.1f} (event_rate={event_rate:.3f})")
    criterion = DySurvLoss(num_risks=num_risks, pos_weight=pw)
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
            phi, x_recon, mu, logvar = model(x_t[idx])
            loss = criterion(phi, x_recon, mu, logvar, x_t[idx], d_t[idx], e_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            phi_v, xr_v, mu_v, lv_v = model(x_v)
            val_loss = criterion(phi_v, xr_v, mu_v, lv_v, x_v, d_v, e_v).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}/{val_loss:.4f}  "
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
    parser = argparse.ArgumentParser(description="DySurv baseline")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    # Hyperparameters from DySurv (Mesinovic et al. 2024)
    parser.add_argument("--encoded_features", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
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
        print(f"  DySurv — {ds.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'=' * 65}")

        x_train, dur_train, evt_train, didx_train, cuts = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val, dur_val, evt_val, didx_val, _ = load_split(proc_dir, tname, "val", dataset_key=ds)
        x_test, dur_test, evt_test, didx_test, _ = load_split(proc_dir, tname, "test", dataset_key=ds)
        num_bins = len(cuts)
        seq_len = x_train.shape[1]

        # Per-feature standardization using training stats (essential for RNN stability)
        mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
        std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        std[std < 1e-8] = 1.0
        x_train = (x_train - mu) / std
        x_val = (x_val - mu) / std
        x_test = (x_test - mu) / std

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
        print(f"  Bins (K): {num_bins}  Risks: {num_risks}  Seq len: {seq_len}")

        model, train_info = train_model(
            x_train, didx_train, evt_train,
            x_val, didx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins, seq_len=seq_len,
            encoded_features=args.encoded_features,
            lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size,
        )
        device = next(model.parameters()).device

        results = {"model_info": {"type": "DySurv", **train_info,
                                  "n_features": x_train.shape[-1],
                                  "num_risks": num_risks, "num_bins": num_bins,
                                  "seq_len": seq_len}}

        for split_name, x_s, dur_s, evt_s in [
            ("val", x_val, dur_val, evt_val),
            ("test", x_test, dur_test, evt_test),
        ]:
            with torch.no_grad():
                surv = model.predict_surv(
                    torch.from_numpy(x_s).float().to(device)
                ).cpu().numpy()
            metrics = evaluate_surv(surv, dur_s, evt_s, cuts, num_risks)
            results[split_name] = metrics
            m = metrics.get("mean", metrics.get("risk1", {}))
            print(f"  {split_name}: Ctd={m.get('ctd', 'n/a'):.4f}  "
                  f"IBS={m.get('ibs', 'n/a'):.4f}  IBLL={m.get('ibll', 'n/a'):.4f}")

        out_path = results_dir / f"dysurv_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print(f"\n{'=' * 65}")
    print("  DySurv baseline complete")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
