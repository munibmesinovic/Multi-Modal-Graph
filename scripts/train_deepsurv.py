#!/usr/bin/env python3
"""DeepSurv baseline for MM-GraphSurv — single-risk and cause-specific competing-risk.

Adapted from czifan/DeepSurv.pytorch (https://github.com/czifan/DeepSurv.pytorch).
Uses the Cox partial likelihood loss with a deep MLP, plus a Breslow estimator
to produce survival curves for Ctd / IBS / IBLL evaluation.

For single-risk datasets (eICU, MIMIC, SUPPORT):
    Standard DeepSurv.

For competing-risk datasets (MC-MED, PBC2):
    Cause-specific DeepSurv — one model per risk, treating other events as censored.

Usage:
    python scripts/run_baselines_deepsurv.py --dataset eicu
    python scripts/run_baselines_deepsurv.py --dataset mcmed
    python scripts/run_baselines_deepsurv.py --dataset all

Outputs:
    results/baselines/deepsurv_{dataset}.json — Ctd / IBS / IBLL per split
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
import torch.optim as optim

# ---- Setup path so we can import from src/ ---------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import patch_scipy_simps  # noqa: E402

patch_scipy_simps()

from pycox.evaluation import EvalSurv  # noqa: E402


# ---- DeepSurv model (adapted from czifan/DeepSurv.pytorch) ------------------

class DeepSurv(nn.Module):
    """Deep Cox proportional hazards network.

    MLP that outputs a single scalar risk score per patient.
    Architecture: input → [Linear → BN → ReLU → Dropout] × L → Linear → 1
    """

    def __init__(self, in_features: int, hidden_dims: list[int] = (256, 128),
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns risk score (B, 1)."""
        return self.net(x)


class CoxPHLoss(nn.Module):
    """Negative log partial likelihood for Cox PH.

    Adapted from czifan/DeepSurv.pytorch NegativeLogLikelihood.
    """

    def forward(self, risk_pred: torch.Tensor, durations: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        """
        risk_pred: (B, 1) predicted log-risk
        durations: (B,) event/censor times
        events:    (B,) binary event indicator
        """
        # Sort by duration (descending) for efficient risk set computation
        _, idx = torch.sort(durations, descending=True)
        risk_pred = risk_pred[idx].squeeze()
        events = events[idx].float()

        # Log-sum-exp trick for numerical stability
        # For each event i: log(sum_{j in R_i} exp(h_j)) where R_i = {j : t_j >= t_i}
        # Since sorted descending, R_i is simply all indices from 0 to i
        exp_risk = torch.exp(risk_pred)
        cumsum_exp = torch.cumsum(exp_risk, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-7)

        # Partial likelihood: sum over events of (h_i - log(sum_{j in R_i} exp(h_j)))
        partial_ll = risk_pred - log_cumsum
        loss = -torch.sum(partial_ll * events) / (torch.sum(events) + 1e-7)
        return loss


# ---- Breslow estimator for survival function --------------------------------

def breslow_baseline_hazard(risk_scores: np.ndarray, durations: np.ndarray,
                            events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cumulative baseline hazard using the Breslow estimator.

    Returns (unique_times, cum_baseline_hazard).
    """
    # Sort by time
    idx = np.argsort(durations)
    t = durations[idx]
    e = events[idx]
    h = risk_scores[idx]

    exp_h = np.exp(h)

    unique_times = np.unique(t[e > 0])
    cum_hazard = np.zeros(len(unique_times))

    for i, ti in enumerate(unique_times):
        at_risk = t >= ti
        d_i = np.sum((t == ti) & (e > 0))
        cum_hazard[i] = d_i / (np.sum(exp_h[at_risk]) + 1e-10)

    cum_baseline_hazard = np.cumsum(cum_hazard)
    return unique_times, cum_baseline_hazard


def predict_survival(risk_scores: np.ndarray, baseline_times: np.ndarray,
                     cum_baseline_hazard: np.ndarray,
                     time_grid: np.ndarray) -> pd.DataFrame:
    """Predict survival function S(t|x) = S_0(t)^exp(h(x)) on a time grid.

    Returns (len(time_grid), N) DataFrame suitable for pycox EvalSurv.
    """
    # Interpolate cumulative baseline hazard onto time_grid
    H0 = np.interp(time_grid, baseline_times, cum_baseline_hazard, left=0.0)

    # S(t|x_i) = exp(-H_0(t) * exp(h_i))
    exp_h = np.exp(risk_scores).reshape(1, -1)  # (1, N)
    H0_grid = H0.reshape(-1, 1)  # (T, 1)
    surv = np.exp(-H0_grid * exp_h)  # (T, N)

    return pd.DataFrame(surv, index=time_grid)


# ---- Dataset registry -------------------------------------------------------

DATASETS = {
    "eicu":    {"tensor_name": "eICU",    "num_risks": 1},
    "mimic":   {"tensor_name": "MIMIC",   "num_risks": 1},
    "support": {"tensor_name": "SUPPORT", "num_risks": 1},
    "mcmed":   {"tensor_name": "MCMED",   "num_risks": 2},
    "pbc2":    {"tensor_name": "PBC2",    "num_risks": 2},
}


# ---- Data loading -----------------------------------------------------------

# DeepSurv (Katzman et al. 2018) is a static-input model. For large datasets,
# use only static demographics mean-pooled across time to match the original
# paper's design and avoid feature explosion from flattening temporal data.
STATIC_RANGES = {
    "eicu":  slice(0, 62),    # demographics only (62 cols)
    "mimic": slice(35, 71),   # static demographics only (36 cols)
    "mcmed": slice(32, 55),   # static demographics only (23 cols)
    "hirid": slice(18, 35),   # static only (17 cols) — age+sex+height+APACHE (D72)
    "hirid_circ": slice(18, 35),  # same feature layout as hirid mortality
    "support": None,          # all static (14 features, s=1)
    "pbc2": slice(11, 15),    # static only: drug, age, sex, histologic (4 cols)
}


def load_split(processed_dir: Path, name: str, split: str, dataset_key: str = ""):
    """Load preprocessed tensors and return flat (N, F) array + labels."""
    x = np.load(processed_dir / f"x_{split}_{name}.npy")
    with open(processed_dir / f"y_{split}_surv_{name}.p", "rb") as f:
        durations_raw, events = pickle.load(f)

    static_range = STATIC_RANGES.get(dataset_key)
    if static_range is not None:
        x_flat = x[:, :, static_range].mean(axis=1).astype(np.float32)
    else:
        n, s, feat = x.shape
        x_flat = x.reshape(n, s * feat).astype(np.float32)

    events = events.astype(np.int64)
    # MC-MED ships raw 4-class labels {0,1,2,3}; collapse {3 -> 2} per
    # configs/mcmed.yaml:event_collapse so the 2-risk cohort matches the
    # MMG/DeepHit/DD/DySurv evaluation. Mirrors calibrate.py:_maybe_remap_events.
    if dataset_key == "mcmed":
        events[events == 3] = 2

    return x_flat, durations_raw.astype(np.float32), events


def _remove_constant_cols(x_train, x_val, x_test):
    """Drop features with zero variance in training set."""
    std = x_train.std(axis=0)
    keep = std > 1e-10
    return x_train[:, keep], x_val[:, keep], x_test[:, keep], keep


# ---- Evaluation -------------------------------------------------------------

def evaluate_survival_df(surv_df: pd.DataFrame, durations: np.ndarray,
                         events_binary: np.ndarray) -> dict:
    """Compute Ctd, IBS, IBLL using pycox EvalSurv."""
    ev = EvalSurv(surv_df, durations, events_binary, censor_surv="km")
    ctd = ev.concordance_td()
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    ibll = ev.integrated_nbll(time_grid)
    return {"ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll)}


# ---- Training ----------------------------------------------------------------

def train_deepsurv(x_train: np.ndarray, dur_train: np.ndarray, evt_train: np.ndarray,
                   x_val: np.ndarray, dur_val: np.ndarray, evt_val: np.ndarray,
                   hidden_dims: list[int] = (256, 128),
                   dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-4,
                   epochs: int = 100, patience: int = 15, batch_size: int = 256,
                   device: str = "auto") -> tuple[DeepSurv, dict]:
    """Train a DeepSurv model with early stopping on validation c-index."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[1]
    model = DeepSurv(in_features, hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = CoxPHLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Convert to tensors
    x_t = torch.from_numpy(x_train).float().to(device)
    d_t = torch.from_numpy(dur_train).float().to(device)
    e_t = torch.from_numpy(evt_train).float().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    d_v = torch.from_numpy(dur_val).float().to(device)
    e_v = torch.from_numpy(evt_val).float().to(device)

    best_val_ci = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "train_ci": [], "val_ci": []}

    n_train = len(x_train)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 8:
                continue
            risk = model(x_t[idx])
            loss = criterion(risk, d_t[idx], e_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_risk = model(x_v).squeeze().cpu().numpy()
            val_loss = criterion(model(x_v), d_v, e_v).item()
            # c-index: lower risk score = higher survival → negate for concordance
            from lifelines.utils import concordance_index
            val_ci = concordance_index(dur_val, -val_risk, evt_val)

            train_risk = model(x_t).squeeze().cpu().numpy()
            train_ci = concordance_index(dur_train, -train_risk, evt_train)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_ci"].append(train_ci)
        history["val_ci"].append(val_ci)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_train_loss:.4f}/{val_loss:.4f}  "
                  f"CI={train_ci:.4f}/{val_ci:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best val CI={best_val_ci:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    info = {
        "best_val_ci": float(best_val_ci),
        "epochs_run": epoch,
        "final_train_ci": float(train_ci),
    }
    return model, info


# ---- Single-risk runner ------------------------------------------------------

def run_single_risk(x_train, dur_train, evt_train,
                    x_val, dur_val, evt_val,
                    x_test, dur_test, evt_test,
                    **train_kwargs) -> dict:
    """Train DeepSurv and evaluate with full survival curves."""
    print(f"  Training DeepSurv: {x_train.shape[0]} patients, {x_train.shape[1]} features")
    model, train_info = train_deepsurv(
        x_train, dur_train, evt_train,
        x_val, dur_val, evt_val,
        **train_kwargs,
    )
    device = next(model.parameters()).device

    # Get risk scores on train (for Breslow), val, test
    with torch.no_grad():
        h_train = model(torch.from_numpy(x_train).float().to(device)).squeeze().cpu().numpy()
        h_val = model(torch.from_numpy(x_val).float().to(device)).squeeze().cpu().numpy()
        h_test = model(torch.from_numpy(x_test).float().to(device)).squeeze().cpu().numpy()

    # Breslow baseline hazard from training data
    base_times, cum_H0 = breslow_baseline_hazard(h_train, dur_train, evt_train)

    results = {}
    for split_name, h_s, dur_s, evt_s in [
        ("val", h_val, dur_val, evt_val),
        ("test", h_test, dur_test, evt_test),
    ]:
        time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
        surv_df = predict_survival(h_s, base_times, cum_H0, time_grid)
        metrics = evaluate_survival_df(surv_df, dur_s, (evt_s > 0).astype(int))
        results[split_name] = {"risk1": metrics}
        print(f"  {split_name}: Ctd={metrics['ctd']:.4f}  IBS={metrics['ibs']:.4f}  "
              f"IBLL={metrics['ibll']:.4f}")

    results["model_info"] = {
        "type": "DeepSurv",
        **train_info,
        "n_features": x_train.shape[1],
    }
    return results


# ---- Competing-risk cause-specific runner ------------------------------------

def run_competing_risk(x_train, dur_train, evt_train,
                       x_val, dur_val, evt_val,
                       x_test, dur_test, evt_test,
                       num_risks: int,
                       **train_kwargs) -> dict:
    """Cause-specific DeepSurv: one model per risk, others treated as censored."""
    results = {"val": {}, "test": {}}
    val_ctds, test_ctds = [], []

    for r in range(1, num_risks + 1):
        print(f"\n  --- Risk {r}/{num_risks} ---")
        evt_train_r = (evt_train == r).astype(np.int64)
        evt_val_r = (evt_val == r).astype(np.int64)
        evt_test_r = (evt_test == r).astype(np.int64)

        n_events = evt_train_r.sum()
        print(f"  Training events for risk {r}: {n_events}/{len(evt_train_r)} "
              f"({n_events / len(evt_train_r) * 100:.1f}%)")

        if n_events < 10:
            print(f"  Skipping risk {r}: too few events ({n_events})")
            for s in ["val", "test"]:
                results[s][f"risk{r}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
            continue

        model, train_info = train_deepsurv(
            x_train, dur_train, evt_train_r,
            x_val, dur_val, evt_val_r,
            **train_kwargs,
        )
        device = next(model.parameters()).device

        with torch.no_grad():
            h_train = model(torch.from_numpy(x_train).float().to(device)).squeeze().cpu().numpy()
            h_val = model(torch.from_numpy(x_val).float().to(device)).squeeze().cpu().numpy()
            h_test = model(torch.from_numpy(x_test).float().to(device)).squeeze().cpu().numpy()

        base_times, cum_H0 = breslow_baseline_hazard(h_train, dur_train, evt_train_r)

        for split_name, h_s, dur_s, evt_s_r in [
            ("val", h_val, dur_val, evt_val_r),
            ("test", h_test, dur_test, evt_test_r),
        ]:
            if evt_s_r.sum() < 5:
                print(f"  {split_name} risk{r}: skipped (only {evt_s_r.sum()} events)")
                results[split_name][f"risk{r}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
                continue
            time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
            surv_df = predict_survival(h_s, base_times, cum_H0, time_grid)
            metrics = evaluate_survival_df(surv_df, dur_s, evt_s_r)
            results[split_name][f"risk{r}"] = metrics
            print(f"  {split_name} risk{r}: Ctd={metrics['ctd']:.4f}  "
                  f"IBS={metrics['ibs']:.4f}  IBLL={metrics['ibll']:.4f}")
            if split_name == "val":
                val_ctds.append(metrics["ctd"])
            else:
                test_ctds.append(metrics["ctd"])

    # Mean across risks
    for split_name, ctds in [("val", val_ctds), ("test", test_ctds)]:
        valid_risks = [f"risk{r+1}" for r in range(num_risks)
                       if not np.isnan(results[split_name].get(f"risk{r+1}", {}).get("ctd", float("nan")))]
        results[split_name]["mean"] = {
            "ctd": float(np.mean([results[split_name][k]["ctd"] for k in valid_risks])) if valid_risks else float("nan"),
            "ibs": float(np.mean([results[split_name][k]["ibs"] for k in valid_risks])) if valid_risks else float("nan"),
            "ibll": float(np.mean([results[split_name][k]["ibll"] for k in valid_risks])) if valid_risks else float("nan"),
        }

    results["model_info"] = {
        "type": "CauseSpecificDeepSurv",
        "num_risks": num_risks,
        "n_features": x_train.shape[1],
    }
    return results


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSurv baseline for MM-GraphSurv")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    # Hyperparameters following Katzman et al. DeepSurv:
    # Original uses SGD+Nesterov, we use Adam (standard PyTorch reimpl).
    # lr=1e-4, moderate network, dropout=0.4
    parser.add_argument("--hidden_dims", type=str, default="100,100",
                        help="Comma-separated hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]
    train_kwargs = dict(
        hidden_dims=hidden_dims, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
    )

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
        print(f"  DeepSurv baseline — {ds.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'=' * 65}")

        x_train, dur_train, evt_train = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val, dur_val, evt_val = load_split(proc_dir, tname, "val", dataset_key=ds)
        x_test, dur_test, evt_test = load_split(proc_dir, tname, "test", dataset_key=ds)

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")

        x_train, x_val, x_test, keep_mask = _remove_constant_cols(x_train, x_val, x_test)
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} constant features → {x_train.shape[1]} remaining")

        if num_risks == 1:
            results = run_single_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                **train_kwargs,
            )
        else:
            results = run_competing_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                num_risks=num_risks,
                **train_kwargs,
            )

        out_path = results_dir / f"deepsurv_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print(f"\n{'=' * 65}")
    print("  DeepSurv baseline complete")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
