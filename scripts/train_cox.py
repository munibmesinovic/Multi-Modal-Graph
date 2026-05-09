#!/usr/bin/env python3
"""Cox PH baseline for MM-GraphSurv — single-risk and cause-specific competing-risk.

For single-risk datasets (eICU, MIMIC, SUPPORT):
    Standard Cox PH via lifelines.CoxPHFitter.

For competing-risk datasets (MC-MED, PBC2):
    Cause-specific Cox PH — one model per risk, treating other events as censored.

Usage:
    python scripts/run_baselines_cox.py --dataset eicu
    python scripts/run_baselines_cox.py --dataset mcmed
    python scripts/run_baselines_cox.py --dataset all

Outputs:
    results/baselines/cox_{dataset}.json     — Ctd / IBS / IBLL per split
    results/baselines/cox_{dataset}_surv.npy — survival predictions on test set
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

# ---- Setup path so we can import from src/ ---------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import patch_scipy_simps  # noqa: E402

patch_scipy_simps()

from lifelines import CoxPHFitter  # noqa: E402
from pycox.evaluation import EvalSurv  # noqa: E402


# ---- Dataset registry -------------------------------------------------------

DATASETS = {
    "eicu":    {"tensor_name": "eICU",    "num_risks": 1},
    "mimic":   {"tensor_name": "MIMIC",   "num_risks": 1},
    "support": {"tensor_name": "SUPPORT", "num_risks": 1},
    "mcmed":   {"tensor_name": "MCMED",   "num_risks": 2},
    "pbc2":    {"tensor_name": "PBC2",    "num_risks": 2},
}


# ---- Data loading -----------------------------------------------------------

# Static feature ranges — Cox PH uses only static demographics for large
# datasets (clinical convention: admission data only for Cox regression).
# Verified against feature_names_*.p pickles:
#   eICU:  [0:62]  demographics, [62:93] labs/vitals
#   MIMIC: [0:35]  dynamic, [35:71] static demographics, [71:839] rad, [839:1339] ICD
#   MCMED: [0:32]  dynamic, [32:55] static demographics, [55:823] rad, [823:1323] ICD
STATIC_RANGES = {
    "eicu":  slice(0, 62),    # demographics only (62 cols)
    "mimic": slice(35, 71),   # static demographics only (36 cols) — NOT rad/ICD
    "mcmed": slice(32, 55),   # static demographics only (23 cols) — NOT rad/ICD
    "hirid": slice(18, 35),   # static only (17 cols) — age+sex+height+APACHE (D72)
    "hirid_circ": slice(18, 35),  # same feature layout as hirid mortality
    "support": None,          # all static (14 features, s=1)
    "pbc2": slice(11, 15),    # static only: drug, age, sex, histologic (4 cols)
}


def load_split(processed_dir: Path, name: str, split: str, dataset_key: str = ""):
    """Load preprocessed tensors for Cox PH.

    For large datasets, uses only static features mean-pooled across time.
    For small datasets, flattens all features.
    """
    x = np.load(processed_dir / f"x_{split}_{name}.npy")  # (N, S, F)
    with open(processed_dir / f"y_{split}_surv_{name}.p", "rb") as f:
        durations_raw, events = pickle.load(f)

    static_range = STATIC_RANGES.get(dataset_key)
    if static_range is not None:
        x_flat = x[:, :, static_range].mean(axis=1).astype(np.float64)
    else:
        n, s, feat = x.shape
        x_flat = x.reshape(n, s * feat).astype(np.float64)

    durations = durations_raw.astype(np.float64)
    events = events.astype(np.int64)

    # MC-MED ships raw 4-class labels {0,1,2,3}; collapse {3 -> 2} per
    # configs/mcmed.yaml:event_collapse so the 2-risk cohort matches the
    # MMG/DeepHit/DD/DySurv evaluation. Mirrors calibrate.py:_maybe_remap_events.
    if dataset_key == "mcmed":
        events[events == 3] = 2

    return x_flat, durations, events


def _remove_constant_cols(x_train, x_val, x_test):
    """Drop features with zero variance in training set (Cox PH will fail on them)."""
    std = x_train.std(axis=0)
    keep = std > 1e-10
    return x_train[:, keep], x_val[:, keep], x_test[:, keep], keep


# ---- Evaluation -------------------------------------------------------------

def evaluate_survival(surv_df: pd.DataFrame, durations: np.ndarray,
                      events_binary: np.ndarray) -> dict:
    """Compute Ctd, IBS, IBLL using pycox EvalSurv."""
    ev = EvalSurv(surv_df, durations, events_binary, censor_surv="km")
    ctd = ev.concordance_td()
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    ibll = ev.integrated_nbll(time_grid)
    return {"ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll)}


def _fit_coxph(x_train, dur_train, evt_binary, penalizer=0.01):
    """Fit standard Cox PH via lifelines."""
    n_features = x_train.shape[1]
    col_names = [f"f{i}" for i in range(n_features)]
    df_train = pd.DataFrame(x_train, columns=col_names)
    df_train["duration"] = dur_train
    df_train["event"] = evt_binary

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df_train, duration_col="duration", event_col="event",
            show_progress=True)
    return cph, col_names


def _coxph_survival(model, x_eval, col_names, time_grid):
    """Predict survival curves from a fitted CoxPH model."""
    df = pd.DataFrame(x_eval, columns=col_names)
    surv = model.predict_survival_function(df, times=time_grid)
    return surv


# ---- Single-risk Cox PH -----------------------------------------------------

def run_single_risk(x_train, dur_train, evt_train,
                    x_val, dur_val, evt_val,
                    x_test, dur_test, evt_test,
                    penalizer: float = 0.01) -> dict:
    """Fit standard Cox PH and evaluate on val + test."""
    n_features = x_train.shape[1]

    print(f"  Fitting Cox PH: {len(x_train)} patients, {n_features} features, "
          f"penalizer={penalizer} ...")
    t0 = time.time()
    model, col_names = _fit_coxph(x_train, dur_train, (evt_train > 0).astype(int), penalizer)
    print(f"  Fitted in {time.time() - t0:.1f}s  (concordance_train={model.concordance_index_:.4f})")

    results = {}
    for split_name, x_s, dur_s, evt_s in [
        ("val", x_val, dur_val, evt_val),
        ("test", x_test, dur_test, evt_test),
    ]:
        time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
        surv = _coxph_survival(model, x_s, col_names, time_grid)
        metrics = evaluate_survival(surv, dur_s, (evt_s > 0).astype(int))
        results[split_name] = {"risk1": metrics}
        print(f"  {split_name}: Ctd={metrics['ctd']:.4f}  IBS={metrics['ibs']:.4f}  "
              f"IBLL={metrics['ibll']:.4f}")

    results["model_info"] = {
        "type": "CoxPH",
        "penalizer": penalizer,
        "n_features": n_features,
        "train_concordance": float(model.concordance_index_),
    }
    return results


# ---- Competing-risk cause-specific Cox PH ------------------------------------

def run_competing_risk(x_train, dur_train, evt_train,
                       x_val, dur_val, evt_val,
                       x_test, dur_test, evt_test,
                       num_risks: int,
                       penalizer: float = 0.01) -> dict:
    """Cause-specific Cox PH: one model per risk, others censored."""
    n_features = x_train.shape[1]
    results = {"val": {}, "test": {}}
    val_ctds, test_ctds = [], []

    for r in range(1, num_risks + 1):
        print(f"\n  --- Risk {r}/{num_risks} ---")
        evt_train_r = (evt_train == r).astype(int)
        evt_val_r = (evt_val == r).astype(int)
        evt_test_r = (evt_test == r).astype(int)

        n_events = evt_train_r.sum()
        print(f"  Training events for risk {r}: {n_events}/{len(evt_train_r)} "
              f"({n_events/len(evt_train_r)*100:.1f}%)")

        if n_events < 10:
            print(f"  Skipping risk {r}: too few events ({n_events})")
            for s in ["val", "test"]:
                results[s][f"risk{r}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
            continue

        t0 = time.time()
        model, col_names = _fit_coxph(x_train, dur_train, evt_train_r, penalizer)
        print(f"  Fitted in {time.time() - t0:.1f}s  (concordance_train={model.concordance_index_:.4f})")

        for split_name, x_s, dur_s, evt_s_r in [
            ("val", x_val, dur_val, evt_val_r),
            ("test", x_test, dur_test, evt_test_r),
        ]:
            if evt_s_r.sum() < 5:
                print(f"  {split_name} risk{r}: skipped (only {evt_s_r.sum()} events)")
                results[split_name][f"risk{r}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
                continue
            time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
            surv = _coxph_survival(model, x_s, col_names, time_grid)
            metrics = evaluate_survival(surv, dur_s, evt_s_r)
            results[split_name][f"risk{r}"] = metrics
            print(f"  {split_name} risk{r}: Ctd={metrics['ctd']:.4f}  "
                  f"IBS={metrics['ibs']:.4f}  IBLL={metrics['ibll']:.4f}")
            if split_name == "val":
                val_ctds.append(metrics["ctd"])
            else:
                test_ctds.append(metrics["ctd"])

    for split_name, ctds in [("val", val_ctds), ("test", test_ctds)]:
        valid_risks = [f"risk{r+1}" for r in range(num_risks)
                       if not np.isnan(results[split_name].get(f"risk{r+1}", {}).get("ctd", float("nan")))]
        results[split_name]["mean"] = {
            "ctd": float(np.mean([results[split_name][k]["ctd"] for k in valid_risks])) if valid_risks else float("nan"),
            "ibs": float(np.mean([results[split_name][k]["ibs"] for k in valid_risks])) if valid_risks else float("nan"),
            "ibll": float(np.mean([results[split_name][k]["ibll"] for k in valid_risks])) if valid_risks else float("nan"),
        }

    results["model_info"] = {
        "type": "CauseSpecificCoxPH",
        "penalizer": penalizer,
        "n_features": n_features,
        "num_risks": num_risks,
    }
    return results


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cox PH baseline for MM-GraphSurv")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    parser.add_argument("--penalizer", type=float, default=0.01,
                        help="L2 penalizer for Cox PH (default 0.01)")
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

        print(f"\n{'='*65}")
        print(f"  Cox PH baseline — {ds.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'='*65}")

        # Load data
        x_train, dur_train, evt_train = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val, dur_val, evt_val = load_split(proc_dir, tname, "val", dataset_key=ds)
        x_test, dur_test, evt_test = load_split(proc_dir, tname, "test", dataset_key=ds)

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean()*100:.1f}%")

        # Drop constant features (Cox PH will fail on zero-variance columns)
        x_train, x_val, x_test, keep_mask = _remove_constant_cols(x_train, x_val, x_test)
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} constant features → {x_train.shape[1]} remaining")

        # Run
        if num_risks == 1:
            results = run_single_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                penalizer=args.penalizer,
            )
        else:
            results = run_competing_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                num_risks=num_risks,
                penalizer=args.penalizer,
            )

        # Save
        out_path = results_dir / f"cox_{ds}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print(f"\n{'='*65}")
    print("  Cox PH baseline complete")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
