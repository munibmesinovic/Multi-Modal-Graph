#!/usr/bin/env python3
"""SurvTRACE baseline for MM-GraphSurv.

SurvTRACE (Wang & Sun, arXiv:2110.00855, 2021) is a static-input survival model
that uses BERT-style cross-feature attention. Source code:
    https://github.com/RyanWangZf/SurvTRACE

This trainer mirrors the DeepHit baseline pattern exactly so SurvTRACE results
are directly comparable: same load_split, same mean-pool/flatten of dynamic
features, same MC-MED 2-risk collapse, same pycox EvalSurv evaluation. The
only differences are:
  - SurvTRACE's transformer architecture (defaults from STConfig)
  - PC-Hazard loss (NLLPCHazardLoss) — requires interval fraction "proportion"
  - Use SurvTRACE's LabelTransform to compute (idx_durations, events, t_frac)
    against OUR cuts so the time grid matches DeepHit's.

The SurvTRACE code is expected via the SURVTRACE_ROOT environment variable
pointing at a clone of https://github.com/RyanWangZf/SurvTRACE.

Default hyperparameters follow the SurvTRACE METABRIC example notebook
(experiment_metabric.ipynb): batch_size=64, lr=1e-3, weight_decay=1e-4,
epochs=20, early_stop_patience=5. Architecture defaults from STConfig:
hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
intermediate_size=64, dropout=0.0, attention_probs_dropout_prob=0.1.

Usage:
    python scripts/train_survtrace.py --dataset eicu --seed 42
    python scripts/train_survtrace.py --dataset mcmed --seed 42
    python scripts/train_survtrace.py --dataset all --seed 42

Outputs:
    checkpoints/{ds}_survtrace_seed{S}/best_model.pth
    checkpoints/{ds}_survtrace_seed{S}/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Add the SurvTRACE clone to the Python path. The location is resolved from
# the SURVTRACE_ROOT env var so this tree contains no paths outside itself.
import os
_st_root = os.environ.get("SURVTRACE_ROOT")
if not _st_root:
    raise RuntimeError(
        "SURVTRACE_ROOT env var is not set. Clone "
        "https://github.com/RyanWangZf/SurvTRACE and export "
        "SURVTRACE_ROOT=/path/to/SurvTRACE before running.")
SURVTRACE_ROOT = Path(_st_root)
if not SURVTRACE_ROOT.exists():
    raise RuntimeError(f"SURVTRACE_ROOT={SURVTRACE_ROOT} does not exist.")
sys.path.insert(0, str(SURVTRACE_ROOT))

from src.utils import patch_scipy_simps  # noqa: E402
patch_scipy_simps()

# SurvTRACE was written against numpy 1.x; restore the removed alias before
# importing their modules so train_utils.py:32 (`np.Inf`) keeps working.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from pycox.evaluation import EvalSurv  # noqa: E402

# SurvTRACE imports — these come from the cloned third-party repo
from survtrace.config import STConfig  # noqa: E402
from survtrace.model import SurvTraceSingle, SurvTraceMulti  # noqa: E402
from survtrace.train_utils import Trainer  # noqa: E402
from survtrace.utils import LabelTransform, set_random_seed  # noqa: E402


# =============================================================================
# Dataset registry — mirrors train_deephit.py exactly. MC-MED is 2-risk after
# the {3 -> 2} collapse applied below.
# =============================================================================

DATASETS = {
    "eicu":           {"tensor_name": "eICU",    "num_risks": 1},
    "mimic":          {"tensor_name": "MIMIC",   "num_risks": 1},
    "mcmed":          {"tensor_name": "MCMED",   "num_risks": 2},
    "hirid":          {"tensor_name": "HIRID",   "num_risks": 1},
    "hirid_circ":     {"tensor_name": "HIRID",   "num_risks": 1},
    "hirid_expanded": {"tensor_name": "HIRID",   "num_risks": 1},
    "support":        {"tensor_name": "SUPPORT", "num_risks": 1},
    "pbc2":           {"tensor_name": "PBC2",    "num_risks": 2},
}

# Same static ranges DeepHit uses — guarantees apples-to-apples input shape.
STATIC_RANGES = {
    "eicu":           slice(0, 62),
    "mimic":          slice(35, 71),
    "mcmed":          slice(32, 55),
    "hirid":          slice(18, 35),
    "hirid_circ":     slice(18, 35),
    "hirid_expanded": slice(56, 73),
    "support":        None,
    "pbc2":           slice(11, 15),
}


# =============================================================================
# Data loading — identical to train_deephit.py except the MC-MED collapse
# happens here so the rest of the pipeline never sees event=3.
# =============================================================================

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

    # MC-MED 2-risk collapse: events {0=cens, 1=ICU, 2=Hosp, 3=Obs}
    # -> {0=cens, 1=ICU, 2=Hosp∪Obs} so all 5 baselines see the same labels.
    if dataset_key == "mcmed":
        events = events.copy()
        events[events == 3] = 2

    return x_flat, durations, events, cuts


def _remove_constant_cols(x_train, x_val, x_test):
    std = x_train.std(axis=0)
    keep = std > 1e-10
    return x_train[:, keep], x_val[:, keep], x_test[:, keep], keep


# =============================================================================
# Label construction — wrap our continuous (durations, events, cuts) into
# SurvTRACE's expected DataFrame format.
# =============================================================================

def _make_label_df(durations: np.ndarray, events: np.ndarray,
                   labtrans: LabelTransform, num_risks: int) -> pd.DataFrame:
    """Transform continuous (durations, events) into SurvTRACE's label DataFrame.

    Single-risk:   columns = [duration, event, proportion]
    Competing-risk: columns = [duration, event_0, ..., event_{R-1}, proportion]
    """
    # LabelTransform.transform returns (idx_durations, events_filtered, t_frac).
    # NOTE: idx_durations is the discrete bin index (after their internal -1
    # adjustment). events_filtered is the events array as-is, possibly with
    # entries at bin 0 zeroed out by their boundary handling.
    idx, ev_lt, t_frac = labtrans.transform(durations.astype(np.float64),
                                            events.astype(np.float64))

    if num_risks == 1:
        # Treat any non-zero event as event=1 (single-risk binary).
        return pd.DataFrame({
            "duration": idx.astype(np.int64),
            "event":    (ev_lt > 0).astype(np.int64),
            "proportion": t_frac.astype(np.float32),
        })

    # Competing-risk: one event_k column per risk k, set 1 if event matches.
    # We use `ev_lt` (post-LabelTransform), NOT raw `events`, so that samples
    # whose event landed at bin 0 are treated as censored — matching what
    # LabelTransform.transform() does internally for the loss.
    df = pd.DataFrame({"duration": idx.astype(np.int64)})
    for r in range(num_risks):
        df[f"event_{r}"] = (ev_lt == float(r + 1)).astype(np.int64)
    df["proportion"] = t_frac.astype(np.float32)
    return df


def _make_feature_df(x: np.ndarray) -> pd.DataFrame:
    """Wrap an (N, F) feature matrix into a SurvTRACE-style DataFrame.

    SurvTRACE expects categorical columns first, numerical after. Our
    preprocessed features are all continuous (z-scored), so we prepend
    one dummy categorical column of zeros (vocab_size=1) — required because
    SurvTRACE's BertEmbeddings.forward calls `word_embeddings(input_ids)`
    and would fail with an empty `input_ids` tensor.
    """
    n, f = x.shape
    cols = ["_cat_dummy"] + [f"num_{i}" for i in range(f)]
    df = pd.DataFrame(np.zeros((n, f + 1), dtype=np.float32), columns=cols)
    df.iloc[:, 0] = 0  # categorical column — single value, vocab_size=1
    df.iloc[:, 1:] = x
    return df


# =============================================================================
# Configuration — apply SurvTRACE defaults + our cuts.
# =============================================================================

def build_st_config(num_features: int, num_risks: int, cuts: np.ndarray,
                    checkpoint_path: Path) -> dict:
    """Build the STConfig dict for one (dataset, seed) run."""
    # STConfig is an EasyDict; deepcopy so we don't mutate the package global.
    cfg = deepcopy(STConfig)

    # Architecture defaults (unchanged from STConfig source) — this is what
    # "default hyperparams" means for SurvTRACE:
    #   hidden_size=16, intermediate_size=64, num_hidden_layers=3,
    #   num_attention_heads=2, hidden_dropout_prob=0.0,
    #   attention_probs_dropout_prob=0.1, early_stop_patience=5,
    #   initializer_range=0.001, layer_norm_eps=1e-12

    # Data-derived fields normally set by SurvTRACE's load_data():
    cfg["num_categorical_feature"] = 1   # one dummy column (vocab_size=1)
    cfg["num_numerical_feature"] = int(num_features)
    cfg["num_feature"] = int(num_features) + 1
    cfg["vocab_size"] = 1
    cfg["num_event"] = int(num_risks)
    cfg["duration_index"] = np.asarray(cuts, dtype=np.float64)
    cfg["out_feature"] = int(len(cuts) - 1)  # = labtrans.out_features
    cfg["horizons"] = [0.25, 0.5, 0.75]      # only used by SurvTRACE's own evaluator
    cfg["checkpoint"] = str(checkpoint_path)

    # Their model.init_weights() reads cfg.pruned_heads / tie_word_embeddings —
    # the package defaults already cover these. Ensure they exist.
    cfg.setdefault("pruned_heads", {})
    cfg.setdefault("tie_word_embeddings", True)
    return cfg


# =============================================================================
# Evaluation — pycox EvalSurv (matches DeepHit / Cox / DeepSurv exactly).
# =============================================================================

def evaluate_survtrace(model, x_test_df: pd.DataFrame, durations: np.ndarray,
                       events: np.ndarray, cuts: np.ndarray,
                       num_risks: int, batch_size: int = 1024) -> dict:
    """Evaluate via pycox EvalSurv on the test set.

    For multi-event, calls predict_surv(event=k) once per risk.
    """
    time_index = cuts.astype(float)
    results = {}
    ctds, ibss, iblls = [], [], []

    for r in range(num_risks):
        e_label = r + 1  # our label encoding: 1..num_risks
        if num_risks == 1:
            surv_t = model.predict_surv(x_test_df, batch_size=batch_size)
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
        else:
            surv_t = model.predict_surv(x_test_df, batch_size=batch_size, event=r)
            mask = (events == 0) | (events == e_label)
            if mask.sum() < 10 or (events[mask] == e_label).sum() < 5:
                results[f"risk{e_label}"] = {
                    "ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")
                }
                continue
            events_bin = (events[mask] == e_label).astype(int)

        surv = surv_t.detach().cpu().numpy()  # (N, K)
        # SurvTRACE pads a leading column of zeros in predict_hazard → surv has
        # shape (N, K) where K = len(cuts). The index = duration_index = cuts.
        if surv.shape[1] != len(time_index):
            # Defensive: should be equal; if off-by-one, align by truncation.
            k = min(surv.shape[1], len(time_index))
            surv = surv[:, :k]
            t_idx = time_index[:k]
        else:
            t_idx = time_index

        surv_df = pd.DataFrame(surv[mask].T, index=t_idx)
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
            ctds.append(ctd); ibss.append(ibs); iblls.append(ibll)

    results["mean"] = {
        "ctd":  float(np.mean(ctds))  if ctds  else float("nan"),
        "ibs":  float(np.mean(ibss))  if ibss  else float("nan"),
        "ibll": float(np.mean(iblls)) if iblls else float("nan"),
    }
    return results


# =============================================================================
# Train — wraps SurvTRACE's Trainer.fit and returns the best model.
# =============================================================================

def train_survtrace(x_train, dur_train, evt_train,
                    x_val,   dur_val,   evt_val,
                    cuts: np.ndarray, num_risks: int,
                    checkpoint_path: Path,
                    batch_size: int, epochs: int, lr: float, weight_decay: float,
                    early_stop_patience: int = 15):
    """Build STConfig, label DataFrames, model, train, return (model, info)."""
    # 1. Discretization shared across train/val (fit on train durations/events).
    labtrans = LabelTransform(cuts=np.asarray(cuts, dtype=np.float64))
    labtrans.fit(dur_train.astype(np.float64), evt_train.astype(np.float64))

    # 2. Build label DataFrames.
    df_y_train = _make_label_df(dur_train, evt_train, labtrans, num_risks)
    df_y_val   = _make_label_df(dur_val,   evt_val,   labtrans, num_risks)

    # 3. Feature DataFrames (categorical dummy + numerical).
    df_train = _make_feature_df(x_train)
    df_val   = _make_feature_df(x_val)

    # 4. Configure SurvTRACE.
    cfg = build_st_config(num_features=x_train.shape[1], num_risks=num_risks,
                           cuts=cuts, checkpoint_path=checkpoint_path)
    cfg["early_stop_patience"] = int(early_stop_patience)

    # 5. Instantiate the right model variant.
    if num_risks == 1:
        model = SurvTraceSingle(cfg)
    else:
        model = SurvTraceMulti(cfg)

    # 6. Train. Trainer auto-moves model to GPU if available.
    trainer = Trainer(model)
    t0 = time.time()
    train_loss, val_loss = trainer.fit(
        (df_train, df_y_train), (df_val, df_y_val),
        batch_size=batch_size, epochs=epochs,
        learning_rate=lr, weight_decay=weight_decay,
    )
    elapsed = time.time() - t0

    # Trainer.fit loads the best checkpoint into the model on early stop;
    # if no early stop fired, manually load if the file exists.
    if Path(checkpoint_path).exists():
        try:
            model.load_state_dict(torch.load(str(checkpoint_path),
                                             map_location="cpu"))
            if torch.cuda.is_available():
                model.cuda()
                model.use_gpu = True
        except Exception as exc:
            print(f"  WARN: could not reload best ckpt: {exc}")

    info = {
        "train_loss_per_epoch": [float(x) for x in train_loss],
        "val_loss_per_epoch":   [float(x) for x in val_loss],
        "best_val_loss": float(min(val_loss)) if val_loss else float("nan"),
        "epochs_run": len(train_loss),
        "elapsed_seconds": float(elapsed),
    }
    return model, df_train, df_val, info


# =============================================================================
# Main — per-dataset, per-seed loop matching train_deephit.py's CLI surface.
# =============================================================================

def _set_seed(seed: int):
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _proc_dir_for(ds: str, seed: int) -> Path:
    """v2 uses `processed_seed{S}/` per dataset; fall back to `processed/`."""
    per_seed = ROOT / "data" / ds / f"processed_seed{seed}"
    if per_seed.exists():
        return per_seed
    return ROOT / "data" / ds / "processed"


def main():
    parser = argparse.ArgumentParser(description="SurvTRACE baseline for MM-GraphSurv")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"])
    parser.add_argument("--seed", type=int, default=42)
    # Architecture defaults from STConfig (paper search-range middle/top values).
    # Training length is bumped from notebook defaults (epochs=20, patience=5)
    # to match our other baselines (DeepHit/Cox/DeepSurv use 100/15) — the
    # notebook defaults are tuned for METABRIC (1.9k samples) / SUPPORT (8.8k),
    # while our datasets are 6-30x larger and need more epochs to converge.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    for ds in targets:
        info = DATASETS[ds]
        tname = info["tensor_name"]
        num_risks = info["num_risks"]
        proc_dir = _proc_dir_for(ds, args.seed)

        if not proc_dir.exists():
            print(f"\n>>> Skipping {ds}: {proc_dir} not found")
            continue

        ckpt_dir = ROOT / "checkpoints" / f"{ds}_survtrace_seed{args.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "best_model.pth"
        metrics_file = ckpt_dir / "metrics.json"
        if metrics_file.exists() and ckpt_file.exists() and not args.force:
            print(f"[skip] {ds} seed{args.seed}: metrics + ckpt already present")
            continue

        print(f"\n{'=' * 65}")
        print(f"  SurvTRACE baseline — {ds.upper()} seed={args.seed} "
              f"({num_risks} risk{'s' if num_risks > 1 else ''})")
        print(f"{'=' * 65}")

        x_train, dur_train, evt_train, cuts = load_split(proc_dir, tname, "train", dataset_key=ds)
        x_val,   dur_val,   evt_val,   _    = load_split(proc_dir, tname, "val",   dataset_key=ds)
        x_test,  dur_test,  evt_test,  _    = load_split(proc_dir, tname, "test",  dataset_key=ds)
        num_bins = len(cuts)

        print(f"  Shapes: train={x_train.shape} val={x_val.shape} test={x_test.shape}")
        print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%  "
              f"Bins (K): {num_bins}  Risks: {num_risks}")

        x_train, x_val, x_test, keep_mask = _remove_constant_cols(x_train, x_val, x_test)
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} constant features → {x_train.shape[1]} remaining")

        try:
            model, df_train, df_val, train_info = train_survtrace(
                x_train, dur_train, evt_train,
                x_val,   dur_val,   evt_val,
                cuts=cuts, num_risks=num_risks,
                checkpoint_path=ckpt_file,
                batch_size=args.batch_size, epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay,
                early_stop_patience=args.early_stop_patience,
            )
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {exc}")
            continue

        # Evaluate on val + test using pycox EvalSurv (same as DeepHit).
        results = {"model_info": {"type": "SurvTRACE",
                                   **train_info,
                                   "n_features": int(x_train.shape[1]),
                                   "num_risks": num_risks,
                                   "num_bins": int(num_bins),
                                   "seed": args.seed,
                                   "hyperparameters": {
                                       "batch_size": args.batch_size,
                                       "epochs": args.epochs,
                                       "early_stop_patience": args.early_stop_patience,
                                       "lr": args.lr,
                                       "weight_decay": args.weight_decay,
                                   }}}

        for split_name, x_s, dur_s, evt_s in [
            ("val",  x_val,  dur_val,  evt_val),
            ("test", x_test, dur_test, evt_test),
        ]:
            df_s = _make_feature_df(x_s)
            metrics = evaluate_survtrace(model, df_s, dur_s, evt_s, cuts, num_risks)
            results[split_name] = metrics
            m = metrics.get("mean", metrics.get("risk1", {}))
            print(f"  {split_name}: Ctd={m.get('ctd', float('nan')):.4f}  "
                  f"IBS={m.get('ibs', float('nan')):.4f}  "
                  f"IBLL={m.get('ibll', float('nan')):.4f}")

        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x)
                      if isinstance(x, (np.floating, np.integer)) else str(x))
        print(f"  Wrote {metrics_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
