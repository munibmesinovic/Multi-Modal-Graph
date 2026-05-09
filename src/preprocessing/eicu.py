"""eICU preprocessing — survival adaptation of the DynaGraph pipeline.

Starts from Emma Rocheteau's intermediate CSVs (the same ones DynaGraph uses)
and produces a (N, 6, ~93) tensor with single-risk ICU mortality
labels for MM-GraphSurv.

Input CSVs (from ``cfg.data.raw_dir``):
    preprocessed_labels.csv          per-patient outcomes
    preprocessed_flat.csv            static demographics
    preprocessed_diagnoses.csv       admission diagnoses (one-hot)
    preprocessed_diagnoses_post.csv  post-discharge diagnoses (not used here)
    timeseries_lab.csv               hourly labs
    timeseries_aperiodic.csv         hourly vitals (aperiodic)

Output: dynamic + static fused into (N, n_blocks, F) where time is grouped
into ``hours_per_block``-hour windows by mean aggregation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import (
    stratified_split_indices,
    make_quantile_cuts,
    assign_bin_indices,
    save_split,
    print_audit,
)
from ..utils import resolve_path

warnings.filterwarnings("ignore")


# Lab features dropped per the original GraphSurv eICU_Survival.ipynb (cell 16).
# This is the 18-feature list — DynaGraph drops 32 (wider net), but the
# golden reference for MM-GraphSurv uses the smaller list and keeps blood
# gas / calibration features that DynaGraph removes.
LAB_FEATURES_TO_DROP = [
    "-basos", "-eos", "-lymphs", "-monos", "-polys",
    "ALT (SGPT)", "AST (SGOT)", "PT", "PT - INR", "PTT",
    "albumin", "alkaline phos.", "lactate", "phosphate",
    "total bilirubin", "total protein", "troponin - I",
    "urinary specific gravity",
]


def _resample_hourly(ts: pd.DataFrame) -> pd.DataFrame:
    """Replicate the DynaGraph notebook's hourly resampling pipeline."""
    ts = ts.groupby(level=[0, 1]).mean()

    ts.reset_index(level=1, inplace=True)
    start = pd.to_datetime("2000-01-01 00:00:00")
    ts.time = pd.to_timedelta(ts.time, errors="coerce") + start
    ts.time = ts.time.dt.ceil(freq="H")
    ts.time = ts.time - start
    ts.time = pd.to_timedelta(ts.time, unit="T")
    ts.set_index("time", append=True, inplace=True)
    ts.reset_index(level=0, inplace=True)
    ts = ts.groupby("patient").resample("H", closed="right", label="right").mean().drop(columns="patient")

    # ffill ONLY within each patient. bfill would let earlier timesteps
    # inherit from later (unobserved-at-the-time) values, which is forward
    # leakage for a survival model. Remaining NaN will be handled by the
    # train-set population mean impute after the split.
    ts.update(ts.groupby(level=0).ffill())

    ts.reset_index(level=1, inplace=True)
    ts.time = pd.to_timedelta(ts.time, errors="coerce")
    ts.time = ts.time.astype(int) / (1_000_000_000 * 60)  # → minutes
    ts.reset_index(inplace=True)
    ts.set_index(["patient", "time"], inplace=True)
    return ts


def _train_mean_impute(x_blocks: np.ndarray, n_dyn: int, train_idx: np.ndarray) -> np.ndarray:
    """Population-mean imputation (paper §4.1).

    For each dynamic feature, compute mean from non-NaN values in TRAINING
    patients only, then fill any remaining NaN in train/val/test with that mean.
    Static and pooled features are assumed already non-NaN.
    """
    out = x_blocks.copy()
    for j in range(n_dyn):
        col = out[train_idx, :, j]
        mean = np.nanmean(col) if np.isnan(col).any() else col.mean()
        if np.isnan(mean):
            mean = 0.0
        out[:, :, j] = np.where(np.isnan(out[:, :, j]), mean, out[:, :, j])
    return out


def run_pipeline(cfg: dict) -> dict:
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    seed = cfg["data"]["seed"]
    n_hours = cfg["data"]["n_hours"]
    n_blocks = cfg["data"]["n_blocks"]
    hours_per_block = n_hours // n_blocks
    K = cfg["data"]["num_durations"]
    exclude_age_over = cfg["data"].get("exclude_age_over", 89)

    # ---- Load raw -------------------------------------------------------------
    print("Loading raw eICU CSVs ...")
    labels = pd.read_csv(raw_dir / "preprocessed_labels.csv", index_col="patient")
    labels = labels[["unitdischargeoffset", "unitdischargestatus_Expired"]]
    # Cohort: LOS ≥ 24h (no upper bound on the FILTER). Patients with
    # LOS > 240h are KEPT but their duration is clipped to 240h and the
    # event is right-censored to 0 (paper §4.1: 10-day horizon).
    los_h = labels["unitdischargeoffset"] / 60.0
    labels = labels[los_h >= 24.0]
    # Clip durations and right-censor events for over-the-horizon stays
    over_horizon = (labels["unitdischargeoffset"] / 60.0) > 240.0
    labels.loc[over_horizon, "unitdischargeoffset"] = 240.0 * 60.0
    labels.loc[over_horizon, "unitdischargestatus_Expired"] = 0

    flats = pd.read_csv(raw_dir / "preprocessed_flat.csv", index_col="patient")
    # Drop patients with the `> 89` (age > 89) indicator, then drop the
    # column itself plus the null-weight/null-height indicators. This
    # combination matches the golden reference cohort exactly.
    if "> 89" in flats.columns:
        flats = flats[flats["> 89"] != 1]
    flats = flats.drop(columns=[c for c in ("> 89", "nullweight", "nullheight") if c in flats.columns])

    ts_lab_raw = pd.read_csv(raw_dir / "timeseries_lab.csv", index_col=["patient", "time"])
    ts_aper_raw = pd.read_csv(raw_dir / "timeseries_aperiodic.csv", index_col=["patient", "time"])

    for name, df in [("labels", labels), ("flats", flats),
                     ("ts_lab", ts_lab_raw), ("ts_aper", ts_aper_raw)]:
        print(f"  {name:10s} {df.shape}")

    # ---- Cohort: keep patients present in labels + flats ---------------------
    print("\nBuilding cohort ...")
    final = labels.merge(flats, left_index=True, right_index=True)
    print(f"  After labels ⋂ flats : {len(final)}")

    # ---- Lab time-series ------------------------------------------------------
    print("\nResampling labs ...")
    ts_lab = _resample_hourly(ts_lab_raw)
    cols_to_drop = [c for c in LAB_FEATURES_TO_DROP if c in ts_lab.columns]
    ts_lab = ts_lab.drop(columns=cols_to_drop)
    print(f"  Lab features kept    : {ts_lab.shape[1]} (dropped {len(cols_to_drop)})")

    # Permissive merge: keep patients even if some labs are entirely missing.
    # Population-mean imputation happens after the train/val/test split below.
    final = final.merge(ts_lab, left_index=True, right_index=True, how="left")
    n_after_lab = final.index.get_level_values(0).nunique() if isinstance(final.index, pd.MultiIndex) else final.index.nunique()
    print(f"  After lab left merge : {n_after_lab}")

    # ---- Aperiodic vitals -----------------------------------------------------
    print("\nResampling aperiodic vitals ...")
    ts_aper = _resample_hourly(ts_aper_raw)
    final = final.merge(ts_aper, left_index=True, right_index=True, how="left")
    print(f"  After vitals merge   : {final.index.get_level_values(0).nunique() if isinstance(final.index, pd.MultiIndex) else final.index.nunique()}")

    # Within-patient ffill across the merged frame so static cols and
    # observed dynamic measurements get propagated forward. We do NOT bfill
    # (leakage — see PIPELINE_LOG.md "Option A"). Remaining NaN will be
    # filled by train-set population mean after the split.
    final.update(final.groupby(level=0).ffill())

    # No additional ≥24h-entries filter needed: the labels-table cohort
    # already enforced LOS ≥ 24 hours, which is what the golden reference uses.

    # ---- Survival labels (D25: ICU mortality) ---------------------------------
    label_df = final[["unitdischargeoffset", "unitdischargestatus_Expired"]].droplevel(1)
    label_df = label_df[~label_df.index.duplicated(keep="first")]
    # unitdischargestatus_Expired is a one-hot: 1 = died in ICU, 0 = survived ICU
    label_df["icu_death"] = pd.to_numeric(label_df["unitdischargestatus_Expired"], errors="coerce").fillna(0).astype(int)

    durations_all = (label_df["unitdischargeoffset"].values.astype(np.float32) / 60.0)  # min → hours
    events_all = label_df["icu_death"].values.astype(np.int64)

    # ---- Build (N, T, F) tensor ----------------------------------------------
    print("\nAssembling 3-D tensor ...")
    feat_cols = [c for c in final.columns if c not in ("unitdischargeoffset", "unitdischargestatus_Expired")]
    X_df = final[feat_cols]

    # Persist as numpy array so we can index into it later.
    pids = np.asarray(label_df.index.values)
    n_patients = len(pids)
    n_features = len(feat_cols)
    x_all = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)
    for i, pid in enumerate(pids):
        seq = X_df.loc[pid]
        seq = seq.values if isinstance(seq, pd.DataFrame) else seq.values.reshape(1, -1)
        t = min(seq.shape[0], n_hours)
        x_all[i, n_hours - t:] = seq[-t:]
        if t < n_hours:
            x_all[i, :n_hours - t] = seq[0]

    # Compute per-patient modality presence BEFORE imputation.
    # has_dynamic[i] = True if patient i had at least one non-NaN dynamic cell
    # across the 24-hour window. Static is always present (cohort filter).
    has_dynamic = ~np.isnan(x_all).all(axis=(1, 2))
    has_static = np.ones(n_patients, dtype=bool)
    modality_mask_all = np.stack([has_dynamic, has_static], axis=1)  # (N, 2)
    print(f"  modality mask: dynamic={has_dynamic.sum()}/{n_patients}  static={has_static.sum()}/{n_patients}")

    print(f"  x_all={x_all.shape}, NaN before impute={np.isnan(x_all).sum()}")

    # ---- Aggregate to n_blocks via mean per block ----------------------------
    # nanmean so block means don't get poisoned by NaN cells before imputation.
    x_blocks = np.nanmean(
        x_all.reshape(n_patients, n_blocks, hours_per_block, n_features), axis=2,
    )
    print(f"  After block agg     : {x_blocks.shape} (mean over {hours_per_block}h)")

    # ---- Stratified split FIRST so imputation/scaling fit on train only ------
    train_idx, val_idx, test_idx = stratified_split_indices(
        n_patients, events_all,
        cfg["data"]["splits"]["train"],
        cfg["data"]["splits"]["val"],
        cfg["data"]["splits"]["test"],
        seed=seed,
    )

    # Population-mean imputation on dynamic features (paper §4.1)
    n_dyn_block = sum(1 for c in feat_cols
                       if not c.startswith(("gender", "age", "admissionheight",
                                            "admissionweight", "hour", "teaching",
                                            "ethnicity", "unittype", "unitadmit",
                                            "unitvisit", "unitstay", "physician",
                                            "numbeds", "region")))
    x_blocks = _train_mean_impute(x_blocks, n_dyn=n_features, train_idx=train_idx)
    print(f"  NaN after impute   : {np.isnan(x_blocks).sum()}")

    # Standardise on training only
    scaler = StandardScaler()
    flat_train = x_blocks[train_idx].reshape(-1, n_features)
    scaler.fit(flat_train)
    x_blocks = scaler.transform(x_blocks.reshape(-1, n_features)).reshape(n_patients, n_blocks, n_features).astype(np.float32)

    cuts = make_quantile_cuts(
        durations_all[train_idx], K,
        events_train=events_all[train_idx],
    )
    bins = assign_bin_indices(durations_all, cuts)

    splits = {}
    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[split] = {
            "x": x_blocks[idx],
            "durations_idx": bins[idx],
            "durations_raw": durations_all[idx],
            "events": events_all[idx],
            "modality_mask": modality_mask_all[idx],
            "pids": pids[idx],
        }

    save_split(out_dir, "eICU", splits, cuts, feature_names=feat_cols,
               modality_keys=["dynamic", "static"])
    print_audit("eICU", splits, cuts, n_features=n_features, n_blocks=n_blocks)

    return {
        "name": "eicu",
        "n_total": n_patients,
        "shape_train": splits["train"]["x"].shape,
        "out_dir": str(out_dir),
    }
