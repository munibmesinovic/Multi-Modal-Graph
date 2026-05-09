"""HIRID preprocessing — ICU mortality survival adaptation.

Pipeline:
  Stage 1 (ONCE, cached):
      Stream 250 imputed_stage/csv batches (5.5 GB total, ~22 MB each) to:
        * build (N, 24, 18) time-series tensor at hourly resolution,
        * compute per-patient ICU LOS = max(reldatetime) / 3600 in hours,
        * join with general_table.csv for age + sex (static),
        * apply ffill-only within patient (D10), then train-mean impute post-split.
      Cached artifacts in data/hirid/cache/:
        * hirid_hourly_features.npy  (N, 24, 18)
        * hirid_los_hours.npy        (N,)
        * hirid_event.npy            (N,)        discharge_status == 'dead'
        * hirid_static.npy           (N, 2)      age, sex
        * hirid_pids.npy             (N,)        patient IDs

  Stage 2 (per-seed, fast):
      Apply our 5-seed stratified 70/10/20 split, train-mean impute remaining NaN,
      StandardScaler on train only, pycox discretization, save.

Conventions (MM-GraphSurv):
  * 24h observation window (first 24 hours of each stay, matching eICU + MIMIC)
  * Clip durations at 240h horizon; right-censor events beyond (D25)
  * ffill-only imputation (no bfill, D10)
  * Per-modality StandardScaler (dynamic 18 + static 2) fit on train only (D21)
  * K = 10 pycox quantile cuts fit on training events (D9)
  * Tensor layout: [dynamic, static] → (N, 6, 20); first 18 cols dynamic, last 2 static
  * Output s = 6 × 4h by mean-pooling 4 consecutive hourly windows (matches
    eICU/MIMIC s=6 convention)
  * 70/10/20 split matching eICU (single-risk ICU mortality, similar cohort size)

OOM safety: batches streamed one-at-a-time, peak RAM ≈ one batch + tensor ≈ 150 MB.
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

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


# 18 clinical time-series features in the imputed-staging CSVs
FEATURE_COLS = [
    "vm1", "vm3", "vm4", "vm5", "vm13", "vm20", "vm28", "vm62",
    "vm136", "vm146", "vm172", "vm174", "vm176",
    "pm41", "pm42", "pm43", "pm44", "pm87",
]

DYNAMIC_FEATURE_NAMES = [
    "HR", "ABP_sys", "ABP_dia", "MAP", "CO", "SpO2", "RASS", "PeakPressure",
    "lactate_arterial", "lactate_venous", "INR", "glucose", "CRP",
    "dobutamine", "milrinone", "levosimendan", "theophylline", "analgesics",
]
# Static feature names are built dynamically in _build_cache (includes
# age, sex, height, APACHE II/IV one-hots + presence indicators).


# ----------------------------------------------------------------------------
# Stage 1: cache (runs once, shared across seeds)
# ----------------------------------------------------------------------------

def _build_static_rich(labels_df: pd.DataFrame, raw_dir: Path, apache_top_k: int = 5) -> tuple[np.ndarray, list[str]]:
    """Build richer static features beyond age + sex.

    Adds:
      - height (continuous, 7.4% NaN → mean-impute)
      - APACHE II Group top-K one-hot + 'other' bucket + presence indicator (47% NaN)
      - APACHE IV Group top-K one-hot + 'other' bucket + presence indicator (50% NaN)

    Returns:
      static (N, 3 + 2*(K+1) + 2) float32, feature_names list.
    """
    candidates = [
        raw_dir / "benchmark_output" / "general_table_extended.parquet",
        raw_dir.parent / "benchmark_output" / "general_table_extended.parquet",
    ]
    ext_df = None
    for c in candidates:
        if c.exists():
            ext_df = pd.read_parquet(c)
            break
    if ext_df is None:
        print("    [warn] general_table_extended.parquet not found; falling back to age+sex only")
        static = np.stack([
            labels_df["age"].values.astype(np.float32),
            labels_df["sex_is_male"].values.astype(np.float32),
        ], axis=1)
        return static, ["age", "sex_is_male"]

    ext_df = ext_df[["patientid", "height", "APACHE II Group", "APACHE IV Group"]]
    labels_df = labels_df.merge(ext_df, on="patientid", how="left")

    # height: mean-impute 7.4% missing
    h_mean = labels_df["height"].mean()
    labels_df["height"] = labels_df["height"].fillna(h_mean)

    feats = [
        labels_df["age"].values.astype(np.float32),
        labels_df["sex_is_male"].values.astype(np.float32),
        labels_df["height"].values.astype(np.float32),
    ]
    names = ["age", "sex_is_male", "height"]

    # APACHE II one-hot (top-K + 'other') + presence indicator
    for label, col in [("apII", "APACHE II Group"), ("apIV", "APACHE IV Group")]:
        vals = labels_df[col]
        top = vals.value_counts(dropna=True).head(apache_top_k).index.tolist()
        for i, code in enumerate(top):
            feats.append((vals == code).astype(np.float32).values)
            names.append(f"{label}_top{i+1}_code{int(code)}")
        is_other = vals.notna() & ~vals.isin(top)
        feats.append(is_other.astype(np.float32).values)
        names.append(f"{label}_other")
        feats.append(vals.notna().astype(np.float32).values)
        names.append(f"{label}_present")

    static = np.stack(feats, axis=1).astype(np.float32)
    return static, names


def _build_cache(raw_dir: Path, cache_dir: Path, n_hours: int = 24) -> dict:
    """Stream the 5.5 GB raw batches once and cache per-patient arrays."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = cache_dir / "hirid_hourly_features.npy"
    los_path = cache_dir / "hirid_los_hours.npy"
    event_path = cache_dir / "hirid_event.npy"
    static_path = cache_dir / "hirid_static.npy"
    static_names_path = cache_dir / "hirid_static_names.json"
    pids_path = cache_dir / "hirid_pids.npy"
    cohort_path = cache_dir / "hirid_cohort_summary.json"

    # Cache must include static_names.json to be valid (schema change guard)
    if all(p.exists() for p in [feat_path, los_path, event_path, static_path, pids_path, static_names_path]):
        print(f"  Cache hit: loading pre-built tensors from {cache_dir}")
        import json
        with open(static_names_path) as f:
            static_names = json.load(f)
        return {
            "x_hourly": np.load(feat_path),
            "los_hours": np.load(los_path),
            "event": np.load(event_path),
            "static": np.load(static_path),
            "static_names": static_names,
            "pids": np.load(pids_path),
        }

    print(f"  Cache miss: building from raw (first run only, ~30 min)")

    # --- cohort + static: labels ∩ general_table[age, sex] + APACHE/height ---
    labels_df = pd.read_csv(raw_dir / "hirid_labels.csv")
    print(f"    hirid_labels.csv rows      : {len(labels_df)}")
    gen = pd.read_csv(raw_dir / "general_table.csv")
    gen = gen[["patientid", "age", "sex", "discharge_status"]].copy()
    gen["sex_is_male"] = gen["sex"].map({"M": 1.0, "F": 0.0})
    gen["event"] = (gen["discharge_status"] == "dead").astype(np.int64)

    labels_df = labels_df.merge(
        gen[["patientid", "age", "sex_is_male", "event"]], on="patientid", how="left"
    )
    n_before = len(labels_df)
    labels_df = labels_df.dropna(subset=["age", "sex_is_male"]).reset_index(drop=True)
    print(f"    After age/sex non-null    : {len(labels_df)} (dropped {n_before - len(labels_df)})")

    labels_df = labels_df.sort_values("patientid").reset_index(drop=True)
    pids = labels_df["patientid"].values.astype(np.int64)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    n_patients = len(pids)

    # Rich static: age + sex + height + APACHE II/IV one-hots + presence indicators
    static, static_names = _build_static_rich(labels_df, raw_dir)
    event = labels_df["event"].values.astype(np.int64)  # (N,)
    print(f"    Cohort size                : {n_patients}")
    print(f"    Event rate (raw mortality) : {event.mean()*100:.2f}%")
    print(f"    Static features ({len(static_names)}): {static_names}")
    print(f"    Age mean ± std             : {static[:, 0].mean():.1f} ± {static[:, 0].std():.1f}")
    print(f"    Sex (male %)               : {static[:, 1].mean()*100:.1f}%")
    print(f"    Height mean ± std          : {static[:, 2].mean():.1f} ± {static[:, 2].std():.1f}")
    print(f"    APACHE II presence         : {static[:, -3].mean()*100:.1f}%")
    print(f"    APACHE IV presence         : {static[:, -1].mean()*100:.1f}%")

    # --- stream CSV batches, build hourly tensor + per-patient LOS ---
    imputed_dir = raw_dir / "imputed_stage" / "csv"
    csv_files = sorted(glob.glob(str(imputed_dir / "*.csv")))
    print(f"    Scanning {len(csv_files)} batches from {imputed_dir}")

    n_feat = len(FEATURE_COLS)
    x_hourly = np.full((n_patients, n_hours, n_feat), np.nan, dtype=np.float32)
    max_reldatetime = np.full(n_patients, np.nan, dtype=np.float64)  # seconds

    cohort_set = set(pids.tolist())

    for file_i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path, usecols=["patientid", "reldatetime"] + FEATURE_COLS)
        df = df[df["patientid"].isin(cohort_set)]
        if len(df) == 0:
            continue

        # Update per-patient max reldatetime (→ LOS) using ALL observations
        max_per_pid = df.groupby("patientid")["reldatetime"].max()
        for pid, rmax in max_per_pid.items():
            idx = pid_to_idx[pid]
            if np.isnan(max_reldatetime[idx]) or rmax > max_reldatetime[idx]:
                max_reldatetime[idx] = rmax

        # Aggregate first 24h features → hourly means
        df24 = df[df["reldatetime"] <= 86100].copy()  # 23h 55m cutoff
        if len(df24):
            df24["hour_bin"] = (df24["reldatetime"] // 3600).astype(np.int64)
            df24 = df24[df24["hour_bin"] < n_hours]
            grouped = df24.groupby(["patientid", "hour_bin"])[FEATURE_COLS].mean()
            for (pid, h), row in grouped.iterrows():
                x_hourly[pid_to_idx[pid], h] = row.values.astype(np.float32)

        if (file_i + 1) % 50 == 0:
            print(f"    Processed {file_i + 1}/{len(csv_files)} batches ...")

    # Any patient with no raw observations at all is dropped (shouldn't happen
    # in the cohort but guard against it).
    has_data = ~np.isnan(max_reldatetime)
    if not has_data.all():
        drop_n = (~has_data).sum()
        print(f"    Dropping {drop_n} patients with no raw observations")
        pids = pids[has_data]
        x_hourly = x_hourly[has_data]
        max_reldatetime = max_reldatetime[has_data]
        static = static[has_data]
        event = event[has_data]
        n_patients = len(pids)

    # ffill-only within each patient for hourly features (D10: no bfill)
    print(f"    Forward-fill within patient (no bfill)")
    for i in range(n_patients):
        for f in range(n_feat):
            col = x_hourly[i, :, f]
            nan_mask = np.isnan(col)
            if nan_mask.all() or not nan_mask.any():
                continue
            idx_arr = np.where(~nan_mask, np.arange(n_hours), 0)
            np.maximum.accumulate(idx_arr, out=idx_arr)
            x_hourly[i, :, f] = col[idx_arr]
            # Residual leading-NaN (pre-first-observation) will get filled by
            # train-set population mean after the seed split.

    # LOS in hours
    los_hours = (max_reldatetime / 3600.0).astype(np.float32)
    print(f"    LOS stats (hours): "
          f"min={np.nanmin(los_hours):.1f}  median={np.nanmedian(los_hours):.1f}  "
          f"max={np.nanmax(los_hours):.1f}  mean={np.nanmean(los_hours):.1f}")

    # Save cache
    np.save(feat_path, x_hourly)
    np.save(los_path, los_hours)
    np.save(event_path, event)
    np.save(static_path, static)
    np.save(pids_path, pids)

    import json
    with open(static_names_path, "w") as f:
        json.dump(static_names, f, indent=2)
    with open(cohort_path, "w") as f:
        json.dump({
            "n_patients": int(n_patients),
            "event_rate_raw_mortality": float(event.mean()),
            "los_min": float(np.nanmin(los_hours)),
            "los_max": float(np.nanmax(los_hours)),
            "los_median": float(np.nanmedian(los_hours)),
            "hourly_nan_fraction_post_ffill": float(np.isnan(x_hourly).mean()),
            "dynamic_feature_names": DYNAMIC_FEATURE_NAMES,
            "static_feature_names": static_names,
            "n_dynamic": len(DYNAMIC_FEATURE_NAMES),
            "n_static": len(static_names),
        }, f, indent=2)

    return {
        "x_hourly": x_hourly, "los_hours": los_hours, "event": event,
        "static": static, "static_names": static_names, "pids": pids,
    }


# ----------------------------------------------------------------------------
# Stage 2: per-seed split, standardize, discretize
# ----------------------------------------------------------------------------

def _train_mean_impute(x_blocks: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Population-mean impute remaining NaN using training fold only (D10)."""
    out = x_blocks.copy()
    n_feat = out.shape[-1]
    for j in range(n_feat):
        col_train = out[train_idx, :, j]
        if np.isnan(col_train).any():
            mean = np.nanmean(col_train)
        else:
            mean = col_train.mean()
        if np.isnan(mean):
            mean = 0.0
        out[:, :, j] = np.where(np.isnan(out[:, :, j]), mean, out[:, :, j])
    return out


def run_pipeline(cfg: dict) -> dict:
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    cache_dir = resolve_path(cfg["data"]["cache_dir"])
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    seed = cfg["data"]["seed"]
    n_hours = cfg["data"]["n_hours"]           # 24
    n_blocks = cfg["data"]["n_blocks"]         # 6
    hours_per_block = n_hours // n_blocks      # 4
    K = cfg["data"]["num_durations"]           # 10
    horizon_hours = cfg["data"].get("horizon_hours", 240.0)

    print(f"Loading cached HIRID artifacts ...")
    cache = _build_cache(raw_dir, cache_dir, n_hours=n_hours)
    x_hourly = cache["x_hourly"]          # (N, 24, 18)
    los_hours = cache["los_hours"]        # (N,)
    event_raw = cache["event"]            # (N,)
    static = cache["static"]              # (N, S)  S ≥ 2
    static_names = cache["static_names"]  # list of length S
    pids = cache["pids"]                  # (N,)
    n_patients = len(pids)

    # --- cohort filter: LOS ≥ 24h (drop short stays) ---
    cohort = los_hours >= 24.0
    n_before = n_patients
    x_hourly = x_hourly[cohort]
    los_hours = los_hours[cohort]
    event_raw = event_raw[cohort]
    static = static[cohort]
    pids = pids[cohort]
    n_patients = len(pids)
    print(f"  LOS ≥ 24h filter           : {n_before} → {n_patients}")

    # --- clip durations at horizon; right-censor events beyond (D25) ---
    over_horizon = los_hours > horizon_hours
    durations_all = np.minimum(los_hours, horizon_hours).astype(np.float32)
    events_all = event_raw.copy()
    events_all[over_horizon] = 0
    print(f"  Over-horizon stays (>{horizon_hours:.0f}h) right-censored: {over_horizon.sum()}")
    print(f"  Final event rate           : {events_all.mean()*100:.2f}%")

    # --- aggregate 24 hourly → 6 x 4h windows (mean per block, nanmean-safe) ---
    x_dyn_blocks = np.nanmean(
        x_hourly.reshape(n_patients, n_blocks, hours_per_block, -1), axis=2,
    ).astype(np.float32)
    n_dyn = x_dyn_blocks.shape[-1]  # 18
    print(f"  Dynamic block tensor       : {x_dyn_blocks.shape}")

    # --- broadcast static (age, sex) across windows and concat ---
    # Order to match eicu convention: dynamic first, static last (paper tensor layout).
    static_broadcast = np.broadcast_to(
        static[:, None, :], (n_patients, n_blocks, static.shape[1])
    ).astype(np.float32).copy()
    x_blocks = np.concatenate([x_dyn_blocks, static_broadcast], axis=-1)
    n_features = x_blocks.shape[-1]
    print(f"  Full block tensor          : {x_blocks.shape} (18 dynamic + 2 static)")

    # --- stratified split (D11: 70/10/20 on ICU datasets) ---
    train_idx, val_idx, test_idx = stratified_split_indices(
        n_patients, events_all,
        cfg["data"]["splits"]["train"],
        cfg["data"]["splits"]["val"],
        cfg["data"]["splits"]["test"],
        seed=seed,
    )
    print(f"  Split sizes (seed {seed})     : "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # --- train-mean impute remaining NaN in dynamic block features ---
    nan_before = np.isnan(x_blocks).sum()
    x_blocks = _train_mean_impute(x_blocks, train_idx=train_idx)
    nan_after = np.isnan(x_blocks).sum()
    print(f"  NaN fill: {nan_before} → {nan_after}")
    assert nan_after == 0, f"{nan_after} NaN remaining after train-mean impute"

    # --- per-modality StandardScaler on TRAIN only (D21) ---
    scaler = StandardScaler()
    flat_train = x_blocks[train_idx].reshape(-1, n_features)
    scaler.fit(flat_train)
    x_blocks = scaler.transform(
        x_blocks.reshape(-1, n_features)
    ).reshape(n_patients, n_blocks, n_features).astype(np.float32)
    # z-clip at ±10 (D34 — outlier safety)
    x_blocks = np.clip(x_blocks, -10.0, 10.0)

    # --- pycox K=10 quantile cuts on training durations + events ---
    cuts = make_quantile_cuts(
        durations_all[train_idx], K,
        events_train=events_all[train_idx],
    )
    bins = assign_bin_indices(durations_all, cuts)

    # --- modality mask (both always present for HIRID) ---
    modality_mask_all = np.ones((n_patients, 2), dtype=np.uint8)

    splits = {}
    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[split] = {
            "x": x_blocks[idx],
            "durations_idx": bins[idx],
            "durations_raw": durations_all[idx].astype(np.float64),
            "events": events_all[idx],
            "modality_mask": modality_mask_all[idx],
            "pids": pids[idx],
        }

    feature_names = DYNAMIC_FEATURE_NAMES + static_names
    save_split(
        out_dir, "HIRID", splits, cuts, feature_names=feature_names,
        modality_keys=["dynamic", "static"],
    )
    print_audit("HIRID", splits, cuts, n_features=n_features, n_blocks=n_blocks)

    return {
        "name": "hirid",
        "n_total": n_patients,
        "shape_train": splits["train"]["x"].shape,
        "out_dir": str(out_dir),
    }
