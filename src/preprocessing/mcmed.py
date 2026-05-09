"""MC-MED preprocessing — emergency department cohort with 4 modalities.

Reads the MC-MED v1.0 CSVs (visits, labs, numerics, pmh, rads, waveform_summary)
from ``configs/mcmed.yaml:data.raw_dir`` and writes (N, 6, 1323) tensors plus
discrete-time competing-risk labels to ``data.processed_dir``.

  • Discrete-time labels via pycox LabTransDiscreteTime with the cuts from
    ``cfg.data.cuts_hours``
  • Stratified split by visit, deterministic per seed
  • Clinical-Longformer embeddings cached via longformer_cache.py
  • Aggregates the (N, 24, F) hourly tensor to (N, 6, F) 4-hour blocks
    via mean pooling (matching the paper's s=6 specification)
  • Outputs all 4 events: 0=censor (Discharge), 1=ICU, 2=Inpatient, 3=Observation

Inputs (read from cfg.data.raw_dir):
    visits.csv          demographics + triage vitals + ED outcome
    labs.csv            laboratory time-series
    numerics.csv        vital sign time-series
    pmh.csv             past medical history (ICD-like codes)
    rads.csv            radiology report text
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .base import (
    stratified_split_indices,
    save_split,
    print_audit,
)
from .longformer_cache import get_or_compute_embeddings
from ..utils import resolve_path

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 1. Visits — demographics, triage, ED outcome
# ----------------------------------------------------------------------------

ED_DISPO_MAP = {
    "Discharge": 0,
    "ICU": 1,
    "Inpatient": 2,
    "Observation": 3,
}
TRIAGE_ACUITY_MAP = {
    "1-Resuscitation": 5, "2-Emergent": 4, "3-Urgent": 3,
    "4-Semi-Urgent": 2, "5-Non-Urgent": 1,
}


def _load_visits(raw_dir: Path, required_cols: list[str], min_los: float) -> pd.DataFrame:
    visits = pd.read_csv(raw_dir / "visits.csv")
    visits = visits.dropna(subset=required_cols)
    visits = visits[[
        "MRN", "CSN", "Age", "Gender", "Race", "Ethnicity",
        "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2",
        "Triage_SBP", "Triage_DBP", "Triage_acuity", "ED_LOS", "ED_dispo",
    ]]
    visits = visits[(visits["ED_LOS"] >= min_los) & (visits["ED_LOS"] <= 24)]

    # One-hot encode demographics
    for col in ["Race", "Gender", "Ethnicity"]:
        oh = pd.get_dummies(visits[col], prefix=col, dtype=int)
        visits = visits.drop(col, axis=1).join(oh, rsuffix=f"_{col}")

    visits["Outcome"] = visits["ED_dispo"].map(ED_DISPO_MAP)
    visits = visits.dropna(subset=["Outcome"])
    visits["Triage_acuity_ordinal"] = visits["Triage_acuity"].map(TRIAGE_ACUITY_MAP).fillna(-1)
    visits = visits.drop(columns=["ED_dispo", "Triage_acuity"])
    visits["Outcome"] = visits["Outcome"].astype(int)
    return visits


# ----------------------------------------------------------------------------
# 2. Radiology — Clinical-Longformer embeddings (cached)
# ----------------------------------------------------------------------------

def _load_rad_embeddings(raw_dir: Path, cache_path: Path, model_name: str, max_length: int) -> pd.DataFrame:
    rads = pd.read_csv(raw_dir / "rads.csv")
    rads = rads.dropna(subset=["Study", "Impression"])
    rads["Text"] = rads["Study"].astype(str) + " " + rads["Impression"].astype(str)
    rads = rads[["CSN", "Text"]].drop_duplicates(subset=["CSN"], keep="first")
    rads = rads.set_index("CSN")
    emb = get_or_compute_embeddings(
        texts=rads["Text"].tolist(),
        cache_path=cache_path,
        model_name=model_name,
        max_length=max_length,
    )
    cols = [f"rad_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=rads.index, columns=cols)


# ----------------------------------------------------------------------------
# 3. PMH — past medical history → top-k ICD-like one-hot
# ----------------------------------------------------------------------------

def _load_pmh(raw_dir: Path, mrns_in_visits, top_k: int) -> pd.DataFrame:
    pmh = pd.read_csv(raw_dir / "pmh.csv")[["MRN", "Code"]]
    pmh = pmh[pmh["MRN"].isin(mrns_in_visits)]
    top_codes = pmh["Code"].value_counts().nlargest(top_k).index.tolist()
    pmh = pmh[pmh["Code"].isin(top_codes)]
    pmh = pmh.set_index("MRN")
    icd_oh = pd.get_dummies(pmh["Code"], dtype=int, prefix="icd")
    return icd_oh.groupby(level=0).sum().clip(upper=1)


# ----------------------------------------------------------------------------
# 4. Time series — labs and vitals → hourly per CSN
# ----------------------------------------------------------------------------

def _resample_hourly(df, csn_col, time_col, missing_thresh=0.85) -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()]
    df[time_col] = df[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _load_labs(raw_dir: Path) -> pd.DataFrame:
    labs = pd.read_csv(raw_dir / "labs.csv")
    labs["Result_time"] = pd.to_datetime(labs["Result_time"], errors="coerce")
    labs = labs[labs["Result_time"].notna()]

    labs = labs.pivot_table(
        index=["CSN", "Result_time"], columns="Component_name",
        values="Component_value", aggfunc="first",
    )
    # Drop columns with >85% missing
    labs = labs.loc[:, labs.isnull().mean() < 0.85]
    for c in labs.columns:
        labs[c] = pd.to_numeric(labs[c], errors="coerce")

    labs = _shift_resample(labs, time_index="Result_time")
    labs.index = labs.index.rename("Time", level="Result_time")
    return labs


def _load_vitals(raw_dir: Path) -> pd.DataFrame:
    vitals = pd.read_csv(raw_dir / "numerics.csv")
    vitals["Time"] = pd.to_datetime(vitals["Time"], errors="coerce")
    vitals = vitals[vitals["Time"].notna()]

    vitals = vitals.pivot_table(
        index=["CSN", "Time"], columns="Measure", values="Value", aggfunc="first",
    )
    vitals = vitals.loc[:, vitals.isnull().mean() < 0.85]
    for c in vitals.columns:
        vitals[c] = pd.to_numeric(vitals[c], errors="coerce")

    vitals = _shift_resample(vitals, time_index="Time")
    return vitals


def _shift_resample(df: pd.DataFrame, time_index: str) -> pd.DataFrame:
    """Time-shift each CSN to start at 0, then resample hourly with ffill only."""
    df = df.reset_index()
    df[time_index] = pd.to_datetime(df[time_index], errors="coerce")
    df = df.set_index(["CSN", time_index])
    df = df.reset_index(level=1)

    minimum_shifts = df.groupby("CSN")[time_index].min()
    df = df.merge(minimum_shifts, left_index=True, right_index=True, suffixes=("_x", "_y"))
    df[time_index] = df[f"{time_index}_x"] - df[f"{time_index}_y"]
    df = df.drop(columns=[f"{time_index}_x", f"{time_index}_y"])
    df = df.set_index(time_index, append=True)

    df = df.groupby(level=[0, 1]).mean()
    df = df.reset_index(level=1)
    df[time_index] = df[time_index].dt.ceil(freq="h")
    df[time_index] = pd.to_timedelta(df[time_index], unit="T")
    df = df.set_index(time_index, append=True).reset_index(level=0)
    df = df.groupby("CSN").resample("h", closed="right", label="right").mean().drop(columns="CSN")
    # ffill only — no bfill (leakage). Direct groupby.ffill() assignment is
    # more reliable than `df.update(...)` with a multiindex. Remaining NaN
    # handled by train-set population mean imputation after the split.
    df = df.groupby(level=0).ffill()

    # IMPORTANT: do NOT drop patients with any-NaN here. With ffill-only,
    # many patients will legitimately have NaN in never-measured features.
    # Those cells are filled by train-set population mean after the split.
    return df


# ----------------------------------------------------------------------------
# 5. Tensor builder
# ----------------------------------------------------------------------------

def _split_sequence(df: pd.DataFrame, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    pids = df.index.get_level_values(0).unique()
    out = np.full((len(pids), n_steps, df.shape[1]), np.nan, dtype=np.float32)
    for i, pid in enumerate(pids):
        seq = df.loc[pid].values
        t = min(seq.shape[0], n_steps)
        out[i, -t:, :] = seq[-t:]
    # ffill within each sequence
    for i in range(out.shape[0]):
        for j in range(out.shape[2]):
            valid = np.where(~np.isnan(out[i, :, j]))[0]
            if len(valid):
                first = valid[0]
                if first > 0:
                    out[i, :first, j] = out[i, first, j]
                # forward fill any later NaNs
                for k in range(1, n_steps):
                    if np.isnan(out[i, k, j]):
                        out[i, k, j] = out[i, k - 1, j]
    out = np.nan_to_num(out, nan=0.0)
    return out, np.array(pids)


# ----------------------------------------------------------------------------
# 6. Main pipeline
# ----------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> dict:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    cache_dir = resolve_path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["data"]["seed"]
    n_hours = cfg["data"]["n_hours"]
    n_blocks = cfg["data"]["n_blocks"]
    hours_per_block = n_hours // n_blocks
    K = cfg["data"]["num_durations"]

    np.random.seed(seed)

    print("=" * 65)
    print("  MC-MED preprocessing pipeline")
    print("=" * 65)

    # ---- 1. Visits ------------------------------------------------------------
    print("\n[1/6] Loading visits.csv ...")
    visits = _load_visits(
        raw_dir,
        required_cols=cfg["data"]["required_columns"],
        min_los=cfg["data"]["min_los_hours"],
    )
    print(f"  visits: {visits.shape} unique CSN={visits['CSN'].nunique()}")

    # ---- 2. Radiology embeddings ---------------------------------------------
    print("\n[2/6] Loading radiology + Clinical-Longformer ...")
    rad_df = _load_rad_embeddings(
        raw_dir,
        cache_path=cache_dir / "rad_embeddings.npy",
        model_name=cfg["model"]["longformer_model"],
        max_length=cfg["model"]["longformer_max_length"],
    )
    # Per-visit radiology presence indicator, captured BEFORE the fillna(0)
    # merge so we can tell "no report" from "Longformer happened to output ~0".
    csns_with_rad = set(rad_df.index.astype(int)) if rad_df.index.dtype != object else set(rad_df.index)
    visits = visits.set_index("CSN").merge(rad_df, how="left", left_index=True, right_index=True).fillna(0).reset_index()

    # ---- 3. PMH (ICD-like codes) ---------------------------------------------
    print("\n[3/6] Loading PMH (ICD codes) ...")
    icd_df = _load_pmh(raw_dir, visits["MRN"].unique(), top_k=cfg["model"]["top_icd_codes"])
    mrns_with_icd = set(icd_df.index)
    visits = visits.set_index("MRN").merge(icd_df, how="left", left_index=True, right_index=True).fillna(0).reset_index()
    visits = visits.set_index("CSN")
    visits = visits[~visits.index.duplicated(keep="first")]
    # Capture CSN → MRN mapping now (before we drop MRN later for the merge).
    csn_to_mrn = visits["MRN"].to_dict()

    # ---- 4. Labs + vitals time series ----------------------------------------
    print("\n[4/6] Loading labs.csv and numerics.csv ...")
    labs = _load_labs(raw_dir)
    vitals = _load_vitals(raw_dir)
    csns_with_labs = set(labs.index.get_level_values(0))
    csns_with_vitals = set(vitals.index.get_level_values(0))

    # Outer join labs and vitals on (CSN, time) so neither modality gets
    # truncated by the other. An inner join would drop every timestamp where
    # only one of the two modalities was measured — that's most timestamps.
    time_series = vitals.merge(labs, left_index=True, right_index=True, how="outer")
    n_dyn_cols = time_series.shape[1]
    print(f"  vitals cols: {vitals.shape[1]}  labs cols: {labs.shape[1]}  joined: {n_dyn_cols}")
    print(f"  Time-series rows (outer): {len(time_series)}  unique CSNs: {time_series.index.get_level_values(0).nunique()}")

    # Left-merge with visits (visits has no time dim; pandas broadcasts).
    # Keep all CSNs from visits even if they have no time-series observation.
    final = time_series.merge(
        visits.drop(columns=["MRN"]), left_index=True, right_index=True, how="right",
    )
    n_after_ts_merge = final.index.get_level_values(0).nunique()
    print(f"  After ts ⋉ visits (right join): {n_after_ts_merge} CSNs")

    # Pop labels before imputation (they come from visits, always present)
    label_df = final[["ED_LOS", "Outcome"]].groupby(level=0).first()
    durations_all = label_df["ED_LOS"].values.astype(np.float32)
    events_all = label_df["Outcome"].values.astype(np.int64)
    final = final.drop(columns=["ED_LOS", "Outcome"])

    # ---- 5. Build (N, n_hours, F) ------------------------------------------
    print(f"\n[5/6] Building tensor with n_steps={n_hours}, F={final.shape[1]} ...")
    x_24, pids = _split_sequence(final, n_steps=n_hours)
    n_features = x_24.shape[2]
    print(f"  x_24={x_24.shape}, NaN pre-impute={np.isnan(x_24).sum()}")
    # Align durations/events with pids order (ledger may have shuffled)
    label_df = label_df.reindex(pids)
    durations_all = label_df["ED_LOS"].values.astype(np.float32)
    events_all = label_df["Outcome"].values.astype(np.int64)

    n = x_24.shape[0]

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split_indices(
        n, events_all,
        cfg["data"]["splits"]["train"],
        cfg["data"]["splits"]["val"],
        cfg["data"]["splits"]["test"],
        seed=seed,
    )

    # Train-set population mean impute for any remaining NaN in the
    # dynamic-only block (labs + vitals, first n_dyn_cols columns of the
    # merged frame). Static/ICD/rad modalities live further right and were
    # already filled with 0 during the left merge so they don't need it.
    # Fit on train only — no leakage.
    n_dyn = n_dyn_cols
    train_dyn = x_24[train_idx, :, :n_dyn]
    pop_mean = np.nanmean(train_dyn.reshape(-1, n_dyn), axis=0)
    pop_mean = np.where(np.isnan(pop_mean), 0.0, pop_mean)
    nan_mask = np.isnan(x_24[:, :, :n_dyn])
    if nan_mask.any():
        idx = np.where(nan_mask)
        x_24[:, :, :n_dyn][idx] = pop_mean[idx[2]]
    # Any stray NaN in static/ICD/rad (shouldn't happen) → 0
    x_24 = np.nan_to_num(x_24, nan=0.0)
    print(f"  NaN after impute: {np.isnan(x_24).sum()}  (dynamic cols 0:{n_dyn})")

    # Quantile-transform first n_dyn features per timestep, fit on train only
    scalers = {}
    for t in range(n_hours):
        s = QuantileTransformer(random_state=seed, n_quantiles=min(1000, len(train_idx)))
        s.fit(x_24[train_idx, t, :n_dyn])
        scalers[t] = s
    for split_idx in (train_idx, val_idx, test_idx):
        for t in range(n_hours):
            x_24[split_idx, t, :n_dyn] = scalers[t].transform(x_24[split_idx, t, :n_dyn])

    # D21: Standardise static features (between dynamic and rad blocks).
    # MC-MED layout: [0:n_dyn] dynamic, [n_dyn:rad_start] static, [rad_start:icd_start] rad, [icd_start:] icd.
    # Identify rad_start from feature names (first column starting with "rad_").
    all_cols = list(final.columns)
    rad_start = next(i for i, c in enumerate(all_cols) if c.startswith("rad_"))
    stat_lo = n_dyn
    stat_hi = rad_start
    if stat_hi > stat_lo:
        stat_scaler = StandardScaler()
        flat_train_stat = x_24[train_idx, :, stat_lo:stat_hi].reshape(-1, stat_hi - stat_lo)
        stat_scaler.fit(flat_train_stat)
        x_24[:, :, stat_lo:stat_hi] = stat_scaler.transform(
            x_24[:, :, stat_lo:stat_hi].reshape(-1, stat_hi - stat_lo)
        ).reshape(n, n_hours, stat_hi - stat_lo)
        print(f"  Static cols [{stat_lo}:{stat_hi}] scaled: "
              f"train mean={x_24[train_idx, :, stat_lo:stat_hi].mean():.3f} "
              f"std={x_24[train_idx, :, stat_lo:stat_hi].std():.3f}")

    # Aggregate to (N, n_blocks, F) by mean over each block
    x_blocks = x_24.reshape(n, n_blocks, hours_per_block, n_features).mean(axis=2).astype(np.float32)
    print(f"  x_blocks={x_blocks.shape} after {hours_per_block}h block agg")

    # ---- 6. Discrete-time bins (custom MC-MED cuts) --------------------------
    cuts = np.array(cfg["data"]["cuts_hours"], dtype=np.float32)
    bins = np.searchsorted(cuts, durations_all, side="right") - 1
    bins = np.clip(bins, 0, len(cuts) - 2).astype(np.int64)

    # Build per-visit modality mask:
    #   dynamic = visit had at least one lab OR vital observation
    #   static  = always True (cohort filter enforces Race, Triage_*, etc.)
    #   icd     = patient's MRN had at least one PMH code in the top-500
    #   rad     = visit's CSN had at least one non-empty rad report
    def _lookup(csn, lut):
        # pids may be float32/64 depending on pandas inference; cast.
        try:
            return int(csn)
        except (TypeError, ValueError):
            return csn
    has_dynamic = np.array([
        (_lookup(p, None) in csns_with_labs) or (_lookup(p, None) in csns_with_vitals)
        for p in pids
    ])
    has_static = np.ones(len(pids), dtype=bool)
    has_icd = np.array([
        csn_to_mrn.get(_lookup(p, None)) in mrns_with_icd
        for p in pids
    ])
    has_rad = np.array([_lookup(p, None) in csns_with_rad for p in pids])
    modality_mask_all = np.stack([has_dynamic, has_static, has_icd, has_rad], axis=1)
    print(f"  modality mask: dyn={has_dynamic.sum()} stat={has_static.sum()} "
          f"icd={has_icd.sum()} rad={has_rad.sum()} (of {len(pids)})")

    splits = {}
    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[split] = {
            "x": x_blocks[idx],
            "durations_idx": bins[idx],
            "durations_raw": durations_all[idx],
            "events": events_all[idx],
            "modality_mask": modality_mask_all[idx],
            "pids": np.asarray(pids)[idx],
        }

    save_split(out_dir, "MCMED", splits, cuts, feature_names=list(final.columns),
               modality_keys=["dynamic", "static", "icd", "rad"])
    print_audit("MCMED", splits, cuts, n_features=n_features, n_blocks=n_blocks)

    return {
        "name": "mcmed",
        "n_total": n,
        "shape_train": splits["train"]["x"].shape,
        "out_dir": str(out_dir),
    }
