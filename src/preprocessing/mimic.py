"""MIMIC-IV preprocessing — ICU mortality with 4 modalities.

Reads the MIMIC-IV v3 module CSVs (admissions, chartevents, icustays,
labevents, patients, diagnoses_icd, d_items, d_labitems, radiology) from
``configs/mimic.yaml:data.raw_dir`` and writes (N, 6, 1332) tensors plus
discrete-time survival labels to ``data.processed_dir``.

Cohort:
    • First ICU stay per hospital admission
    • Age >= 18 at ICU intime
    • ICU LOS >= 24 hours
    • ICU mortality (died during ICU stay) as the single risk

Output layout (1332 features):
    [0:33)     dynamic vitals + labs from chartevents/labevents
    [33:56)    static demographics one-hot
    [56:824)   Clinical-Longformer rad embeddings (768 dims)
    [824:1323) ICD code one-hots (top 500 → 499 columns after dedup)
    Note: actual final dim depends on cohort; 1332 is the historical target.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import (
    stratified_split_indices,
    save_split,
    print_audit,
)
from .longformer_cache import get_or_compute_embeddings
from ..utils import resolve_path

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Itemid catalogues (chartevents vitals + labevents labs)
# ----------------------------------------------------------------------------

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "nibp_systolic",
    220180: "nibp_diastolic",
    220181: "nibp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
    223761: "temperature_f",
    223762: "temperature_c",
    220074: "cvp",
    220235: "cuff_pressure",
}

LAB_ITEMIDS = {
    51221: "hematocrit", 51265: "platelet_count", 50912: "creatinine",
    50971: "potassium", 51222: "hemoglobin", 51301: "wbc",
    51279: "rbc", 51250: "mcv", 51248: "mch",
    51277: "rdw", 51006: "urea_nitrogen", 50983: "sodium",
    50902: "chloride", 50882: "bicarbonate", 50868: "anion_gap",
    50931: "glucose", 50960: "magnesium", 50893: "calcium",
    50970: "phosphate", 51237: "inr", 51274: "pt", 51275: "ptt",
    50813: "lactate", 50885: "bilirubin", 50862: "albumin",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _to_block_grid(per_hadm_df: pd.DataFrame, hadm_id, intime, n_hours, n_blocks, columns):
    """Pivot one admission's events into a (n_blocks, len(columns)) array via mean.

    Unobserved cells are NaN (not 0) so the downstream ffill + population-mean
    imputation can distinguish "missing" from "observed zero".
    """
    grid = np.full((n_blocks, len(columns)), np.nan, dtype=np.float32)
    sub = per_hadm_df[per_hadm_df["hadm_id"] == hadm_id]
    if sub.empty:
        return grid
    sub = sub.copy()
    sub["hour"] = ((sub["charttime"] - intime).dt.total_seconds() // 3600).astype(int)
    sub = sub[(sub["hour"] >= 0) & (sub["hour"] < n_hours)]
    if sub.empty:
        return grid
    sub["block"] = sub["hour"] // (n_hours // n_blocks)
    pivot = sub.pivot_table(index="block", columns="itemid", values="valuenum", aggfunc="mean")
    for j, c in enumerate(columns):
        if c in pivot.columns:
            for b in pivot.index:
                v = pivot.at[b, c]
                if not pd.isna(v):
                    grid[int(b), j] = float(v)
    return grid


def _ffill_grid(grid: np.ndarray) -> np.ndarray:
    """Forward fill across the time axis (axis 0) within each feature.

    Uses NaN sentinel rather than zero so that "observed but zero" values
    aren't confused with "unobserved". The caller should pass a grid with
    NaN in unobserved cells; the output has NaN wherever no prior
    observation exists (handled later by train-set population-mean impute).
    """
    out = grid.copy()
    for j in range(out.shape[1]):
        last = np.nan
        for i in range(out.shape[0]):
            if not np.isnan(out[i, j]):
                last = out[i, j]
            else:
                out[i, j] = last
    return out


# ----------------------------------------------------------------------------
# Cohort builder
# ----------------------------------------------------------------------------

def _build_cohort(raw_dir: Path, min_los_hours: float) -> pd.DataFrame:
    print("  Loading icustays / admissions / patients ...")
    icustays = pd.read_csv(raw_dir / "icustays.csv",
                           usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    admissions = pd.read_csv(raw_dir / "admissions.csv",
                             usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                                      "deathtime", "race"])
    patients = pd.read_csv(raw_dir / "patients.csv",
                           usecols=["subject_id", "anchor_age", "anchor_year", "gender"])

    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"])
    cohort = cohort.merge(patients, on="subject_id")
    for c in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
        cohort[c] = pd.to_datetime(cohort[c], errors="coerce")
    cohort["age"] = cohort["anchor_age"] + (cohort["intime"].dt.year - cohort["anchor_year"])
    cohort["los_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0

    # First ICU stay per hadm, age 18+, LOS >= min_los
    cohort = cohort.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    cohort = cohort[(cohort["age"] >= 18) & (cohort["los_hours"] >= min_los_hours)]
    print(f"  Cohort after filters: {len(cohort)}")
    return cohort.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Vitals + labs extraction
# ----------------------------------------------------------------------------

def _cached_stream_filter(path: Path, hadm_ids: set, itemids: list[int],
                           cache_file: Path | None, source_label: str
                           ) -> pd.DataFrame:
    """Stream a large MIMIC events CSV in chunks, keeping rows whose
    ``hadm_id`` is in cohort and ``itemid`` is in the requested list.
    Optionally caches the filtered DataFrame to ``cache_file`` so that
    subsequent seed runs skip the ~40 GB CSV pass.

    Cache validity key is ``(hadm_ids_set, itemids_set)``; if either
    differs from the call, the cache is invalidated and rebuilt.
    """
    import pickle as _pk
    if cache_file is not None and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                blob = _pk.load(f)
            cached_hadm = set(blob.get("hadm_ids", []))
            cached_item = set(blob.get("itemids", []))
            if cached_hadm == set(hadm_ids) and cached_item == set(itemids):
                print(f"  [cache] {source_label}: loading filtered frame "
                      f"from {cache_file.name} ({len(blob['df']):,d} rows)")
                return blob["df"]
            print(f"  [cache-invalid] {source_label}: hadm/itemid key "
                  f"changed — rebuilding")
        except Exception as exc:
            print(f"  [cache-error] {source_label}: {exc} — rebuilding")
    print(f"  Streaming {source_label} (chunked) ...")
    chunks = []
    for chunk in pd.read_csv(path, usecols=["hadm_id", "charttime", "itemid", "valuenum"],
                             chunksize=5_000_000):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if not chunk.empty:
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["hadm_id", "charttime", "itemid", "valuenum"])
    df["charttime"] = pd.to_datetime(df["charttime"])
    print(f"  {source_label} rows kept: {len(df):,d}")
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        blob = {"hadm_ids": list(hadm_ids),
                "itemids": list(itemids),
                "df": df}
        with open(cache_file, "wb") as f:
            _pk.dump(blob, f, protocol=_pk.HIGHEST_PROTOCOL)
        print(f"  [cache-write] {source_label}: saved to {cache_file.name}")
    return df


def _load_chartevents(path: Path, hadm_ids: set, itemids: list[int],
                       cache_dir: Path | None = None) -> pd.DataFrame:
    cache_file = (cache_dir / "chartevents_filtered.pkl") if cache_dir else None
    return _cached_stream_filter(path, hadm_ids, itemids, cache_file,
                                   source_label="chartevents")


def _load_labevents(path: Path, hadm_ids: set, itemids: list[int],
                     cache_dir: Path | None = None) -> pd.DataFrame:
    cache_file = (cache_dir / "labevents_filtered.pkl") if cache_dir else None
    return _cached_stream_filter(path, hadm_ids, itemids, cache_file,
                                   source_label="labevents")


# ----------------------------------------------------------------------------
# Static demographics
# ----------------------------------------------------------------------------

def _build_static(cohort: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = pd.DataFrame(index=cohort.index)
    feats["age"] = cohort["age"].clip(18, 90).astype(float)
    feats = feats.join(pd.get_dummies(cohort["gender"], prefix="gender", dtype=float))
    feats = feats.join(pd.get_dummies(cohort["race"], prefix="race", dtype=float))
    return feats.values.astype(np.float32), list(feats.columns)


# ----------------------------------------------------------------------------
# ICD codes
# ----------------------------------------------------------------------------

def _build_icd(raw_dir: Path, cohort: pd.DataFrame, top_k: int,
               ) -> tuple[np.ndarray, list[str], set]:
    """Per-admission top-K ICD code multi-hot features.

    For each cohort admission, takes ICDs only from the same subject's
    prior admissions (admittime < cohort admittime). The ICD modality
    represents patient history at the time of the index admission.
    """
    if "subject_id" not in cohort.columns or "admittime" not in cohort.columns:
        raise ValueError("cohort must include subject_id and admittime columns")

    print("  Loading diagnoses_icd.csv (prior admissions only) ...")
    icd_raw = pd.read_csv(raw_dir / "diagnoses_icd.csv",
                           usecols=["subject_id", "hadm_id", "icd_code"])
    adm = pd.read_csv(raw_dir / "admissions.csv",
                       usecols=["hadm_id", "admittime"])
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm = adm.rename(columns={"admittime": "icd_admittime"})
    icd_raw = icd_raw.merge(adm, on="hadm_id", how="left")
    cohort_idx = (cohort[["subject_id", "hadm_id", "admittime"]]
                   .rename(columns={"hadm_id": "cohort_hadm_id",
                                     "admittime": "cohort_admittime"}))
    cohort_idx["cohort_admittime"] = pd.to_datetime(cohort_idx["cohort_admittime"])
    merged = icd_raw.merge(cohort_idx, on="subject_id", how="inner")
    merged = merged[merged["icd_admittime"] < merged["cohort_admittime"]]
    icd = merged[["cohort_hadm_id", "icd_code"]].rename(
        columns={"cohort_hadm_id": "hadm_id"})
    n_with_prior = icd["hadm_id"].nunique()
    print(f"    {n_with_prior:,d}/{len(cohort):,d} cohort admissions "
          f"({100*n_with_prior/max(len(cohort),1):.1f}%) have prior ICDs")

    top_codes = icd["icd_code"].value_counts().nlargest(top_k).index.tolist()
    icd = icd[icd["icd_code"].isin(top_codes)]
    icd_oh = pd.crosstab(icd["hadm_id"], icd["icd_code"]).clip(upper=1)

    icd_full = pd.DataFrame(0, index=cohort["hadm_id"].values, columns=top_codes)
    common = icd_oh.index.intersection(icd_full.index)
    icd_full.loc[common, icd_oh.columns] = icd_oh.loc[common].values
    hadm_ids_with_icd = set(icd["hadm_id"].unique())
    return (icd_full.values.astype(np.float32),
            [f"ICD_{c}" for c in top_codes],
            hadm_ids_with_icd)


# ----------------------------------------------------------------------------
# Radiology embeddings (Clinical-Longformer)
# ----------------------------------------------------------------------------

def _build_radiology(raw_dir: Path, cohort: pd.DataFrame, cache_dir: Path,
                     model_name: str, max_length: int
                     ) -> tuple[np.ndarray, list[str], set]:
    print("  Loading radiology.csv ...")
    rads = pd.read_csv(raw_dir / "radiology.csv", usecols=["hadm_id", "charttime", "text"])
    rads = rads.dropna(subset=["hadm_id", "text"])
    rads = rads[rads["hadm_id"].isin(cohort["hadm_id"])]
    rads["charttime"] = pd.to_datetime(rads["charttime"], errors="coerce")

    # Earliest report per admission (before or at intime + min_los_hours)
    intime_lookup = dict(zip(cohort["hadm_id"], pd.to_datetime(cohort["intime"])))
    rads["intime"] = rads["hadm_id"].map(intime_lookup)
    rads = rads[rads["charttime"] >= rads["intime"]]
    rads = rads.sort_values("charttime").drop_duplicates("hadm_id", keep="first")

    text_lookup = dict(zip(rads["hadm_id"], rads["text"]))
    texts = [text_lookup.get(h, "") for h in cohort["hadm_id"]]
    hadm_ids_with_rad = set(rads["hadm_id"].unique())

    emb = get_or_compute_embeddings(
        texts=texts,
        cache_path=cache_dir / "rad_embeddings.npy",
        model_name=model_name,
        max_length=max_length,
    )
    cols = [f"RAD_{i}" for i in range(emb.shape[1])]
    return emb, cols, hadm_ids_with_rad


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> dict:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = resolve_path(cfg["data"]["processed_dir"])
    cache_dir = resolve_path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["data"]["seed"]
    n_hours = cfg["data"]["n_hours"]
    n_blocks = cfg["data"]["n_blocks"]
    K = cfg["data"]["num_durations"]
    min_los = cfg["data"]["min_los_hours"]

    np.random.seed(seed)

    print("=" * 65)
    print("  MIMIC-IV preprocessing pipeline")
    print("=" * 65)

    # ---- 1. Cohort -----------------------------------------------------------
    print("\n[1/6] Building cohort ...")
    cohort = _build_cohort(raw_dir, min_los_hours=min_los)
    hadm_ids = set(cohort["hadm_id"].tolist())

    # ---- 2. Survival labels (D25: ICU mortality) ------------------------------
    print("\n[2/6] Survival outcomes (ICU mortality) ...")
    # Duration = ICU length-of-stay in hours (outtime − intime), clipped at 240h.
    durations_all = ((cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
                     ).clip(lower=1.0).values.astype(np.float32)
    # Event = died during the ICU stay (deathtime falls on or before ICU discharge).
    died_in_icu = (pd.notna(cohort["deathtime"]) &
                   (cohort["deathtime"] <= cohort["outtime"])).astype(int)
    events_all = died_in_icu.values.astype(np.int64)
    # Clip at 240h prediction horizon; right-censor events beyond the horizon.
    over_horizon = durations_all > 240.0
    durations_all = np.minimum(durations_all, 240.0)
    events_all[over_horizon] = 0
    print(f"  events>0: {(events_all > 0).sum()}/{len(events_all)} ({(events_all>0).mean()*100:.1f}%)")

    # ---- 3. Vitals + labs (dynamic) ------------------------------------------
    print("\n[3/6] Vitals + labs ...")
    vitals = _load_chartevents(raw_dir / "chartevents.csv", hadm_ids,
                                 list(VITAL_ITEMIDS.keys()), cache_dir=cache_dir)
    labs = _load_labevents(raw_dir / "labevents.csv", hadm_ids,
                             list(LAB_ITEMIDS.keys()), cache_dir=cache_dir)
    hadm_ids_with_dynamic = set(vitals["hadm_id"].unique()) | set(labs["hadm_id"].unique())

    dyn_columns = list(VITAL_ITEMIDS.keys()) + list(LAB_ITEMIDS.keys())
    dyn_names = list(VITAL_ITEMIDS.values()) + list(LAB_ITEMIDS.values())
    n_dyn = len(dyn_columns)

    # Build (N, n_blocks, n_dyn) by per-hadm pivot+block aggregate
    dyn_grid = np.zeros((len(cohort), n_blocks, n_dyn), dtype=np.float32)
    intime_lookup = dict(zip(cohort["hadm_id"], pd.to_datetime(cohort["intime"])))
    combined = pd.concat([vitals, labs], ignore_index=True)
    for i, hadm_id in enumerate(cohort["hadm_id"].values):
        intime = intime_lookup[hadm_id]
        grid = _to_block_grid(combined, hadm_id, intime, n_hours, n_blocks, dyn_columns)
        dyn_grid[i] = _ffill_grid(grid)

    # Standardise on training only after split
    # ---- 4. Static demographics ----------------------------------------------
    print("\n[4/6] Static demographics ...")
    static_arr, static_names = _build_static(cohort)
    n_static = static_arr.shape[1]
    static_grid = np.broadcast_to(static_arr[:, None, :], (len(cohort), n_blocks, n_static)).copy()

    # ---- 5. ICD + radiology ---------------------------------------------------
    print("\n[5/6] ICD codes ...")
    icd_arr, icd_names, hadm_ids_with_icd = _build_icd(
        raw_dir, cohort, top_k=cfg["model"]["top_icd_codes"],
    )
    icd_grid = np.broadcast_to(icd_arr[:, None, :], (len(cohort), n_blocks, icd_arr.shape[1])).copy()

    print("\n  Radiology (Clinical-Longformer) ...")
    rad_arr, rad_names, hadm_ids_with_rad = _build_radiology(
        raw_dir, cohort, cache_dir,
        model_name=cfg["model"]["longformer_model"],
        max_length=cfg["model"]["longformer_max_length"],
    )
    rad_grid = np.broadcast_to(rad_arr[:, None, :], (len(cohort), n_blocks, rad_arr.shape[1])).copy()

    # ---- 6. Concatenate, scale, split, save ----------------------------------
    print("\n[6/6] Concatenate + standardise + split ...")
    x_all = np.concatenate([dyn_grid, static_grid, rad_grid, icd_grid], axis=-1).astype(np.float32)
    feat_names = dyn_names + static_names + rad_names + icd_names
    n_features = x_all.shape[-1]
    print(f"  x_all={x_all.shape}, F={n_features}")

    n = x_all.shape[0]
    train_idx, val_idx, test_idx = stratified_split_indices(
        n, events_all,
        cfg["data"]["splits"]["train"],
        cfg["data"]["splits"]["val"],
        cfg["data"]["splits"]["test"],
        seed=seed,
    )

    # Train-set population-mean imputation on the dynamic block
    # (Option A: ffill only + train-mean fallback — no bfill leakage).
    train_dyn = x_all[train_idx, :, :n_dyn]
    pop_mean = np.nanmean(train_dyn.reshape(-1, n_dyn), axis=0)
    pop_mean = np.where(np.isnan(pop_mean), 0.0, pop_mean)
    nan_mask = np.isnan(x_all[:, :, :n_dyn])
    if nan_mask.any():
        idx = np.where(nan_mask)
        x_all[:, :, :n_dyn][idx] = pop_mean[idx[2]]
    print(f"  NaN in dyn block after impute: {np.isnan(x_all[:, :, :n_dyn]).sum()}")

    # Standardise dynamic features (rad/icd left as-is)
    scaler = StandardScaler()
    flat_train = x_all[train_idx, :, :n_dyn].reshape(-1, n_dyn)
    scaler.fit(flat_train)
    x_all[:, :, :n_dyn] = scaler.transform(x_all[:, :, :n_dyn].reshape(-1, n_dyn)).reshape(n, n_blocks, n_dyn)

    # D21: Standardise static features separately (between dynamic and rad blocks).
    # Static cols sit at [n_dyn : n_dyn + n_static]; rad starts right after.
    stat_lo = n_dyn
    stat_hi = n_dyn + n_static
    stat_scaler = StandardScaler()
    flat_train_stat = x_all[train_idx, :, stat_lo:stat_hi].reshape(-1, stat_hi - stat_lo)
    stat_scaler.fit(flat_train_stat)
    x_all[:, :, stat_lo:stat_hi] = stat_scaler.transform(
        x_all[:, :, stat_lo:stat_hi].reshape(-1, stat_hi - stat_lo)
    ).reshape(n, n_blocks, stat_hi - stat_lo)

    # Quantile cuts
    from .base import make_quantile_cuts, assign_bin_indices
    cuts = make_quantile_cuts(durations_all[train_idx], K)
    bins = assign_bin_indices(durations_all, cuts)

    # Per-admission modality mask, computed from which source CSVs had
    # entries for each hadm_id (before any zero-fill / imputation).
    hadm_ids_arr = cohort["hadm_id"].values
    has_dynamic = np.array([h in hadm_ids_with_dynamic for h in hadm_ids_arr])
    has_static = np.ones(len(hadm_ids_arr), dtype=bool)
    has_icd = np.array([h in hadm_ids_with_icd for h in hadm_ids_arr])
    has_rad = np.array([h in hadm_ids_with_rad for h in hadm_ids_arr])
    modality_mask_all = np.stack([has_dynamic, has_static, has_icd, has_rad], axis=1)
    print(f"  modality mask: dyn={has_dynamic.sum()} stat={has_static.sum()} "
          f"icd={has_icd.sum()} rad={has_rad.sum()} (of {len(hadm_ids_arr)})")

    splits = {}
    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[split] = {
            "x": x_all[idx],
            "durations_idx": bins[idx],
            "durations_raw": durations_all[idx],
            "events": events_all[idx],
            "modality_mask": modality_mask_all[idx],
            "pids": hadm_ids_arr[idx],
        }

    save_split(out_dir, "MIMIC", splits, cuts, feature_names=feat_names,
               modality_keys=["dynamic", "static", "icd", "rad"])
    print_audit("MIMIC", splits, cuts, n_features=n_features, n_blocks=n_blocks)

    return {
        "name": "mimic",
        "n_total": n,
        "shape_train": splits["train"]["x"].shape,
        "out_dir": str(out_dir),
    }
