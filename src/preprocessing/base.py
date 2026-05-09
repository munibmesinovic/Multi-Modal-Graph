"""Shared preprocessing utilities: splits, discretization, audit reporting."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# Train/val/test split
# ----------------------------------------------------------------------------

def stratified_split_indices(
    n: int, events: np.ndarray, frac_train: float, frac_val: float, frac_test: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patient-level stratified split using event label for stratification.
    Returns (train_idx, val_idx, test_idx) — disjoint, covering [0, n).
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    idx = np.arange(n)
    strat = (events > 0).astype(int)

    train_idx, temp_idx = train_test_split(
        idx, test_size=(frac_val + frac_test), random_state=seed,
        stratify=strat,
    )
    rel = frac_test / (frac_val + frac_test)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel, random_state=seed,
        stratify=strat[temp_idx],
    )
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return train_idx, val_idx, test_idx


# ----------------------------------------------------------------------------
# Discrete-time bin transform
# ----------------------------------------------------------------------------

def make_quantile_cuts(durations_train: np.ndarray, num_durations: int,
                       events_train: np.ndarray | None = None,
                       scheme: str = "quantiles") -> np.ndarray:
    """Discrete-time bin edges via pycox LabTransDiscreteTime.

    Matches the golden reference convention: pycox places cuts using
    event-only quantiles when scheme='quantiles' (the default), not all
    durations. K=10 by default per paper §3.1.
    """
    import scipy.integrate
    if not hasattr(scipy.integrate, "simps"):
        scipy.integrate.simps = scipy.integrate.simpson
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime

    if events_train is None:
        # Fall back to treating every patient as an event so we can still
        # produce cuts when caller didn't pass events.
        events_train = np.ones_like(durations_train, dtype=np.int64)

    lt = LabTransDiscreteTime(num_durations, scheme=scheme)
    lt.fit(durations_train.astype(np.float64), events_train.astype(np.int64))
    return lt.cuts.astype(np.float64)


def assign_bin_indices(durations: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Map continuous durations to bin index in [0, K-1]."""
    idx = np.searchsorted(cuts, durations, side="right") - 1
    idx = np.clip(idx, 0, len(cuts) - 2)
    return idx.astype(np.int64)


# ----------------------------------------------------------------------------
# Save / load split outputs
# ----------------------------------------------------------------------------

def save_split(
    out_dir: Path,
    name: str,
    splits: dict[str, dict],
    cuts: np.ndarray,
    feature_names: Sequence[str] | None = None,
    extra: dict | None = None,
    modality_keys: Sequence[str] | None = None,
) -> None:
    """Persist a processed split to ``out_dir``.

    splits: {"train": {"x": ..., "durations_idx": ..., "events": ...,
                       "durations_raw": ..., "modality_mask": (N, M),
                       "pids": (N,)}, ...}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, data in splits.items():
        np.save(out_dir / f"x_{split}_{name}.npy", data["x"].astype(np.float32))
        if "durations_raw" in data:
            np.save(out_dir / f"durations_{split}_{name}.npy", data["durations_raw"].astype(np.float64))
        np.save(out_dir / f"events_{split}_{name}.npy", data["events"].astype(np.int64))
        # y_*_surv_*.p stores (durations_raw_hours, events) to match the
        # golden reference convention. The training-time dataset class is
        # responsible for binning durations into K discrete buckets via cuts.
        with open(out_dir / f"y_{split}_surv_{name}.p", "wb") as f:
            durations_for_y = (
                data["durations_raw"].astype(np.float64)
                if "durations_raw" in data
                else data["durations_idx"].astype(np.float64)
            )
            pickle.dump((durations_for_y, data["events"].astype(np.int64)), f)

        # Optional modality mask: (N, M) boolean, 1 if modality present.
        if "modality_mask" in data:
            np.save(
                out_dir / f"modality_mask_{split}_{name}.npy",
                data["modality_mask"].astype(np.uint8),
            )
        # Optional patient IDs — useful for audit / cross-linking with raw data.
        if "pids" in data:
            np.save(out_dir / f"pids_{split}_{name}.npy", np.asarray(data["pids"]))

    np.save(out_dir / f"cuts_{name}.npy", cuts.astype(np.float64))
    # pycox LabTransDiscreteTime(K) returns a K-element ``cuts`` array and the
    # model output is K-dimensional (one hazard per discrete time point).
    np.save(out_dir / f"out_features_{name}.npy", np.array(len(cuts), dtype=np.int64))

    if feature_names is not None:
        with open(out_dir / f"feature_names_{name}.p", "wb") as f:
            pickle.dump(list(feature_names), f)

    if modality_keys is not None:
        with open(out_dir / f"modality_keys_{name}.p", "wb") as f:
            pickle.dump(list(modality_keys), f)

    if extra is not None:
        with open(out_dir / f"meta_{name}.json", "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()}, f, indent=2)


def print_audit(name: str, splits: dict, cuts: np.ndarray, n_features: int, n_blocks: int) -> None:
    n_total = sum(s["x"].shape[0] for s in splits.values())
    print()
    print("=" * 65)
    print(f"  {name.upper()} preprocessing audit")
    print("=" * 65)
    print(f"  Total patients : {n_total}")
    for split, data in splits.items():
        x = data["x"]
        e = data["events"]
        ev_pct = (e > 0).mean() * 100
        print(f"  {split:5s}: x={x.shape}  events>0: {(e>0).sum()}/{len(e)} ({ev_pct:.1f}%)")
    print(f"  Time grid     : {n_blocks} blocks")
    print(f"  Features/step : {n_features}")
    print(f"  Bins (K)      : {len(cuts)}")
    print(f"  Cuts          : {cuts.tolist()}")
    print("=" * 65)
