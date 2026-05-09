"""Aalen-Johansen recalibration for competing-risk CIFs.

Per Alberge et al. AISTATS 2026 §4.2: the post-hoc per-(risk, bin) shift
``shift[r, k] = AJ_r(τ_k) − mean_val F̂_r(τ_k)`` is fit on a held-out
calibration fold and added to every patient's CIF at test time. Because
the same constant is added at every (r, k), within-bin patient ranking is
preserved, so concordance is preserved exactly (Theorem 3). Isotonic
calibration breaks this for competing risks; AJ recalibration is the
proper analogue.

Used by v2 only for MC-MED, which is the only competing-risk dataset.
"""
from __future__ import annotations

import numpy as np
from lifelines import AalenJohansenFitter


def fit_shifts(cif_val: np.ndarray, dur_raw_val: np.ndarray,
               events_val: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Compute per-(risk, bin) additive shift on the validation fold.

    Args:
        cif_val      (N, R, K) per-patient CIF on val.
        dur_raw_val  (N,) val durations.
        events_val   (N,) val events in {0, 1, ..., R}.
        cuts         (K,) bin upper bounds.

    Returns:
        shift (R, K) — same shift constant per patient at each (r, k).
    """
    R, K = cif_val.shape[1], len(cuts)
    shift = np.zeros((R, K), dtype=float)
    for r in range(R):
        e_label = r + 1
        try:
            ajf = AalenJohansenFitter()
            ajf.fit(dur_raw_val, events_val, event_of_interest=e_label)
            aj_at_cuts = ajf.predict(cuts).to_numpy()
        except Exception as err:
            print(f"  [AJ fit failed for risk{e_label}: {err}] — shift[{r}, :] = 0")
            continue
        model_marginal = cif_val[:, r, :].mean(axis=0)
        shift[r] = aj_at_cuts - model_marginal
    return shift


def monotonize(shift: np.ndarray) -> np.ndarray:
    """Force shift to be non-decreasing along the bin axis so cif + shift
    stays monotonic in time."""
    return np.maximum.accumulate(shift, axis=1)


def apply(cif: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Add per-(risk, bin) shift, clip to [0, 1]. Same constant per patient
    at each (r, k) → preserves Ctd exactly."""
    return np.clip(cif + shift[None, :, :], 0.0, 1.0)
