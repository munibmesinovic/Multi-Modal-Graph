"""Calibration & discrimination analysis beyond raw IBS.

Added per D42 to address the IBS conservatism bias — models predicting
population-average curves get artificially low IBS while patient-specific
models are penalized for personalisation.

Metrics:
  - IPA (Index of Prediction Accuracy): 1 - IBS/IBS_KM (Kattan & Gerds 2018)
  - Cumulative/dynamic AUC(t): time-dependent discrimination (Uno et al. 2007)
  - Calibration curves: predicted vs observed per decile
  - D-calibration: chi-squared uniformity test (Haider et al. 2020)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def gerds_decomposition(
    surv: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    time_grid: np.ndarray,
) -> dict:
    """Decompose IBS into discrimination and calibration components.

    Uses the Gerds & Schumacher (2006) decomposition:
        IBS = IBS_marginal - Discrimination + Calibration_excess
    where:
        - IBS_marginal is the IBS of the Kaplan-Meier (population-average) estimator
        - Discrimination measures how much the model improves on the marginal
        - Calibration_excess measures how much worse calibration is vs marginal

    Equivalently: IBS = IBS_marginal - (Discrimination - Calibration_excess)
    So: a model with high discrimination and low calibration excess beats KM.

    Args:
        surv: (N, K) survival probabilities S(t) = 1 - CIF(t) at each time point
        durations: (N,) raw event/censor times
        events: (N,) event indicators (0=censored, 1=event)
        time_grid: (T,) time points for integration

    Returns:
        dict with keys: ibs_total, ibs_marginal, discrimination, calibration_excess,
        skill_score (= 1 - IBS/IBS_marginal, analogous to R²)
    """
    from pycox.evaluation import EvalSurv
    from lifelines import KaplanMeierFitter

    N = len(durations)
    K = surv.shape[1]

    # Total IBS from the model
    time_index = np.linspace(surv.shape[1], 1, surv.shape[1])  # dummy; replaced below
    # We need actual time_index matching the surv columns

    # Fit KM on the same data
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)

    # Evaluate KM survival at the time_grid points
    km_surv_at_grid = kmf.survival_function_at_times(time_grid).values.flatten()

    # For model IBS: interpolate surv at time_grid points
    # surv columns correspond to discrete time bins; we need a time_index
    # This function expects pre-computed surv_df with proper time index
    # Use the wrapper below instead
    raise NotImplementedError("Use gerds_decomposition_from_eval() instead")


def compute_ipa(
    surv_df: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_grid: np.ndarray,
) -> dict:
    """Compute IPA (Index of Prediction Accuracy) and IBS decomposition.

    IPA = 1 - IBS_model / IBS_KM   (Kattan & Gerds 2018)

    Analogous to R² in regression: measures improvement over the "predict
    the population-average KM curve for everyone" baseline.

    - IPA = 0: model is no better than the KM marginal (no personalisation)
    - IPA > 0: model improves on KM (useful patient-specific predictions)
    - IPA < 0: model is worse than just predicting the marginal curve
    - IPA = 1: perfect predictions

    This metric properly rewards personalisation: a model that predicts
    identical curves for all patients gets IPA ≈ 0 regardless of how
    well-calibrated those curves are on average.

    Args:
        surv_df: (K, N) DataFrame with time index as rows, patients as columns
                 (pycox convention: S(t) values, index = time points)
        durations: (N,) raw event/censor times
        events: (N,) binary event indicators
        time_grid: (T,) integration time points

    Returns:
        dict with: ibs_model, ibs_km, ipa
    """
    from pycox.evaluation import EvalSurv
    from lifelines import KaplanMeierFitter

    N = len(durations)

    # Model IBS
    ev_model = EvalSurv(surv_df, durations, events, censor_surv="km")
    ibs_model = ev_model.integrated_brier_score(time_grid)

    # KM (marginal) IBS — predict same curve for everyone
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    km_surv = kmf.survival_function_at_times(surv_df.index).values.flatten()

    # Build a surv_df where every patient gets the KM curve
    km_surv_df = pd.DataFrame(
        np.tile(km_surv.reshape(-1, 1), (1, N)),
        index=surv_df.index,
    )
    ev_km = EvalSurv(km_surv_df, durations, events, censor_surv="km")
    ibs_km = ev_km.integrated_brier_score(time_grid)

    # IPA = 1 - IBS_model / IBS_KM (Kattan & Gerds 2018)
    ipa = 1.0 - (ibs_model / ibs_km) if ibs_km > 0 else 0.0

    return {
        "ibs_model": float(ibs_model),
        "ibs_km": float(ibs_km),
        "ipa": float(ipa),
    }


# Standard reporting time horizons per dataset type
REPORTING_HORIZONS = {
    # ICU datasets: hours
    "eicu":     [24, 48, 72, 120, 240],
    "mimic":    [24, 48, 72, 120, 240],
    # Cancer datasets: days (converted from years)
    "gbsg":     [365, 1095, 1825],       # 1yr, 3yr, 5yr
    "metabric": [365, 1095, 1825],       # 1yr, 3yr, 5yr
    # Longitudinal
    "pbc2":     [365, 1095, 1825],       # 1yr, 3yr, 5yr
    "support":  [30, 90, 180, 365],      # 1mo, 3mo, 6mo, 1yr
    # ED
    "mcmed":    [2, 4, 6, 8],            # hours
}


def cumulative_dynamic_auc(
    surv_df: pd.DataFrame,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    times: np.ndarray | None = None,
    dataset: str | None = None,
    n_times: int = 20,
) -> dict:
    """Cumulative/dynamic AUC at clinically meaningful time points (Uno et al. 2007).

    Reports AUC at standard horizons for each dataset type (e.g., 24h/48h/72h
    for ICU, 1yr/3yr/5yr for cancer). Papers typically report AUC at these
    specific time points rather than a mean across arbitrary grid.

    Uses IPCW from scikit-survival for proper censoring adjustment.

    Args:
        surv_df: (K, N_test) pycox-format survival DataFrame
        durations_train: (N_train,) for censoring model
        events_train: (N_train,) train events
        durations_test: (N_test,) test durations
        events_test: (N_test,) test events
        times: explicit time points (overrides dataset-based defaults)
        dataset: dataset name for standard reporting horizons
        n_times: fallback grid size if no times or dataset specified

    Returns:
        dict with:
            times: (T,) evaluated time points
            auc_values: (T,) AUC at each time point
            mean_auc: scalar mean across time points
            auc_at: {horizon_label: auc} for reporting (e.g. {"24h": 0.85})
    """
    from sksurv.metrics import cumulative_dynamic_auc as _cd_auc

    # Build structured arrays for sksurv
    dt = np.dtype([("event", bool), ("time", float)])
    surv_train = np.array(
        [(bool(e), float(t)) for e, t in zip(events_train, durations_train)],
        dtype=dt,
    )
    surv_test = np.array(
        [(bool(e), float(t)) for e, t in zip(events_test, durations_test)],
        dtype=dt,
    )

    # Determine time points
    if times is None and dataset in REPORTING_HORIZONS:
        times = np.array(REPORTING_HORIZONS[dataset], dtype=float)
        # Filter to times within the observed range
        t_max = np.percentile(durations_test, 95)
        times = times[times <= t_max]
    if times is None or len(times) == 0:
        t_min = np.percentile(durations_test[events_test > 0], 5) if (events_test > 0).any() else durations_test.min()
        t_max = np.percentile(durations_test[events_test > 0], 95) if (events_test > 0).any() else durations_test.max()
        times = np.linspace(t_min, t_max, n_times)

    # For each time point, the risk estimate is 1 - S(t) = CIF(t)
    surv_times = surv_df.index.values.astype(float)
    N_test = surv_df.shape[1]
    risk_estimates = np.zeros((len(times), N_test))

    for i, t in enumerate(times):
        idx = np.searchsorted(surv_times, t, side="right") - 1
        idx = max(0, min(idx, len(surv_times) - 1))
        risk_estimates[i] = 1.0 - surv_df.iloc[idx].values

    # Compute AUC at each time point
    auc_values = []
    valid_times = []
    for i, t in enumerate(times):
        try:
            aucs, _ = _cd_auc(surv_train, surv_test, risk_estimates[i], [t])
            auc_values.append(float(aucs[0]))
            valid_times.append(float(t))
        except Exception:
            continue

    mean_auc = float(np.mean(auc_values)) if auc_values else float("nan")

    # Build labeled dict for reporting
    auc_at = {}
    for t, a in zip(valid_times, auc_values):
        if dataset in ("eicu", "mimic", "mcmed"):
            label = f"{int(t)}h"
        elif t >= 365:
            label = f"{int(t/365)}yr"
        elif t >= 30:
            label = f"{int(t/30)}mo"
        else:
            label = f"{int(t)}d"
        auc_at[label] = a

    return {
        "times": np.array(valid_times),
        "auc_values": np.array(auc_values),
        "mean_auc": mean_auc,
        "auc_at": auc_at,
    }


def calibration_curve(
    surv_df: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_point: float,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve at a specific time point.

    Groups patients into bins by predicted survival probability at `time_point`,
    then computes the observed event rate (KM estimate) within each bin.

    Args:
        surv_df: (K, N) pycox-format survival DataFrame
        durations: (N,)
        events: (N,)
        time_point: time at which to evaluate calibration
        n_bins: number of quantile bins

    Returns:
        dict with: predicted (n_bins,), observed (n_bins,), bin_counts (n_bins,)
    """
    from lifelines import KaplanMeierFitter

    # Get predicted S(t) at the time point for each patient
    # Interpolate if time_point is not exactly in the index
    times = surv_df.index.values.astype(float)
    idx = np.searchsorted(times, time_point, side="right") - 1
    idx = max(0, min(idx, len(times) - 1))
    pred_surv = surv_df.iloc[idx].values  # (N,)

    # Bin patients by predicted survival
    try:
        bin_edges = np.quantile(pred_surv, np.linspace(0, 1, n_bins + 1))
        # Make edges unique
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # All predictions are the same — no calibration curve possible
            return {"predicted": np.array([pred_surv.mean()]),
                    "observed": np.array([float("nan")]),
                    "bin_counts": np.array([len(pred_surv)])}
        bin_indices = np.digitize(pred_surv, bin_edges[1:-1])
    except Exception:
        return {"predicted": np.array([]), "observed": np.array([]),
                "bin_counts": np.array([])}

    predicted_means = []
    observed_survs = []
    bin_counts = []

    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if mask.sum() < 2:
            continue
        predicted_means.append(pred_surv[mask].mean())
        bin_counts.append(int(mask.sum()))

        # KM estimate of S(time_point) within this bin
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(durations[mask], event_observed=events[mask])
            obs = kmf.survival_function_at_times([time_point]).values.flatten()[0]
        except Exception:
            obs = float("nan")
        observed_survs.append(obs)

    return {
        "predicted": np.array(predicted_means),
        "observed": np.array(observed_survs),
        "bin_counts": np.array(bin_counts),
    }


def d_calibration(
    surv_df: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """D-calibration test (Haider et al. 2020).

    Tests whether the predicted probability at each patient's event/censor
    time is uniformly distributed across [0, 1]. A well-calibrated model
    produces p-value > 0.05 (fail to reject uniformity).

    Args:
        surv_df: (K, N) pycox-format survival DataFrame
        durations: (N,)
        events: (N,) binary event indicators

    Returns:
        dict with: statistic (chi-squared), p_value, bin_counts (n_bins,),
        is_calibrated (bool, p > 0.05)
    """
    from scipy.stats import chisquare

    N = len(durations)
    times = surv_df.index.values.astype(float)

    # For each patient, get S(T_i) where T_i is their event/censor time
    pred_at_event = np.zeros(N)
    for i in range(N):
        t_i = durations[i]
        idx = np.searchsorted(times, t_i, side="right") - 1
        idx = max(0, min(idx, len(times) - 1))
        pred_at_event[i] = surv_df.iloc[idx, i]

    # Only use uncensored patients for D-calibration
    uncensored = events > 0
    if uncensored.sum() < 20:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "bin_counts": np.array([]), "is_calibrated": False}

    p_vals = pred_at_event[uncensored]

    # Bin into n_bins equal-width bins on [0, 1]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    observed_counts, _ = np.histogram(p_vals, bins=bin_edges)
    expected_count = uncensored.sum() / n_bins

    stat, p_value = chisquare(observed_counts, f_exp=[expected_count] * n_bins)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "bin_counts": observed_counts.tolist(),
        "is_calibrated": bool(p_value > 0.05),
    }
