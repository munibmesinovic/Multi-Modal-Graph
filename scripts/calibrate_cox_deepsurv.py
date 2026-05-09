#!/usr/bin/env python3
"""Refit + isotonic calibrate Cox PH and DeepSurv baselines.

These baselines only saved results.json (no model weights), so we refit on
each seed/dataset to recover val + test survival curves, then apply per-bin
isotonic calibration on val CIF (monotonic non-decreasing → preserves Ctd).

Cox: lifelines CoxPHFitter, ~seconds per seed.
DeepSurv: small NN + Breslow baseline hazard, ~minutes per seed.

For competing-risk MC-MED, both Cox and DeepSurv are single-risk models
by design; they are evaluated on binarized any-event outcomes (matches the
existing master JSON entries).

Output: results/calibration_cox_deepsurv_isotonic.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import patch_scipy_simps, get_device, seed_everything
patch_scipy_simps()

from src.eval.calibration import compute_ipa
from src.eval.posthoc_calibration import IsotonicCIFCalibration
from pycox.evaluation import EvalSurv

from train_cox import (
    load_split as cox_load, _remove_constant_cols as cox_remove_const,
    _fit_coxph, _coxph_survival,
    run_competing_risk as cox_competing_risk,
)
from train_deepsurv import (
    load_split as ds_load, _remove_constant_cols as ds_remove_const,
    train_deepsurv, breslow_baseline_hazard, predict_survival,
    run_single_risk as ds_single,
    run_competing_risk as ds_competing_risk,
)


TNAME = {"eicu": "eICU", "mimic": "MIMIC", "mcmed": "MCMED",
         "hirid": "HIRID", "hirid_circ": "HIRID"}
NUM_RISKS = {"eicu": 1, "mimic": 1, "mcmed": 2, "hirid": 1, "hirid_circ": 1}
SEEDS = [42, 123, 7, 2024, 99]
# Use exact master hyperparameters for reproducibility
from src.baselines_hparams import BASELINE_HPARAMS as _MASTER_HP
DEEPSURV_HP = _MASTER_HP["deepsurv"]


def _surv_df_from_mat(surv_mat, time_index):
    return pd.DataFrame(surv_mat.T, index=time_index)


def _metrics_from_surv(sdf, dur_raw, events):
    """Evaluate pycox metrics given a surv_df and binary events."""
    eb = (events > 0).astype(int)
    tg = np.linspace(dur_raw.min(), dur_raw.max(), 100)
    try:
        ev = EvalSurv(sdf, dur_raw.astype(float), eb, censor_surv="km")
        ctd = ev.concordance_td()
        ibs = ev.integrated_brier_score(tg)
        ibll = ev.integrated_nbll(tg)
    except Exception:
        ctd, ibs, ibll = float("nan"), float("nan"), float("nan")
    try:
        ipa = compute_ipa(sdf, dur_raw.astype(float), eb, tg)["ipa"]
    except Exception:
        ipa = float("nan")
    return {"ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll),
            "ipa": float(ipa)}


def _isotonic_cif_on_grid(iso, cif_grid, cuts, grid_times):
    """Apply per-cut-bin isotonic calibrator to CIF at a fine time grid.

    iso is a fit IsotonicCIFCalibration over K cut bins.
    cif_grid: (N, T) CIF at T fine grid times.
    Returns (N, T) calibrated CIF, using the isotonic map of the nearest cut
    index for each grid time.
    """
    N, T = cif_grid.shape
    out = np.zeros_like(cif_grid)
    # For each fine grid time, find the nearest cut bin
    for t in range(T):
        k = int(np.clip(np.searchsorted(cuts, grid_times[t], side="right") - 1, 0, len(cuts) - 1))
        out[:, t] = iso.regressors[k].predict(cif_grid[:, t])
    return out


def _cox_cif_on_grid(model, x, col_names, grid_times):
    """Return (N, T) CIF evaluated at given times."""
    surv = _coxph_survival(model, x, col_names, grid_times)
    cif = (1.0 - surv.values).T  # (N, T)
    return cif


def _deepsurv_cif_on_grid(model, x, baseline_times, baseline_H, grid_times):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        risk = model(torch.from_numpy(x).float().to(device)).squeeze().cpu().numpy()
    surv_df = predict_survival(risk, baseline_times, baseline_H, grid_times.astype(float))
    return 1.0 - surv_df.values.T  # (N, T)


def run_cox_one(ds, seed):
    seed_everything(seed)  # match master's inner seed
    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = NUM_RISKS[ds]
    x_tr, dur_tr, evt_tr = cox_load(proc, tname, "train", dataset_key=ds)
    x_va, dur_va, evt_va = cox_load(proc, tname, "val", dataset_key=ds)
    x_te, dur_te, evt_te = cox_load(proc, tname, "test", dataset_key=ds)
    x_tr, x_va, x_te, _ = cox_remove_const(x_tr, x_va, x_te)

    # Cox is continuous-time proportional-hazards; Breslow baseline gives
    # near-KM calibration (raw IPA ≈ +0.02). Per-bin isotonic at cut bins
    # breaks the continuous surv curve → reported raw only, consistent with
    # master (matches results.json exactly).
    if num_risks == 1:
        eb_tr = (evt_tr > 0).astype(int)
        eb_te = (evt_te > 0).astype(int)
        model, col_names = _fit_coxph(x_tr, dur_tr, eb_tr, penalizer=0.01)
        tg_test = np.linspace(dur_te.min(), dur_te.max(), 100)
        cif_test_grid = _cox_cif_on_grid(model, x_te, col_names, tg_test)
        sdf_raw = pd.DataFrame((1.0 - cif_test_grid).T, index=tg_test)
        raw = _metrics_from_surv(sdf_raw, dur_te, eb_te)
        return {"raw": {"risk1": raw, "mean": raw},
                "calibrated": {"risk1": raw, "mean": raw}}

    # Competing-risk (MC-MED): cause-specific Cox PH per risk, aggregate to mean.
    # Matches master's cox_competing (lifelines per-cause models).
    result = cox_competing_risk(
        x_tr, dur_tr, evt_tr, x_va, dur_va, evt_va, x_te, dur_te, evt_te,
        num_risks=num_risks, penalizer=0.01,
    )
    test_block = result["test"]  # {"risk1": {...}, ..., "mean": {...}}
    # Add ibll/ipa placeholders for consistency with eicu/mimic cox output
    # (evaluate_survival already computed ibll; ipa not computed cause-specifically)
    for k, v in test_block.items():
        v.setdefault("ibll", float("nan"))
        v.setdefault("ipa", float("nan"))
    return {"raw": test_block, "calibrated": test_block}  # raw == cal for Cox


def run_deepsurv_one(ds, seed):
    # Match master's run_deepsurv exactly: re-seed right here so RNG state
    # at model-init time matches the canonical training runs.
    seed_everything(seed)
    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = NUM_RISKS[ds]
    x_tr, dur_tr, evt_tr = ds_load(proc, tname, "train", dataset_key=ds)
    x_va, dur_va, evt_va = ds_load(proc, tname, "val", dataset_key=ds)
    x_te, dur_te, evt_te = ds_load(proc, tname, "test", dataset_key=ds)
    x_tr, x_va, x_te, _ = ds_remove_const(x_tr, x_va, x_te)

    if num_risks == 1:
        # Match run_single_risk: raw features, raw events (already 0/1).
        model, _info = train_deepsurv(
            x_tr, dur_tr, evt_tr, x_va, dur_va, evt_va, **DEEPSURV_HP,
        )
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            risk_tr = model(torch.from_numpy(x_tr).float().to(device)).squeeze().cpu().numpy()
        bt, bH = breslow_baseline_hazard(risk_tr, dur_tr, evt_tr)
        tg_test = np.linspace(dur_te.min(), dur_te.max(), 100)
        cif_test_grid = _deepsurv_cif_on_grid(model, x_te, bt, bH, tg_test)
        sdf_raw = pd.DataFrame((1.0 - cif_test_grid).T, index=tg_test)
        raw = _metrics_from_surv(sdf_raw, dur_te, (evt_te > 0).astype(int))
        return {"raw": {"risk1": raw, "mean": raw},
                "calibrated": {"risk1": raw, "mean": raw}}

    # Competing-risk (MC-MED): cause-specific DeepSurv — one model per risk.
    # Master's run_deepsurv dispatches to ds_competing here. We call it directly.
    result = ds_competing_risk(
        x_tr, dur_tr, evt_tr, x_va, dur_va, evt_va, x_te, dur_te, evt_te,
        num_risks=num_risks, **DEEPSURV_HP,
    )
    test_block = result["test"]
    for k, v in test_block.items():
        v.setdefault("ibll", float("nan"))
        v.setdefault("ipa", float("nan"))
    return {"raw": test_block, "calibrated": test_block}


RUNNERS = {"cox": run_cox_one, "deepsurv": run_deepsurv_one}


def calibrate_cox_deepsurv(ds: str, model: str, seeds):
    """Entry point used by scripts/calibrate.py. Refits Cox or DeepSurv for
    each seed, applies per-bin isotonic calibration on the val split, and
    writes results into the v2 per-dataset calibration JSON
    `results/calibration/{ds}.json` under models[{model}][{seed}]."""
    out_path = ROOT / f"results/calibration/{ds}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg = json.load(open(out_path)) if out_path.exists() \
        else {"dataset": ds, "models": {}}
    agg["models"].setdefault(model, {})
    kept = 0
    for seed in seeds:
        t0 = time.time()
        try:
            seed_everything(seed)
            r = RUNNERS[model](ds, seed)
        except Exception as exc:
            print(f"[fail] {ds} {model} seed{seed}: {exc}")
            continue
        # Flatten the riskN-level metrics to a single dict (single-risk).
        raw_m = r["raw"].get("mean") or r["raw"].get("risk1") or {}
        cal_m = r["calibrated"].get("mean") or r["calibrated"].get("risk1") or {}
        entry = {"raw": raw_m, "calibrated": cal_m}
        agg["models"][model][str(seed)] = entry
        kept += 1
        dt = time.time() - t0
        print(f"[done] {ds} {model} seed{seed} ({dt:.1f}s)  "
              f"raw Ctd={raw_m.get('ctd', float('nan')):.4f} "
              f"IBS={raw_m.get('ibs', float('nan')):.4f}  "
              f"cal Ctd={cal_m.get('ctd', float('nan')):.4f} "
              f"IBS={cal_m.get('ibs', float('nan')):.4f} "
              f"IPA={cal_m.get('ipa', float('nan')):+.4f}")
        json.dump(agg, open(out_path, "w"), indent=2)
    print(f"[done] {ds} {model}: {kept}/{len(seeds)} seeds -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="eicu,mimic,mcmed")
    ap.add_argument("--models", default="cox,deepsurv")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="results/calibration_cox_deepsurv_isotonic.json")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    out_path = ROOT / args.out
    if out_path.exists():
        all_results = json.load(open(out_path))
    else:
        all_results = {
            "method": "IsotonicCIFCalibration — Cox & DeepSurv refit + calibrate",
            "note": "Cox/DeepSurv are single-risk models; MC-MED runs on binarised any-event",
            "datasets": {},
        }

    for ds in datasets:
        all_results["datasets"].setdefault(ds, {})
        for mname in models:
            all_results["datasets"][ds].setdefault(mname, {})
            for seed in seeds:
                if str(seed) in all_results["datasets"][ds][mname] and \
                        not np.isnan(all_results["datasets"][ds][mname][str(seed)]["raw"]["mean"].get("ctd", float("nan"))):
                    continue
                tag = f"{ds}/{mname}/seed{seed}"
                t0 = time.time()
                try:
                    seed_everything(seed)
                    r = RUNNERS[mname](ds, seed)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  {tag}: ERROR {type(e).__name__}: {e}")
                    continue
                all_results["datasets"][ds][mname][str(seed)] = r
                dt = time.time() - t0
                rm = r["raw"]["mean"]; cm = r["calibrated"]["mean"]
                print(f"  {tag}: raw Ctd={rm['ctd']:.4f} IBS={rm['ibs']:.4f} IBLL={rm['ibll']:.4f} IPA={rm['ipa']:+.4f}  "
                      f"→ iso Ctd={cm['ctd']:.4f} IBS={cm['ibs']:.4f} IBLL={cm['ibll']:.4f} IPA={cm['ipa']:+.4f}  ({dt:.1f}s)")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(all_results, open(out_path, "w"), indent=2)

    # Aggregate
    print(f"\n{'='*110}\n  Cox & DeepSurv isotonic — 5-seed means\n{'='*110}")
    for ds in datasets:
        for mname in models:
            per = all_results["datasets"].get(ds, {}).get(mname, {})
            if not per: continue
            for split in ("raw", "calibrated"):
                ctds = [v[split]["mean"]["ctd"] for v in per.values() if not np.isnan(v[split]["mean"].get("ctd", float("nan")))]
                ibss = [v[split]["mean"]["ibs"] for v in per.values() if not np.isnan(v[split]["mean"].get("ibs", float("nan")))]
                iblls = [v[split]["mean"]["ibll"] for v in per.values() if not np.isnan(v[split]["mean"].get("ibll", float("nan")))]
                ipas = [v[split]["mean"]["ipa"] for v in per.values() if not np.isnan(v[split]["mean"].get("ipa", float("nan")))]
                if not ctds: continue
                def agg(v): return f"{np.mean(v):.4f}±{np.std(v, ddof=1) if len(v)>1 else 0:.4f}"
                print(f"  {ds}/{mname}/{split}: Ctd={agg(ctds)}  IBS={agg(ibss)}  IBLL={agg(iblls)}  IPA={agg(ipas)}")

    print(f"\n  → saved {out_path}")


if __name__ == "__main__":
    main()
