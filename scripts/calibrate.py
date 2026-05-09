#!/usr/bin/env python3
"""Per-bin isotonic calibration (Platt-style, monotone non-decreasing).

For a given (dataset, model, seed) triple, loads the trained checkpoint,
runs inference on the val + test splits, fits an isotonic CIF calibrator
per risk on the val fold, and applies it to test. Outputs go to
results/calibration/{dataset}.json as:

    { "dataset": <ds>, "models": { <model>: { <seed>: {raw, calibrated} } } }

Raw Ctd is preserved up to ties by construction (same-bin monotone
regression); IBS and IPA typically improve.

Usage:
    python scripts/calibrate.py --dataset mimic --model mmg --seeds 42,123,7,2024,99
    python scripts/calibrate.py --dataset eicu  --model deephit --seeds 42
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import patch_scipy_simps, load_config, get_device, seed_everything
patch_scipy_simps()

from src.eval.calibration import compute_ipa
from src.eval.posthoc_calibration import IsotonicCIFCalibration
from pycox.evaluation import EvalSurv

from src.baselines_hparams import BASELINE_HPARAMS


TNAME = {"eicu": "eICU", "mimic": "MIMIC", "mcmed": "MCMED",
         "hirid": "HIRID", "hirid_circ": "HIRID"}
DEFAULT_SEEDS = [42, 123, 7, 2024, 99]

# Per-dataset hparam overrides (some checkpoint families were trained with
# dataset-specific hparams). Take precedence over BASELINE_HPARAMS[model].
DATASET_HPARAM_OVERRIDES = {
    "dysurv": {
        "mcmed":      {"encoded_features": 20},
        "hirid":      {"encoded_features": 20},
        "hirid_circ": {"encoded_features": 20},
    },
}


def _hp(model: str, ds: str) -> dict:
    hp = dict(BASELINE_HPARAMS.get(model, {}))
    hp.update(DATASET_HPARAM_OVERRIDES.get(model, {}).get(ds, {}))
    return hp


# ---------------------------------------------------------------------
#  CIF extractors — one per model family. Each returns a dict with
#  cif (N, E, K), dur_raw, dur_idx, events, cuts, num_risks.
# ---------------------------------------------------------------------

def _cif_mmg(ds, seed, split):
    from src.train import build_model
    from src.data.dataset import build_dataloaders
    from src.data import modality_slices_for
    from src.losses import compute_cif

    cfg = load_config(ROOT / f"configs/{ds}.yaml")
    cfg["data"]["seed"] = seed
    cfg["data"]["processed_dir"] = f"data/{ds}/processed_seed{seed}"

    ckpt = ROOT / f"checkpoints/{ds}_mmgraphsurv_seed{seed}/best_model.pth"
    if not ckpt.exists():
        return None

    device = get_device()
    seed_everything(seed)
    model = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()

    loaders = build_dataloaders(cfg)
    slices = modality_slices_for(cfg)
    proc = ROOT / cfg["data"]["processed_dir"]
    cuts = np.load(proc / f"cuts_{TNAME[ds]}.npy").astype(float)

    cifs, du_idx, du_raw, evt = [], [], [], []
    with torch.no_grad():
        for batch in loaders[split]:
            out = model(batch["x"].to(device), modality_slices=slices)
            c = compute_cif(out["logits"], per_cause_hazard=True).cpu().numpy()
            cifs.append(c)
            du_idx.append(batch["durations_idx"].numpy())
            du_raw.append(batch.get("durations_raw", batch["durations_idx"]).numpy())
            evt.append(batch["events"].numpy())
    cif = np.concatenate(cifs, 0)
    events = _maybe_remap_events(ds, np.concatenate(evt, 0).astype(np.int64))
    return {
        "cif": cif,
        "dur_raw": np.concatenate(du_raw, 0).astype(float),
        "dur_idx": np.concatenate(du_idx, 0).astype(np.int64),
        "events":  events,
        "cuts":    cuts,
        "num_risks": cfg["dataset"]["num_risks"],
    }


def _cif_deephit(ds, seed, split):
    from train_deephit import DeepHit, load_split, _remove_constant_cols
    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = _num_risks(ds)
    x_tr, *_ = load_split(proc, tname, "train", dataset_key=ds)
    x, dur_raw, evt, dur_idx, cuts = load_split(proc, tname, split, dataset_key=ds)
    _, _, xc, _ = _remove_constant_cols(x_tr, x_tr, x)
    ckpt = ROOT / f"checkpoints/{ds}_deephit_seed{seed}/best_model.pth"
    if not ckpt.exists():
        return None
    hp = _hp("deephit", ds)
    device = get_device()
    model = DeepHit(xc.shape[1], num_risks, len(cuts),
                    hp["shared_dims"], hp["cs_dims"], hp["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pmf = model(torch.from_numpy(xc).float().to(device)).cpu().numpy()
    cif = np.cumsum(pmf, axis=-1)
    return {"cif": cif, "dur_raw": dur_raw.astype(float),
            "dur_idx": dur_idx, "events": _maybe_remap_events(ds, evt),
            "cuts": cuts.astype(float), "num_risks": num_risks}


def _cif_dynamic_deephit(ds, seed, split):
    from train_dynamic_deephit import DynamicDeepHit, load_split
    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = _num_risks(ds)
    x_tr, *_ = load_split(proc, tname, "train", dataset_key=ds)
    x, dur_raw, evt, dur_idx, cuts = load_split(proc, tname, split, dataset_key=ds)
    mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0)
    std = x_tr.reshape(-1, x_tr.shape[-1]).std(0); std[std < 1e-8] = 1.0
    x_s = (x - mu) / std
    ckpt = ROOT / f"checkpoints/{ds}_dynamic_deephit_seed{seed}/best_model.pth"
    if not ckpt.exists():
        return None
    hp = _hp("dynamic_deephit", ds)
    device = get_device()
    seq_len, in_dim = x_s.shape[1], x_s.shape[2]
    model = DynamicDeepHit(in_features=in_dim, num_risks=num_risks,
                            num_bins=len(cuts),
                            rnn_hidden=hp["rnn_hidden"],
                            rnn_layers=hp["rnn_layers"],
                            cs_hidden=hp["cs_hidden"],
                            dropout=hp["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pmf, _ = model(torch.from_numpy(x_s).float().to(device))
        pmf = pmf.cpu().numpy()
    cif = np.cumsum(pmf, axis=-1)
    return {"cif": cif, "dur_raw": dur_raw.astype(float),
            "dur_idx": dur_idx, "events": _maybe_remap_events(ds, evt),
            "cuts": cuts.astype(float), "num_risks": num_risks}


def _cif_dysurv(ds, seed, split):
    from train_dysurv import DySurv, load_split
    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = _num_risks(ds)
    x_tr, *_ = load_split(proc, tname, "train", dataset_key=ds)
    x, dur_raw, evt, dur_idx, cuts = load_split(proc, tname, split, dataset_key=ds)
    mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0)
    std = x_tr.reshape(-1, x_tr.shape[-1]).std(0); std[std < 1e-8] = 1.0
    x_s = (x - mu) / std
    ckpt = ROOT / f"checkpoints/{ds}_dysurv_seed{seed}/best_model.pth"
    if not ckpt.exists():
        return None
    hp = _hp("dysurv", ds)
    device = get_device()
    seq_len, in_dim = x_s.shape[1], x_s.shape[2]
    model = DySurv(in_features=in_dim, seq_len=seq_len,
                   encoded_features=hp["encoded_features"],
                   out_features=len(cuts),
                   num_risks=num_risks).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    seed_everything(seed)
    with torch.no_grad():
        surv = model.predict_surv(torch.from_numpy(x_s).float().to(device))
    surv_np = surv.cpu().numpy()
    # DySurv's predict_surv returns S(t). Single-risk: (N, K) → CIF (N, 1, K).
    # Competing-risk: (N, K, E) → CIF (N, E, K).
    if surv_np.ndim == 2:
        cif = (1.0 - surv_np)[:, None, :]
    else:
        cif = (1.0 - surv_np).transpose(0, 2, 1)
    return {"cif": cif, "dur_raw": dur_raw.astype(float),
            "dur_idx": dur_idx, "events": _maybe_remap_events(ds, evt),
            "cuts": cuts.astype(float), "num_risks": num_risks}


def _cif_survtrace(ds, seed, split):
    """SurvTRACE: static-input transformer (Wang & Sun, 2022). Same load_split
    as DeepHit, but rebuild the BERT-style model and run predict_surv per risk."""
    import os as _os
    _st_root = _os.environ.get("SURVTRACE_ROOT")
    if not _st_root:
        raise RuntimeError(
            "SURVTRACE_ROOT env var is not set; required to evaluate the "
            "SurvTRACE baseline. Set it to a clone of "
            "https://github.com/RyanWangZf/SurvTRACE.")
    sys.path.insert(0, str(Path(_st_root)))
    import numpy as _np
    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf  # SurvTRACE assumes numpy<2.0
    from train_survtrace import (load_split, _remove_constant_cols,
                                  _make_feature_df, build_st_config)
    from survtrace.model import SurvTraceSingle, SurvTraceMulti

    proc = ROOT / f"data/{ds}/processed_seed{seed}"
    tname = TNAME[ds]
    num_risks = _num_risks(ds)

    x_tr, *_ = load_split(proc, tname, "train", dataset_key=ds)
    x, dur_raw, evt, cuts = load_split(proc, tname, split, dataset_key=ds)
    _, _, xc, _ = _remove_constant_cols(x_tr, x_tr, x)

    ckpt = ROOT / f"checkpoints/{ds}_survtrace_seed{seed}/best_model.pth"
    if not ckpt.exists():
        return None

    device = get_device()
    cfg = build_st_config(num_features=xc.shape[1], num_risks=num_risks,
                           cuts=cuts, checkpoint_path=ckpt)
    model = SurvTraceMulti(cfg) if num_risks > 1 else SurvTraceSingle(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    if device.type == "cuda":
        model.cuda(); model.use_gpu = True
    model.eval()

    df = _make_feature_df(xc)
    surv_per_risk = []
    for r in range(num_risks):
        if num_risks == 1:
            s = model.predict_surv(df, batch_size=1024)
        else:
            s = model.predict_surv(df, batch_size=1024, event=r)
        surv_per_risk.append(s.detach().cpu().numpy())
    cif = np.stack([1.0 - s for s in surv_per_risk], axis=1)  # (N, R, K)

    dur_idx = np.searchsorted(cuts, dur_raw, side="right") - 1
    dur_idx = np.clip(dur_idx, 0, len(cuts) - 1).astype(np.int64)

    return {"cif": cif, "dur_raw": dur_raw.astype(float),
            "dur_idx": dur_idx, "events": _maybe_remap_events(ds, evt),
            "cuts": cuts.astype(float), "num_risks": num_risks}


EXTRACTORS = {
    "mmg": _cif_mmg,
    "deephit": _cif_deephit,
    "dynamic_deephit": _cif_dynamic_deephit,
    "dysurv": _cif_dysurv,
    "survtrace": _cif_survtrace,
}


def _num_risks(ds):
    return 2 if ds == "mcmed" else 1


# MC-MED ships 4 raw event classes (0/1/2/3 = censor / ICU adm / Hosp adm /
# Obs). The model trains on the 2-risk collapse {3 -> 2} per
# `dataset.event_collapse` in configs/mcmed.yaml. Apply the same remap
# in-memory at eval time so the test set covers all 6,336 patients
# rather than silently dropping the 889 class-3 events.
def _maybe_remap_events(ds: str, events: np.ndarray) -> np.ndarray:
    if ds != "mcmed":
        return events
    out = events.copy()
    out[out == 3] = 2
    return out




def _metrics(cif, dur_raw, events, cuts, num_risks):
    ctds, ibss, iblls, ipas = [], [], [], []
    for r in range(num_risks):
        e_label = r + 1
        mask = (events == 0) | (events == e_label)
        if mask.sum() < 10 or (events[mask] > 0).sum() < 5:
            continue
        ev_bin = (events[mask] == e_label).astype(int)
        dur_r = dur_raw[mask]
        cif_r = cif[mask, r, :]
        surv_df = pd.DataFrame((1.0 - cif_r).T, index=cuts.astype(float))
        tg = np.linspace(dur_r.min(), dur_r.max(), 100)
        try:
            ev = EvalSurv(surv_df, dur_r, ev_bin, censor_surv="km")
            ctds.append(float(ev.concordance_td()))
            ibss.append(float(ev.integrated_brier_score(tg)))
            iblls.append(float(ev.integrated_nbll(tg)))
            ipas.append(float(compute_ipa(surv_df, dur_r, ev_bin, tg)["ipa"]))
        except Exception:
            pass
    def _m(v): return float(np.mean(v)) if v else float("nan")
    return {"ctd": _m(ctds), "ibs": _m(ibss), "ibll": _m(iblls), "ipa": _m(ipas)}


def _calibrate_one(ds, model, seed):
    ex = EXTRACTORS.get(model)
    if ex is None:
        raise ValueError(f"Unsupported model for calibration: {model}")
    val = ex(ds, seed, "val")
    if val is None:
        return None
    test = ex(ds, seed, "test")
    if test is None:
        return None

    num_risks = val["num_risks"]
    if ds == "mcmed":
        # Competing-risk: Aalen-Johansen recalibration (Alberge 2026).
        # Per-(risk, bin) additive shift fit on val, applied to test.
        # Preserves Ctd exactly; proper for CR (isotonic is not).
        from src.eval.aj_recalibration import fit_shifts, monotonize, apply as aj_apply
        shift = fit_shifts(val["cif"], val["dur_raw"], val["events"], val["cuts"])
        shift = monotonize(shift)
        cif_test_cal = aj_apply(test["cif"], shift)
    else:
        cif_test_cal = test["cif"].copy()
        for r in range(num_risks):
            e_label = r + 1
            mask = (val["events"] == 0) | (val["events"] == e_label)
            if mask.sum() < 10 or (val["events"][mask] > 0).sum() < 5:
                continue
            iso = IsotonicCIFCalibration().fit(
                val["cif"][mask, r, :],
                val["dur_idx"][mask],
                (val["events"][mask] == e_label).astype(int),
            )
            cif_test_cal[:, r, :] = iso.calibrate(test["cif"][:, r, :])

    raw = _metrics(test["cif"],      test["dur_raw"], test["events"], test["cuts"], num_risks)
    cal = _metrics(cif_test_cal,     test["dur_raw"], test["events"], test["cuts"], num_risks)
    return {"raw": raw, "calibrated": cal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TNAME.keys()))
    ap.add_argument("--model", required=True,
                    choices=["mmg", "deephit", "dynamic_deephit", "dysurv",
                             "survtrace", "cox", "deepsurv"])
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ds, model = args.dataset, args.model

    if model in ("cox", "deepsurv"):
        # Cox / DeepSurv: the only "checkpoint" these baselines save is
        # results.json (no refittable state dict). To produce IPA
        # self-sufficiently we refit the model here from the training
        # split, predict on val + test, apply per-bin isotonic calibration
        # on val, evaluate on test. Implemented in scripts/calibrate_cox_deepsurv.py.
        from calibrate_cox_deepsurv import calibrate_cox_deepsurv
        calibrate_cox_deepsurv(ds, model, seeds)
        return

    out_path = ROOT / f"results/calibration/{ds}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg = json.load(open(out_path)) if out_path.exists() \
        else {"dataset": ds, "models": {}}
    agg["models"].setdefault(model, {})

    for s in seeds:
        t0 = time.time()
        try:
            r = _calibrate_one(ds, model, s)
        except Exception as exc:
            print(f"[fail] {ds} {model} seed{s}: {exc}")
            continue
        if r is None:
            print(f"[skip] {ds} {model} seed{s}: ckpt missing")
            continue
        agg["models"][model][str(s)] = r
        rm, cm = r["raw"], r["calibrated"]
        dt = time.time() - t0
        print(f"[done] {ds} {model} seed{s} ({dt:.1f}s)  raw Ctd={rm['ctd']:.4f} IBS={rm['ibs']:.4f}  cal Ctd={cm['ctd']:.4f} IBS={cm['ibs']:.4f}")
        json.dump(agg, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
