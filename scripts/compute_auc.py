#!/usr/bin/env python3
"""Time-dependent cumulative/dynamic AUC at clinical horizons.

For every (dataset, model, seed) triple shipped with this tree, runs
inference on the test split, extracts the predicted CIF at each horizon,
and computes AUC(t) using sksurv.metrics.cumulative_dynamic_auc with
IPCW from the training split. Output written to
``results/dynamic_auc.json`` keyed as ``{dataset}_{model}_seed{seed}``.

Dynamic models only — Cox/DeepSurv produce a single static survival
curve so AUC(t) is flat by construction; they are excluded.

Usage:
    python scripts/compute_auc.py
    python scripts/compute_auc.py --datasets eicu,mimic --models mmg --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import patch_scipy_simps, load_config, get_device, seed_everything
patch_scipy_simps()

from src.baselines_hparams import BASELINE_HPARAMS


TNAME = {"eicu": "eICU", "mimic": "MIMIC", "mcmed": "MCMED",
         "hirid": "HIRID", "hirid_circ": "HIRID"}
DEFAULT_DATASETS = ["eicu", "mimic", "mcmed", "hirid", "hirid_circ"]
DEFAULT_MODELS = ["mmg", "deephit", "dynamic_deephit", "dysurv"]
DEFAULT_SEEDS = [42, 123, 7, 2024, 99]

HORIZONS = {
    "eicu":       [24.0, 48.0, 72.0, 120.0, 168.0, 240.0],
    "mimic":      [24.0, 48.0, 72.0, 120.0, 168.0, 240.0],
    "mcmed":      [2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
    "hirid":      [24.0, 48.0, 72.0, 120.0, 168.0, 240.0],
    "hirid_circ": [24.0, 48.0, 72.0, 120.0, 168.0, 240.0],
}

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


def _num_risks(ds: str) -> int:
    return 2 if ds == "mcmed" else 1


# MC-MED has 4 raw event classes; the 2-risk MMG ckpts were trained with
# {3 -> 2}. Apply the same in-memory remap at AUC time so the IPCW from
# the training split and the test labels agree with what the checkpoint
# was fit to predict.
def _maybe_remap_events(ds: str, events: np.ndarray) -> np.ndarray:
    if ds != "mcmed":
        return events
    out = events.copy()
    out[out == 3] = 2
    return out


def _processed_dir(ds: str, seed: int) -> Path:
    return ROOT / f"data/{ds}/processed_seed{seed}"


def _ckpt(ds: str, model: str, seed: int) -> Path:
    model_stem = "mmgraphsurv" if model == "mmg" else model
    return ROOT / f"checkpoints/{ds}_{model_stem}_seed{seed}/best_model.pth"


def _load_train_test(ds: str, seed: int):
    pdir = _processed_dir(ds, seed)
    tname = TNAME[ds]

    def _load_split(split: str):
        x = np.load(pdir / f"x_{split}_{tname}.npy")
        dur_npy = pdir / f"durations_{split}_{tname}.npy"
        if dur_npy.exists():
            dur = np.load(dur_npy).astype(np.float32)
            ev = np.load(pdir / f"events_{split}_{tname}.npy").astype(np.int64)
        else:
            import pickle
            with open(pdir / f"y_{split}_surv_{tname}.p", "rb") as f:
                dur, ev = pickle.load(f)
            dur = np.asarray(dur).astype(np.float32)
            ev = np.asarray(ev).astype(np.int64)
        return x, dur, ev

    x_tr, dur_tr, ev_tr = _load_split("train")
    x_te, dur_te, ev_te = _load_split("test")
    cuts = np.load(pdir / f"cuts_{tname}.npy").astype(float)
    ev_tr = _maybe_remap_events(ds, ev_tr)
    ev_te = _maybe_remap_events(ds, ev_te)
    return x_tr, dur_tr, ev_tr, x_te, dur_te, ev_te, cuts


def _cif_mmg(ds, seed, x_te, cuts):
    from src.train import build_model
    from src.data import modality_slices_for
    from src.losses import compute_cif

    cfg = load_config(ROOT / f"configs/{ds}.yaml")
    cfg["data"]["seed"] = seed
    cfg["data"]["processed_dir"] = f"data/{ds}/processed_seed{seed}"

    ckpt = _ckpt(ds, "mmg", seed)
    device = get_device()
    seed_everything(seed)
    model = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()

    slices = modality_slices_for(cfg)
    x_t = torch.from_numpy(x_te).float().to(device)
    cifs = []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            out = model(x_t[i:i + 256], modality_slices=slices)
            cifs.append(compute_cif(out["logits"], per_cause_hazard=True).cpu().numpy())
    return np.concatenate(cifs, 0)


def _cif_deephit(ds, seed, x_te_unused, cuts):
    from train_deephit import DeepHit, load_split, _remove_constant_cols
    pdir = _processed_dir(ds, seed)
    tname = TNAME[ds]
    x_tr, *_ = load_split(pdir, tname, "train", dataset_key=ds)
    x_te, _, _, _, _ = load_split(pdir, tname, "test", dataset_key=ds)
    _, _, x_te_red, _ = _remove_constant_cols(x_tr, x_tr, x_te)

    hp = _hp("deephit", ds)
    nr = _num_risks(ds)
    device = get_device()
    model = DeepHit(x_te_red.shape[1], nr, len(cuts),
                    hp["shared_dims"], hp["cs_dims"], hp["dropout"]).to(device)
    model.load_state_dict(torch.load(_ckpt(ds, "deephit", seed),
                                      map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pmf = model(torch.from_numpy(x_te_red).float().to(device)).cpu().numpy()
    if pmf.ndim == 2:
        return pmf.cumsum(axis=-1)[:, None, :]
    return pmf.cumsum(axis=-1)


def _cif_dynamic_deephit(ds, seed, x_te_unused, cuts):
    from train_dynamic_deephit import DynamicDeepHit, load_split
    pdir = _processed_dir(ds, seed)
    tname = TNAME[ds]
    x_tr, *_ = load_split(pdir, tname, "train", dataset_key=ds)
    x_te, _, _, _, _ = load_split(pdir, tname, "test", dataset_key=ds)

    mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0)
    std = x_tr.reshape(-1, x_tr.shape[-1]).std(0); std[std < 1e-8] = 1.0
    x_te_s = (x_te - mu) / std

    hp = _hp("dynamic_deephit", ds)
    nr = _num_risks(ds)
    device = get_device()
    model = DynamicDeepHit(in_features=x_te_s.shape[2], num_risks=nr,
                            num_bins=len(cuts),
                            rnn_hidden=hp["rnn_hidden"], rnn_layers=hp["rnn_layers"],
                            cs_hidden=hp["cs_hidden"], dropout=hp["dropout"]).to(device)
    model.load_state_dict(torch.load(_ckpt(ds, "dynamic_deephit", seed),
                                      map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pmf, _ = model(torch.from_numpy(x_te_s).float().to(device))
        pmf = pmf.cpu().numpy()
    if pmf.ndim == 2:
        return pmf.cumsum(axis=-1)[:, None, :]
    return pmf.cumsum(axis=-1)


def _cif_dysurv(ds, seed, x_te_unused, cuts):
    from train_dysurv import DySurv, load_split
    pdir = _processed_dir(ds, seed)
    tname = TNAME[ds]
    x_tr, *_ = load_split(pdir, tname, "train", dataset_key=ds)
    x_te, _, _, _, _ = load_split(pdir, tname, "test", dataset_key=ds)
    mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0)
    std = x_tr.reshape(-1, x_tr.shape[-1]).std(0); std[std < 1e-8] = 1.0
    x_te_s = (x_te - mu) / std

    hp = _hp("dysurv", ds)
    nr = _num_risks(ds)
    device = get_device()
    model = DySurv(in_features=x_te_s.shape[2], seq_len=x_te_s.shape[1],
                   encoded_features=hp["encoded_features"],
                   out_features=len(cuts), num_risks=nr).to(device)
    model.load_state_dict(torch.load(_ckpt(ds, "dysurv", seed),
                                      map_location=device, weights_only=True))
    model.eval()
    seed_everything(seed)
    with torch.no_grad():
        surv = model.predict_surv(torch.from_numpy(x_te_s).float().to(device)).cpu().numpy()
    if surv.ndim == 2:
        return (1.0 - surv)[:, None, :]
    return (1.0 - surv).transpose(0, 2, 1)


INFER = {
    "mmg":             _cif_mmg,
    "deephit":         _cif_deephit,
    "dynamic_deephit": _cif_dynamic_deephit,
    "dysurv":          _cif_dysurv,
}


def _auc_at_horizons(cif_test, cuts, dur_tr, ev_tr, dur_te, ev_te,
                     horizons, target_risk=0):
    from sksurv.metrics import cumulative_dynamic_auc

    ev_tr_bin = (ev_tr == (target_risk + 1)).astype(bool)
    ev_te_bin = (ev_te == (target_risk + 1)).astype(bool)
    tr_struct = np.zeros(len(dur_tr), dtype=[("event", bool), ("time", float)])
    tr_struct["event"] = ev_tr_bin; tr_struct["time"] = dur_tr.astype(float)
    te_struct = np.zeros(len(dur_te), dtype=[("event", bool), ("time", float)])
    te_struct["event"] = ev_te_bin; te_struct["time"] = dur_te.astype(float)

    cif_r = cif_test[:, target_risk, :]
    aucs = {}
    for t in horizons:
        idx = np.clip(np.searchsorted(cuts, t, side="right") - 1, 0, len(cuts) - 1)
        risk_at_t = cif_r[:, idx]
        try:
            auc_arr, _ = cumulative_dynamic_auc(tr_struct, te_struct, risk_at_t,
                                                 np.array([t]))
            aucs[float(t)] = float(auc_arr[0])
        except Exception:
            aucs[float(t)] = float("nan")
    return aucs


def process_one(ds, model, seed, summary, force=False):
    key = f"{ds}_{model}_seed{seed}"
    if (not force) and (key in summary) and "aucs" in summary[key]:
        print(f"  [skip] {key}: cached")
        return summary

    if not _ckpt(ds, model, seed).exists():
        print(f"  [skip] {key}: ckpt missing")
        return summary

    t0 = time.time()
    try:
        x_tr, dur_tr, ev_tr, x_te, dur_te, ev_te, cuts = _load_train_test(ds, seed)
        cif = INFER[model](ds, seed, x_te, cuts)
        target_risk = 0
        aucs = _auc_at_horizons(cif, cuts, dur_tr, ev_tr, dur_te, ev_te,
                                HORIZONS[ds], target_risk=target_risk)
        elapsed = time.time() - t0
        summary[key] = {
            "dataset": ds, "model": model, "seed": seed,
            "horizons": list(HORIZONS[ds]),
            "aucs": aucs, "target_risk": target_risk,
            "elapsed_seconds": elapsed,
        }
        print(f"  [done] {key}: {elapsed:.1f}s  "
              + "  ".join(f"t={t:.0f}:{a:.3f}" for t, a in aucs.items()))
    except Exception as e:
        import traceback; traceback.print_exc()
        summary[key] = {"dataset": ds, "model": model, "seed": seed,
                        "error": f"{type(e).__name__}: {e}"}
        print(f"  [error] {key}: {e}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--out", default=str(ROOT / "results/dynamic_auc.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    out_path = Path(args.out)
    summary = {}
    if out_path.exists():
        try:
            summary = json.load(open(out_path))
        except Exception:
            summary = {}

    print(f"\n{'='*72}\n  cumulative/dynamic AUC(t) compute")
    print(f"  datasets: {datasets}\n  models:   {models}\n  seeds:    {seeds}")
    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"  out:      {rel}\n{'='*72}")

    for ds in datasets:
        for model in models:
            for seed in seeds:
                summary = process_one(ds, model, seed, summary, force=args.force)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
