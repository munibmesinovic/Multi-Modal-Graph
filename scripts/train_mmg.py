#!/usr/bin/env python3
"""Train MM-GraphSurv on one (dataset, seed) pair.

Configs under configs/ already encode the training recipe — scripts here
only set the per-run seed + processed_dir + checkpoint location.

For MC-MED the raw 4-way ED disposition is collapsed in-memory to the
2-way cohort (ICU admission vs. inpatient care) per the `event_collapse`
entry in configs/mcmed.yaml.

Checkpoint layout: checkpoints/{dataset}_mmgraphsurv_seed{seed}/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import patch_scipy_simps, load_config, seed_everything
patch_scipy_simps()

from src.train import train_one_dataset

DEFAULT_SEEDS = [42, 123, 7, 2024, 99]
DATASETS = ["eicu", "mimic", "mcmed", "hirid"]


def _maybe_collapse_events(cfg, processed_dir: Path):
    """Apply `dataset.event_collapse` remap (if present) directly to the
    preprocessed events_*.npy arrays. Safe re-application: idempotent given
    the mapping is a projection."""
    remap = cfg.get("dataset", {}).get("event_collapse") or {}
    if not remap:
        return
    remap = {int(k): int(v) for k, v in remap.items()}
    tname = {"eicu": "eICU", "mimic": "MIMIC",
             "mcmed": "MCMED", "hirid": "HIRID"}[cfg["dataset"]["name"]]
    for split in ("train", "val", "test"):
        p = processed_dir / f"events_{split}_{tname}.npy"
        if not p.exists():
            continue
        ev = np.load(p)
        needs_write = False
        for src, dst in remap.items():
            if (ev == src).any():
                needs_write = True
                break
        if needs_write:
            for src, dst in remap.items():
                ev[ev == src] = dst
            np.save(p, ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ds = args.dataset

    for s in seeds:
        run_dir = ROOT / f"checkpoints/{ds}_mmgraphsurv_seed{s}"
        metrics_p = run_dir / "metrics.json"
        ckpt_p = run_dir / "best_model.pth"
        if metrics_p.exists() and ckpt_p.exists() and not args.force:
            print(f"[skip] {ds} seed{s}: ckpt exists at {run_dir.name}")
            continue

        cfg = load_config(ROOT / f"configs/{ds}.yaml")
        cfg["data"]["seed"] = s
        cfg["data"]["processed_dir"] = f"data/{ds}/processed_seed{s}"
        _maybe_collapse_events(cfg, ROOT / cfg["data"]["processed_dir"])

        seed_everything(s)
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[run] {ds} mmgraphsurv seed{s} -> {run_dir.name}")
        metrics = train_one_dataset(cfg, run_dir=str(run_dir))
        print(f"[done] {ds} seed{s} in {(time.time()-t0)/60:.1f} min")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
