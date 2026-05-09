"""Evaluation metrics: time-dependent C-index, IBS, IBLL via pycox."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..losses import compute_cif


def _surv_df_per_risk(cif_risk: np.ndarray, time_index: np.ndarray) -> pd.DataFrame:
    """cif_risk: (N, K), time_index: (K,) → DataFrame for pycox EvalSurv."""
    surv = 1.0 - cif_risk
    return pd.DataFrame(surv.T, index=time_index)


@torch.no_grad()
def predict(model, loader, device, modality_slices, drop_modalities=None):
    model.eval()
    per_cause_hazard = getattr(model, "per_cause_hazard", False)
    cifs, durations, events = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, modality_slices=modality_slices, drop_modalities=drop_modalities)
        cif = compute_cif(out["logits"], per_cause_hazard=per_cause_hazard).cpu().numpy()  # (B, E, K)
        cifs.append(cif)
        durations.append(batch["durations_raw"].numpy() if "durations_raw" in batch
                          else batch["durations_idx"].numpy())
        events.append(batch["events"].numpy())
    return np.concatenate(cifs, 0), np.concatenate(durations, 0), np.concatenate(events, 0)


def evaluate_split(model, loader, cuts: np.ndarray, num_risks: int, device,
                   modality_slices, drop_modalities=None) -> dict:
    """Returns {risk_i: {ctd, ibs, ibll}} averaged across risks."""
    from pycox.evaluation import EvalSurv

    cif, durations, events = predict(model, loader, device, modality_slices, drop_modalities)
    time_index = cuts.astype(float)

    out: dict[str, dict[str, float]] = {}
    ctds, ibss, iblls = [], [], []

    for r in range(num_risks):
        e_label = r + 1
        cif_r = cif[:, r, :]
        mask = (events == 0) | (events == e_label)
        if mask.sum() < 10 or (events[mask] > 0).sum() < 5:
            out[f"risk{e_label}"] = {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}
            continue
        events_bin = (events[mask] == e_label).astype(int)
        durations_r = durations[mask]
        cif_r = cif_r[mask]

        surv_df = _surv_df_per_risk(cif_r, time_index)
        try:
            ev = EvalSurv(surv_df, durations_r, events_bin, censor_surv="km")
            ctd = ev.concordance_td()
            time_grid = np.linspace(durations_r.min(), durations_r.max(), 100)
            ibs = ev.integrated_brier_score(time_grid)
            ibll = ev.integrated_nbll(time_grid)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  EvalSurv error for risk {e_label}: {exc}")
            ctd, ibs, ibll = float("nan"), float("nan"), float("nan")

        out[f"risk{e_label}"] = {"ctd": float(ctd), "ibs": float(ibs), "ibll": float(ibll)}
        if not np.isnan(ctd):
            ctds.append(ctd)
            ibss.append(ibs)
            iblls.append(ibll)

    out["mean"] = {
        "ctd": float(np.mean(ctds)) if ctds else float("nan"),
        "ibs": float(np.mean(ibss)) if ibss else float("nan"),
        "ibll": float(np.mean(iblls)) if iblls else float("nan"),
    }
    return out


def eval_summary(metrics: dict) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(
            f"  {k:6s}: Ctd={v['ctd']:.4f}  IBS={v['ibs']:.4f}  IBLL={v['ibll']:.4f}"
        )
    return "\n".join(lines)
