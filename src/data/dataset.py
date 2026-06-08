"""Dataset / DataLoader wrappers for MM-GraphSurv preprocessed tensors."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ..utils import resolve_path


def _event_bin_balanced_weights(events: np.ndarray,
                                durations_idx: np.ndarray,
                                num_bins: int) -> np.ndarray:
    """Per-patient weight so `WeightedRandomSampler` draws balanced across
    (event_status × event-time bin) cells.

    Event patients are upweighted so every event-time bin contributes equally
    in expectation per batch. Censored patients all get the same weight
    summing to 1, so each batch carries a diverse set of event times while
    still containing enough censored patients for survival loss computation.
    """
    events = events.astype(int)
    dur = durations_idx.astype(int)
    weights = np.zeros(len(events), dtype=np.float64)

    event_mask = events == 1
    censor_mask = ~event_mask

    # Per-(event, bin) uniform weighting
    for b in range(num_bins):
        in_bin = event_mask & (dur == b)
        n = in_bin.sum()
        if n > 0:
            # Each event bin should contribute 1/num_bins of event probability;
            # within a bin, each patient is uniform.
            weights[in_bin] = 1.0 / (num_bins * n)

    # Censored: uniform, total mass 1
    n_cen = censor_mask.sum()
    if n_cen > 0:
        weights[censor_mask] = 1.0 / n_cen

    # Normalize so events and censored each get 50% probability mass.
    total_event_weight = weights[event_mask].sum()
    total_censor_weight = weights[censor_mask].sum()
    if total_event_weight > 0:
        weights[event_mask] *= 0.5 / total_event_weight
    if total_censor_weight > 0:
        weights[censor_mask] *= 0.5 / total_censor_weight
    return weights


class MMGraphSurvDataset(Dataset):
    def __init__(self, x: np.ndarray, durations_idx: np.ndarray, events: np.ndarray,
                 durations_raw: np.ndarray | None = None,
                 modality_mask: np.ndarray | None = None,
                 z_clip: float = 10.0):
        # Clip extreme z-scores from heavy-tailed clinical features.
        # StandardScaler doesn't clip, so rare outliers (e.g. BUN z=394)
        # persist. Clipping at |z|<=10 affects <0.01% of values.
        x = np.clip(x, -z_clip, z_clip).astype(np.float32)
        self.x = torch.from_numpy(x)
        self.durations_idx = torch.from_numpy(durations_idx.astype(np.int64))
        self.events = torch.from_numpy(events.astype(np.int64))
        self.durations_raw = (
            torch.from_numpy(durations_raw.astype(np.float32))
            if durations_raw is not None else None
        )
        # modality_mask: (N, num_modalities) bool/uint8 — 1 if modality
        # present for that patient. Missing file ⇒ assume all present.
        if modality_mask is not None:
            self.modality_mask = torch.from_numpy(modality_mask.astype(np.float32))
        else:
            self.modality_mask = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {
            "x": self.x[idx],
            "durations_idx": self.durations_idx[idx],
            "events": self.events[idx],
        }
        if self.durations_raw is not None:
            item["durations_raw"] = self.durations_raw[idx]
        if self.modality_mask is not None:
            item["modality_mask"] = self.modality_mask[idx]
        return item


def _load_split(processed_dir: Path, name: str, split: str,
                num_modalities: int | None = None) -> MMGraphSurvDataset:
    x = np.load(processed_dir / f"x_{split}_{name}.npy")
    with open(processed_dir / f"y_{split}_surv_{name}.p", "rb") as f:
        durations_raw_or_idx, events = pickle.load(f)
    # y file may store either raw durations (float) or bin indices (int).
    # Downstream always wants bin indices; we compute them from cuts here.
    cuts_path = processed_dir / f"cuts_{name}.npy"
    if durations_raw_or_idx.dtype.kind == "f" and cuts_path.exists():
        cuts = np.load(cuts_path)
        durations_raw = durations_raw_or_idx.astype(np.float32)
        durations_idx = np.searchsorted(cuts, durations_raw, side="right") - 1
        durations_idx = np.clip(durations_idx, 0, len(cuts) - 1).astype(np.int64)
    else:
        durations_idx = durations_raw_or_idx.astype(np.int64)
        drp = processed_dir / f"durations_{split}_{name}.npy"
        durations_raw = np.load(drp).astype(np.float32) if drp.exists() else None

    mmp = processed_dir / f"modality_mask_{split}_{name}.npy"
    modality_mask = np.load(mmp) if mmp.exists() else None
    # If the stored mask has a different modality count than expected (e.g.
    # HiRID 4-mod config re-carving an existing 2-mod tensor), drop it —
    # MMG's forward will behave as "all modalities present for all patients"
    # which is correct when we're just re-slicing existing data.
    if modality_mask is not None and num_modalities is not None \
            and modality_mask.ndim == 2 and modality_mask.shape[1] != num_modalities:
        modality_mask = None

    return MMGraphSurvDataset(x, durations_idx, events, durations_raw, modality_mask)


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    name_map = {"eicu": "eICU", "mimic": "MIMIC", "mcmed": "MCMED",
                "pbc2": "PBC2", "support": "SUPPORT",
                "gbsg": "GBSG", "metabric": "METABRIC",
                "hirid": "HIRID", "hirid_circ": "HIRID"}
    name = name_map[cfg["dataset"]["name"]]
    proc_dir = resolve_path(cfg["data"]["processed_dir"])
    num_mods = cfg["dataset"].get("num_modalities", len(cfg["dataset"]["modalities"]))

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 0)
    use_event_balanced = bool(cfg["training"].get("event_bin_balanced_sampler", False))
    num_bins = int(cfg["data"].get("num_durations", 10))

    # Tier-A modality ablation hook: zero the input channels belonging to
    # the specified modalities before they ever reach the model. Architecture
    # (encoders, fusion graph, parameter count) is untouched — the model sees
    # an all-zero block for the ablated modality and learns to ignore it.
    zero_mod = list(cfg.get("data", {}).get("zero_modality_inputs") or [])

    # Honor `dataset.event_collapse` (e.g. MC-MED 4-way → 2-way) by
    # remapping the in-memory events tensor on each split. The on-disk
    # y_*_surv_*.p files are NOT mutated; this lets the same processed
    # data be replayed at any num_risks setting via a config-only knob.
    event_remap = {int(k): int(v)
                   for k, v in (cfg.get("dataset", {}).get("event_collapse") or {}).items()}

    out = {}
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        ds = _load_split(proc_dir, name, split, num_modalities=num_mods)
        if event_remap:
            for src, dst in event_remap.items():
                ds.events[ds.events == src] = dst
        if zero_mod:
            slices = modality_slices_for(cfg)
            for m in zero_mod:
                sel = slices.get(m)
                if sel is None:
                    raise KeyError(f"zero_modality_inputs: unknown modality {m!r}")
                if isinstance(sel, slice):
                    ds.x[:, :, sel] = 0.0
                else:  # index tensor
                    ds.x[:, :, sel] = 0.0
        # Train-only: optional event-bin-balanced sampler (opt-in via config).
        # Disabled by default → other datasets are unaffected.
        if split == "train" and use_event_balanced:
            w = _event_bin_balanced_weights(
                ds.events.numpy(),
                ds.durations_idx.numpy(),
                num_bins=num_bins,
            )
            sampler = WeightedRandomSampler(
                torch.as_tensor(w, dtype=torch.double),
                num_samples=len(ds),
                replacement=True,
            )
            out[split] = DataLoader(ds, batch_size=bs, sampler=sampler,
                                    num_workers=nw, pin_memory=True, drop_last=False)
        else:
            out[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                    num_workers=nw, pin_memory=True, drop_last=False)
    return out


def modality_slices_for(cfg: dict):
    """Compute per-modality column selectors in the (B, S, F) tensor.

    Two modes:
    1. **Contiguous (default)** — if no `dataset.modality_features` is set,
       assume a contiguous layout in the order of `dataset.modalities`.
       Returns `{modality: slice(lo, hi)}`. This matches the original MIMIC /
       eICU / MC-MED layout: dynamic | static | rad | icd.

    2. **Index-based** — if `dataset.modality_features` is set (dict of
       modality → list of column indices), return
       `{modality: torch.as_tensor(idx_list)}`. This is used by 4-mod HiRID
       to carve clinical blocks out of the existing (N, S, 35) tensor
       without re-preprocessing.

    In both modes PyTorch's `x[:, :, sel]` advanced indexing picks the right
    columns per modality, so the rest of the model is unchanged.
    """
    keys = cfg["dataset"]["modalities"]
    feature_indices = cfg["dataset"].get("modality_features")

    if feature_indices:
        import torch as _torch
        out = {}
        for k in keys:
            if k not in feature_indices:
                raise ValueError(f"modality_features missing entry for {k!r}")
            out[k] = _torch.as_tensor(feature_indices[k], dtype=_torch.long)
        return out

    dims = cfg["model"]["modality_dims"]
    raw_dims = cfg["model"].get("raw_modality_dims", {})

    slices = {}
    cursor = 0
    for k in keys:
        d = raw_dims.get(k, dims.get(k))
        if d is None:
            raise ValueError(f"No dimension for modality {k!r} in modality_dims or raw_modality_dims")
        slices[k] = slice(cursor, cursor + d)
        cursor += d
    return slices
