"""Shared utilities: config loading, seeding, scipy patch, paths."""

from __future__ import annotations

import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


def patch_scipy_simps():
    """pycox 0.2.x calls scipy.integrate.simps which was removed in scipy>=1.14.
    Map it to simpson so pycox keeps working."""
    import scipy.integrate
    if not hasattr(scipy.integrate, "simps"):
        scipy.integrate.simps = scipy.integrate.simpson


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the submission/ root if not absolute."""
    p = Path(p)
    if p.is_absolute():
        return p
    return SUBMISSION_ROOT / p


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
