"""Preprocessing pipelines: raw CSVs → (N, T, F) tensors + survival labels.

Each module exposes:
    run_pipeline(cfg: dict) -> dict
        Builds processed tensors from raw inputs and saves to cfg['data']['processed_dir'].
        Returns a summary dict with shapes and label statistics.
"""

from importlib import import_module


PIPELINES = {
    "eicu":  "src.preprocessing.eicu",
    "hirid": "src.preprocessing.hirid",
    "mcmed": "src.preprocessing.mcmed",
    "mimic": "src.preprocessing.mimic",
}


def get_pipeline(name: str):
    if name not in PIPELINES:
        raise ValueError(f"Unknown dataset {name!r}; options: {sorted(PIPELINES)}")
    return import_module(PIPELINES[name]).run_pipeline
