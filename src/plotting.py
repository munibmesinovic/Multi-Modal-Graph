"""Shared matplotlib style for all paper figures.

Nature-style: sans-serif (Helvetica → Nimbus Sans → Arial → Liberation Sans →
DejaVu Sans), crisp axes, math rendered with matching font so labels like
``$C^{td}$`` don't switch typeface mid-line.

Call once at the top of any figure script:

    from src.plotting import set_style
    set_style()
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


SANS_FALLBACK = [
    "Helvetica",
    "Nimbus Sans",
    "TeX Gyre Heros",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
]


def set_style(base_size: float = 10.0) -> None:
    """Apply Nature-journal sans-serif style to matplotlib globally."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS_FALLBACK,
        "mathtext.fontset": "dejavusans",
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


MODALITY_COLORS = {
    "dynamic": "#2A9D8F",   # teal  (was #0072B2 — blue)
    "static":  "#F4A261",   # warm orange (was #E69F00 — yellow-orange)
    "icd":     "#9467BD",   # purple (was #009E73 — green; green clashed with icd cluster fills)
    "rad":     "#D55E00",   # vermillion (was #CC79A7 — pink)
}


MODEL_COLORS = {
    "cox":             "#4F4F4F",
    "deepsurv":        "#A6A6A6",
    "deephit":         "#4F8EE1",
    "dynamic_deephit": "#2A9D8F",
    "dyndh":           "#2A9D8F",
    "dysurv":          "#F4A261",
    "lightgbm":        "#9467BD",
    "mmg":             "#D62728",
    "mmgraphsurv":     "#D62728",
    "survtrace":       "#8C564B",
}


DATASET_TITLE = {
    "eicu":  "eICU",
    "mimic": "MIMIC-IV",
    "mcmed": "MC-MED",
}
