"""Shared publication style for Paper A figures (Risk Analysis / Wiley two-column)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Paul Tol qualitative (colorblind-safe)
TOL = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
    "black": "#000000",
}

TYPE_COLOR = {
    "slow_burn": TOL["blue"],
    "creeping": TOL["teal"],
    "sudden": TOL["orange"],
}
TYPE_LABEL = {
    "slow_burn": "Slow-burn / policy",
    "creeping": "Creeping",
    "sudden": "Sudden",
}

FIG_DIR = Path(__file__).resolve().parent.parent / "draft" / "figures"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
