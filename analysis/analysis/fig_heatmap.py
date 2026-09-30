"""Fig 7 (redesign) — Events × models FWGS heatmap.

8 rows (events, ordered by qwen FWGS desc) × 3 columns (qwen / gemma / muse).
Cell fill is a sequential blue (light → dark = low → high FWGS); each cell is
annotated with the point estimate (bold) and the bootstrap 95% CI (grey,
parenthesised) below. A right-hand colour bar encodes FWGS.

Replaces the forest plot (fig7_forest.py). Data identical to the forest source
(B=2000, SEED=42 from three_model_bootstrap.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from figstyle import TOL, apply_style, save  # noqa: E402

# event, (qwen, gemma, muse) point + (lo, hi) per model
DATA = [
    ("European energy",     0.907, 0.809, 0.847,
     (0.834, 0.963), (0.706, 0.897), (0.764, 0.918)),
    ("Red Sea crisis",      0.705, 0.799, 0.492,
     (0.484, 0.908), (0.667, 0.899), (0.259, 0.720)),
    ("U.S. chip controls",  0.678, 0.538, 0.650,
     (0.500, 0.833), (0.403, 0.670), (0.441, 0.819)),
    ("Toyota explosion",    0.627, 0.638, 0.596,
     (0.511, 0.744), (0.533, 0.735), (0.485, 0.709)),
    ("LA/LB port backlog",  0.323, 0.319, 0.300,
     (0.092, 0.573), (0.138, 0.511), (0.068, 0.565)),
    ("COVID-19 / Wuhan",    0.064, 0.057, 0.016,
     (0.033, 0.095), (0.019, 0.105), (0.000, 0.033)),
    ("Renesas earthquake",  0.000, 0.060, 0.000,
     (0.000, 0.000), (0.000, 0.179), (0.000, 0.000)),
]

MODELS = ["qwen3.6-27b", "gemma4-31b", "muse-glimmer-30b"]
POOLED = (0.610, 0.592, 0.596)
POOLED_CI = ((0.523, 0.690), (0.503, 0.678), (0.509, 0.678))


def main() -> int:
    apply_style()
    n = len(DATA)
    pts = np.array([[d[1], d[2], d[3]] for d in DATA])  # n×3

    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    # sequential blue cmap: light (low) → dark (high)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "fwgs_blue", ["#EAF2FB", "#A9CCE3", "#5499C7", TOL["blue"], "#1A4E7A"])

    im = ax.imshow(pts, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # annotations: point (bold) + CI (grey)
    for i, d in enumerate(DATA):
        cis = [d[4], d[5], d[6]]
        for j in range(3):
            pt = pts[i, j]
            lo, hi = cis[j]
            txt_color = "white" if pt > 0.55 else "#1F2937"
            ax.text(j, i - 0.12, f"{pt:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txt_color)
            ax.text(j, i + 0.22, f"[{lo:.2f},{hi:.2f}]", ha="center",
                    va="center", fontsize=6.5,
                    color="white" if pt > 0.55 else "#6B7280")

    # ticks
    ax.set_xticks(range(3))
    ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d[0] for d in DATA], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # grid between cells
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FWGS", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig7_heatmap")
    plt.close(fig)
    print(f"heatmap: {n} events × {len(MODELS)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
