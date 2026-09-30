"""Fig 7 — Three-model FWGS forest plot with bootstrap 95% CIs.

A forest / dumbbell plot: one row per event, three markers per model with
horizontal 95% CI whiskers, ordered by qwen FWGS (high to low). The pooled
interval is shown as a shaded band at the bottom.

Data is hardcoded from three_model_bootstrap.py (B=2000, SEED=42).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import TOL, apply_style, save

# event, short label, type color key, (qwen, gemma, muse) point + (lo, hi)
DATA = [
    ("European energy",  "creeping", 0.907, 0.809, 0.847,
     (0.834, 0.963), (0.706, 0.897), (0.764, 0.918)),
    ("Red Sea crisis",   "slow_burn", 0.705, 0.799, 0.492,
     (0.484, 0.908), (0.667, 0.899), (0.259, 0.720)),
    ("U.S. chip controls","slow_burn", 0.678, 0.538, 0.650,
     (0.500, 0.833), (0.403, 0.670), (0.441, 0.819)),
    ("Toyota explosion", "sudden", 0.627, 0.638, 0.596,
     (0.511, 0.744), (0.533, 0.735), (0.485, 0.709)),
    ("LA/LB port backlog","creeping", 0.323, 0.319, 0.300,
     (0.092, 0.573), (0.138, 0.511), (0.068, 0.565)),
    ("COVID-19 / Wuhan", "sudden", 0.064, 0.057, 0.016,
     (0.033, 0.095), (0.019, 0.105), (0.000, 0.033)),
    ("Renesas earthquake","sudden", 0.000, 0.060, 0.000,
     (0.000, 0.000), (0.000, 0.179), (0.000, 0.000)),
]

MODELS = [
    ("qwen3.6-27b", TOL["blue"], "o"),
    ("gemma4-31b", TOL["orange"], "s"),
    ("muse-glimmer-30b", TOL["teal"], "D"),
]

TYPE_COLOR = {"slow_burn": TOL["blue"], "creeping": TOL["teal"], "sudden": TOL["orange"]}


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.1, 3.6))

    # order already high -> low; y index from top
    n = len(DATA)
    ys = np.arange(n)[::-1]  # top row = index n-1
    jitter = np.array([-0.13, 0.0, 0.13])

    for row_idx, (label, typ, q, g, m, ciq, cig, cim) in enumerate(DATA):
        y = ys[row_idx]
        pts = [q, g, m]
        cis = [ciq, cig, cim]
        for (mname, color, marker), pt, ci, dx in zip(MODELS, pts, cis, jitter):
            ax.plot([ci[0], ci[1]], [y + dx, y + dx], color=color, lw=1.4,
                    solid_capstyle="round", alpha=0.9, zorder=2)
            ax.plot([pt], [y + dx], marker=marker, ms=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        # type label as subtle left annotation
        ax.text(-0.07, y, label, ha="right", va="center", fontsize=8)
        ax.plot([-0.06, -0.06], [y, y], color=TYPE_COLOR[typ], lw=2.5,
                solid_capstyle="round", alpha=0.7, zorder=1)

    # reference lines
    ax.axvline(0.0, color=TOL["grey"], ls=":", lw=0.8, zorder=0)
    ax.axvline(0.5, color=TOL["grey"], ls="--", lw=0.7, alpha=0.5, zorder=0)

    # pooled band at bottom
    ax.axhspan(-0.9, -0.45, color=TOL["grey"], alpha=0.12, zorder=0)
    ax.text(0.5, -0.68, "pooled (stratified)   qwen 0.610   gemma 0.592   muse 0.596",
            ha="center", va="center", fontsize=7.5, color="black")

    ax.set_xlim(-0.35, 1.02)
    ax.set_ylim(-1.0, n - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("FWGS  ($\\tau{=}30$, $\\lambda{=}0.5$)  with 95% bootstrap CI")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"], alpha=0.6)
    ax.set_axisbelow(True)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, color=c, ls="", ms=6, label=mn)
               for (mn, c, m) in MODELS]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.5,
              bbox_to_anchor=(1.0, 0.0))

    fig.tight_layout()
    save(fig, "fig7_forest")
    plt.close(fig)


if __name__ == "__main__":
    main()
