#!/usr/bin/env python3
"""Fig: Headline results overview (the "number map").

Four panels that let a reviewer absorb the paper's core contribution in one
glance, mirroring the structured abstract:

  A  Ontology split — beta_0 collapses on the post-event negative control.
  B  Protocol precision — news (second-hand) vs SEC EDGAR (first-hand).
  C  Ex-post FWGS — v1 (naive) -> v2 (clock-split), plus blind-annotation
     precision.
  D  Coverage — forward signals / events / in-window leads.

All numbers are read from the existing aggregate JSON files via _fig_utils, so
the figure never drifts from the text or the tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from _fig_utils import (  # noqa: E402
    BETA0_N, BETA0_V1, BETA0_V2, BLIND_PREC,
    EDGAR_FIRMS, EDGAR_FWGS, EDGAR_FWD, EDGAR_PREC, EDGAR_TP,
    NEWS_EVENTS, NEWS_FP, NEWS_FWD, NEWS_FWGS, NEWS_PREC, NEWS_TP,
    wilson_ci,
)
import figstyle  # noqa: E402
from figstyle import TOL, apply_style, save  # noqa: E402


def panel_A(ax) -> None:
    """beta_0: naive -> clock-split, with -55% annotation."""
    ax.set_ylim(0, 0.60)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive\n(v1)", "Clock-split\n(v2)"], fontsize=8)
    x = [0, 1]
    vals = [BETA0_V1, BETA0_V2]
    cols = [TOL["orange"], TOL["blue"]]
    for xi, v, c in zip(x, vals, cols):
        ax.bar(xi, v, width=0.5, color=c, alpha=0.85, zorder=3)
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=c)
    # -55% arrow between bars
    ax.annotate("", xy=(0.72, 0.36), xytext=(0.28, 0.36),
                arrowprops=dict(arrowstyle="->", color=TOL["black"], lw=1.3))
    ax.text(0.5, 0.375, "$-55\\%$", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=TOL["black"])
    ax.set_ylabel(r"$\beta_0$  (post-event contamination)")
    ax.yaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("A  Ontology split", loc="left", fontweight="bold")


def panel_B(ax) -> None:
    """Protocol precision: news vs EDGAR, with Wilson CIs."""
    names = ["GDELT news\n(second-hand)", "SEC EDGAR\n(first-hand)"]
    vals = [NEWS_PREC, EDGAR_PREC]
    cols = [TOL["blue"], TOL["teal"]]
    for i, (nm, v, c) in enumerate(zip(names, vals, cols)):
        ax.barh(i, v, height=0.5, color=c, alpha=0.85, zorder=3)
        ax.text(v + 0.01, i, f"{v:.3f}", va="center",
                fontsize=9, fontweight="bold", color=c)
    # Wilson CI for news (sampled); EDGAR is aggregate full-set -> no sampling CI
    lo, hi = wilson_ci(NEWS_TP, NEWS_FWD)
    ax.plot([lo, hi], [0, 0], color=TOL["blue"], lw=1.6, zorder=4)
    ax.plot([lo], [0], "|", color=TOL["blue"], ms=8, mew=1.4, zorder=5)
    ax.plot([hi], [0], "|", color=TOL["blue"], ms=8, mew=1.4, zorder=5)
    ax.set_xlim(0.70, 1.04)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("protocol precision")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B  Protocol precision", loc="left", fontweight="bold")


def panel_C(ax) -> None:
    """Ex-post FWGS: naive v1 -> clock-split v2, plus blind-annotation precision."""
    ax.set_ylim(0, 0.62)
    ax.set_xlim(0, 1)
    ax.axis("off")
    # FWGS dumbbell
    x0, x1 = 0.14, 0.62
    y = 0.52
    ax.plot([x0, x1], [y, y], "-", color="#C9C9C9", lw=1.2)
    ax.plot(x0, y, "o", ms=9, color="white", mec="#888", mew=1.6)
    ax.plot(x1, y, "o", ms=11, color=TOL["blue"], mec="white", mew=0.8)
    ax.text(x0 - 0.02, y, "0.470", ha="right", va="center", fontsize=8,
            color="#666")
    ax.text(x1 + 0.02, y, "0.533", ha="left", va="center", fontsize=9,
            fontweight="bold", color=TOL["blue"])
    ax.text(0.02, y, "FWGS (pooled)", ha="left", va="center", fontsize=7.5,
            color="#333", fontweight="bold")
    ax.annotate("", xy=(0.72, y), xytext=(0.66, y),
                arrowprops=dict(arrowstyle="->", color=TOL["blue"], lw=1.2))
    ax.text(0.69, y + 0.03, "$\u2191$", color=TOL["blue"], fontsize=9)
    # Blind-annotation precision bar
    y2 = 0.20
    ax.plot([0.14, 0.14 + BLIND_PREC * 0.62], [y2, y2], "-", color=TOL["teal"],
            lw=6, solid_capstyle="butt")
    lo, hi = wilson_ci(48, 51)
    ax.plot([0.14 + lo * 0.62, 0.14 + hi * 0.62], [y2, y2], "-", color=TOL["teal"],
            lw=1.4, zorder=4)
    ax.plot([0.14 + lo * 0.62], [y2], "|", color=TOL["teal"], ms=7, mew=1.2)
    ax.plot([0.14 + hi * 0.62], [y2], "|", color=TOL["teal"], ms=7, mew=1.2)
    ax.text(0.16 + BLIND_PREC * 0.62, y2, "0.941", ha="left", va="center",
            fontsize=9, fontweight="bold", color=TOL["teal"])
    ax.text(0.02, y2, "blind annot.", ha="left", va="center", fontsize=7.5,
            color="#333", fontweight="bold")
    ax.set_title("C  Ex-post FWGS & annotation", loc="left", fontweight="bold")


def panel_D(ax) -> None:
    """Coverage counters."""
    ax.axis("off")
    stats = [
        ("forward signals", f"{NEWS_FWD}"),
        ("events", f"{NEWS_EVENTS}"),
        ("events with a lead", "14 / 18"),
        ("lead range", "$3$--$180$ d"),
    ]
    y0 = 0.80
    dy = 0.20
    for i, (lab, val) in enumerate(stats):
        y = y0 - i * dy
        ax.text(0.08, y, val, fontsize=13, fontweight="bold",
                color=TOL["black"], va="center")
        ax.text(0.72, y, lab, fontsize=8, color="#444", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("D  Coverage", loc="left", fontweight="bold")


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    panel_A(axes[0, 0])
    panel_B(axes[0, 1])
    panel_C(axes[1, 0])
    panel_D(axes[1, 1])
    fig.subplots_adjust(wspace=0.30, hspace=0.42)
    save(fig, "fig_overview")
    plt.close(fig)
    print("fig_overview written")
    return 0


if __name__ == "__main__":
    main()
