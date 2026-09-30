#!/usr/bin/env python3
"""Fig: Cross-corpus validation — GDELT news (second-hand) vs SEC EDGAR (first-hand).

Two linked panels on the same headline metrics:

  Left   Protocol precision (news 0.873 with Wilson CI vs EDGAR 1.000).
  Right  Pooled ex-post FWGS (news 0.533 vs EDGAR 0.733), with the naive (v1)
         news value as a dashed reference line.

The point is the direction of the contrast: a first-hand, legally mandated source
yields cleaner warnings (perfect precision, higher FWGS) than second-hand news, so
the date protocol does not depend on the idiosyncrasies of the news stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from _fig_utils import (  # noqa: E402
    EDGAR_FIRMS, EDGAR_FWGS, EDGAR_FWD, EDGAR_PREC, EDGAR_TP,
    NEWS_FWGS, NEWS_FWD, NEWS_PREC, NEWS_TP, wilson_ci,
)
from figstyle import TOL, apply_style, save  # noqa: E402


def panel_precision(ax) -> None:
    """Protocol precision: news vs EDGAR, news with Wilson CI, EDGAR full-set."""
    labels = ["GDELT news\n(second-hand)", "SEC EDGAR\n(first-hand)"]
    vals = [NEWS_PREC, EDGAR_PREC]
    cols = [TOL["blue"], TOL["teal"]]
    y = [1, 0]
    for yi, v, c in zip(y, vals, cols):
        ax.barh(yi, v, height=0.46, color=c, alpha=0.88, zorder=3)
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=c, zorder=4)
    # Wilson CI for the news aggregate (sampled); EDGAR is a full-set count.
    lo, hi = wilson_ci(NEWS_TP, NEWS_FWD)
    ax.plot([lo, hi], [1, 1], color=TOL["blue"], lw=1.8, zorder=5)
    ax.plot([lo], [1], "|", color=TOL["blue"], ms=10, mew=1.6, zorder=6)
    ax.plot([hi], [1], "|", color=TOL["blue"], ms=10, mew=1.6, zorder=6)
    ax.set_xlim(0.70, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("protocol precision")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Protocol precision", loc="left", fontweight="bold")


def panel_fwgs(ax) -> None:
    """Pooled FWGS: news vs EDGAR, with the naive (v1) news value as reference."""
    labels = ["GDELT news", "SEC EDGAR"]
    vals = [NEWS_FWGS, EDGAR_FWGS]
    cols = [TOL["blue"], TOL["teal"]]
    y = [1, 0]
    for yi, v, c in zip(y, vals, cols):
        ax.plot([0, v], [yi, yi], "-", color=c, lw=5, alpha=0.55,
                solid_capstyle="butt", zorder=2)
        ax.plot([v], [yi], "o", ms=10, color=c, mec="white", mew=1.0, zorder=4)
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=c, zorder=5)
    # Naive (v1) news reference.
    ax.axvline(0.470, color="#BBBBBB", ls="--", lw=1.0, zorder=1)
    ax.text(0.470, 0.20, "naive (v1) 0.470", ha="center", va="center",
            fontsize=6.5, color="#666", rotation=90)
    ax.set_xlim(0.40, 0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pooled ex-post FWGS")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Ex-post FWGS", loc="left", fontweight="bold")


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    panel_precision(axes[0])
    panel_fwgs(axes[1])
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_crosscorpus")
    plt.close(fig)
    print("fig_crosscorpus written")
    return 0


if __name__ == "__main__":
    main()
