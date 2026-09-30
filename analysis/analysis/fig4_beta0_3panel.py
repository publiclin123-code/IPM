#!/usr/bin/env python3
"""Fig 4 (three-panel redesign) — BSCC diagnostic, 18-event post-event pool.

A: paired donuts — beta_0 naive (0.476) vs clock-split (0.214), -55%.
B: window robustness — beta_0 vs post_days {7,14,30} for both ontologies.
C: label-mix shift — confirmation / latent / forward composition, v1 -> v2.

Reads the recomputed 18-event post-event pool numbers (hardcoded from
pipeline/fetch_background.py + extract + compute_bscc). Uses figstyle.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from figstyle import TOL, apply_style, save  # noqa: E402


def panel_A(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.05, 1.15)
    ax.axis("off")
    from matplotlib.patches import Wedge
    beta = [("Naive\n(v1)", 0.476, TOL["red"]),
            ("Clock-split\n(v2)", 0.214, TOL["teal"])]
    centers = [-0.62, 0.62]
    for (lab, b, color), cx in zip(beta, centers):
        clean = 1.0 - b
        ax.add_patch(Wedge((cx, 0.0), 0.42, 90, 90 + 360 * b,
                           width=0.16, facecolor=color, edgecolor="white", lw=0.8))
        ax.add_patch(Wedge((cx, 0.0), 0.42, 90 - 360 * clean, 90,
                           width=0.16, facecolor="#E5E7EB", edgecolor="white", lw=0.8))
        ax.text(cx, 0.0, f"{b:.1%}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(cx, -0.62, lab, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(0.20, 0.0), xytext=(-0.20, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=TOL["grey"], lw=1.4))
    ax.text(0.0, 0.20, "−55%", ha="center", va="center", fontsize=11,
            fontweight="bold", color=TOL["red"])
    ax.text(-0.62, 0.55, "contaminated", ha="center", va="bottom",
            fontsize=6.5, color=TOL["red"])
    ax.set_title("A", loc="left", fontweight="bold", pad=2)


def panel_B(ax):
    windows = [7, 14, 30]
    v1 = [0.500, 0.476, 0.483]
    v2 = [0.194, 0.214, 0.231]
    ax.plot(windows, v1, "-o", color=TOL["red"], lw=1.8, ms=5,
            mec="white", mew=0.6, label="naive (v1)")
    ax.plot(windows, v2, "-s", color=TOL["teal"], lw=1.8, ms=5,
            mec="white", mew=0.6, label="clock-split (v2)")
    for x, y in zip(windows, v1):
        ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom",
                fontsize=7, color=TOL["red"])
    for x, y in zip(windows, v2):
        ax.text(x, y - 0.05, f"{y:.2f}", ha="center", va="top",
                fontsize=7, color=TOL["teal"])
    ax.set_xlim(4, 33)
    ax.set_ylim(0, 0.65)
    ax.set_xticks(windows)
    ax.set_xlabel("post-event window (days)")
    ax.set_ylabel(r"$\beta_0$")
    ax.yaxis.grid(True, ls=":", color=TOL["grey"])
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("B", loc="left", fontweight="bold")


def panel_C(ax):
    cats = [("Confirmation", TOL["teal"]),
            ("Latent", TOL["orange"]),
            ("Forward-looking", TOL["blue"])]
    v1 = np.array([43, 21, 18], dtype=float)
    v2 = np.array([66, 0, 18], dtype=float)
    v1 /= v1.sum()
    v2 /= v2.sum()
    x_left, x_right = 0.0, 1.0
    for (name, color), s1, s2 in zip(cats, v1, v2):
        ax.plot([x_left, x_right], [s1, s2], "-", color=color, lw=2.0, zorder=2)
        ax.plot([x_left], [s1], "o", ms=7, color=color, mec="white", mew=0.8, zorder=3)
        ax.plot([x_right], [s2], "o", ms=7, color=color, mec="white", mew=0.8, zorder=3)
        ax.text(x_left - 0.03, s1, f"{s1:.0%}", ha="right", va="center",
                fontsize=7.5, color=color, fontweight="bold")
        ax.text(x_right + 0.03, s2, f"{s2:.0%}", ha="left", va="center",
                fontsize=7.5, color=color, fontweight="bold")
        yname = max(s1, s2)
        xname = x_left - 0.03 if s1 >= s2 else x_right + 0.03
        halign = "right" if s1 >= s2 else "left"
        ax.text(xname, yname + 0.035, name, ha=halign, va="bottom",
                fontsize=7, color=color)
    ax.text(x_left, 1.06, "Naive (v1)", ha="center", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(x_right, 1.06, "Clock-split (v2)", ha="center", va="bottom",
            fontsize=8, fontweight="bold")
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("Share of post-event signals")
    ax.set_title("C", loc="left", fontweight="bold")


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 0.95, 1.15]})
    panel_A(axes[0])
    panel_B(axes[1])
    panel_C(axes[2])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_beta0")
    plt.close(fig)
    print("fig4_beta0 (3-panel) written")


if __name__ == "__main__":
    main()
