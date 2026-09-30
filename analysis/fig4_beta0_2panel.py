#!/usr/bin/env python3
"""Fig 4 (two-panel redesign) — BSCC diagnostic, 18-event post-event pool.

A: window robustness — beta_0 vs post_days {7,14,30} for both ontologies.
B: label-mix shift — confirmation / latent / forward composition, v1 -> v2.

The paired-donut panel was dropped: the beta_0 collapse (0.500 -> 0.233, -53%)
is a single number already in the text and tab:mechanism.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from figstyle import TOL, apply_style, save  # noqa: E402


def panel_A(ax):
    windows = [7, 14, 30]
    v1 = [0.514, 0.500, 0.494]
    v2 = [0.211, 0.233, 0.236]
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
    # panel label below the axes (a must clear the xlabel)
    ax.text(0.5, -0.34, "(a)", transform=ax.transAxes,
            ha="center", va="top")


def _spread(labels, min_gap=0.07):
    """Enforce a minimum vertical gap between endpoint percentage labels."""
    labels = sorted(labels, key=lambda d: d["y"])
    for i in range(1, len(labels)):
        if labels[i]["y"] - labels[i - 1]["y"] < min_gap:
            labels[i]["y"] = labels[i - 1]["y"] + min_gap
    return labels


def panel_B(ax):
    cats = [("Confirmation", TOL["teal"]),
            ("Latent", TOL["orange"]),
            ("Forward-looking", TOL["blue"])]
    v1 = np.array([42, 22, 20], dtype=float)
    v2 = np.array([66, 0, 20], dtype=float)
    v1 /= v1.sum()
    v2 /= v2.sum()
    x_left, x_right = 0.0, 1.0
    left_lbls, right_lbls = [], []
    for (name, color), s1, s2 in zip(cats, v1, v2):
        ax.plot([x_left, x_right], [s1, s2], "-", color=color, lw=2.0, zorder=2)
        ax.plot([x_left], [s1], "o", ms=7, color=color, mec="white", mew=0.8, zorder=3)
        ax.plot([x_right], [s2], "o", ms=7, color=color, mec="white", mew=0.8, zorder=3)
        left_lbls.append({"y": float(s1), "txt": f"{s1:.0%}", "color": color})
        right_lbls.append({"y": float(s2), "txt": f"{s2:.0%}", "color": color})
    for lbls, x, ha in ((left_lbls, x_left - 0.03, "right"),
                        (right_lbls, x_right + 0.03, "left")):
        for lbl in _spread(lbls):
            ax.text(x, lbl["y"], lbl["txt"], ha=ha, va="center",
                    fontsize=7.5, color=lbl["color"])
    handles = [Line2D([], [], color=c, lw=2.0, marker="o", ms=5,
                      mec="white", mew=0.8) for _, c in cats]
    ax.legend(handles, [n for n, _ in cats], frameon=False, fontsize=7,
              loc="upper left", bbox_to_anchor=(0.02, 0.98))
    # column headers moved below the axes, above the (b) panel label
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x_left, -0.13, "Naive (v1)", transform=tr, ha="center",
            va="top", fontsize=8)
    ax.text(x_right, -0.13, "Clock-split (v2)", transform=tr, ha="center",
            va="top", fontsize=8)
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
    ax.text(0.5, -0.33, "(b)", transform=ax.transAxes,
            ha="center", va="top")


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    panel_A(axes[0])
    panel_B(axes[1])
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_beta0")
    plt.close(fig)
    print("fig4_beta0 (2-panel) written")


if __name__ == "__main__":
    main()
