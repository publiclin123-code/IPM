"""Fig 7 (18-event redesign) — Events x models FWGS heatmap, reading finalize JSON.

Reads the three-model FWGS + bootstrap CI JSON produced by finalize_tables.py,
and renders the heatmap. Rows ordered by qwen FWGS desc; three model columns;
cell fill = sequential blue, annotated with point (bold) + 95% CI (grey).

Usage:
  python analysis/fig7_heatmap_18.py --json /tmp/fig7_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from figstyle import TOL, apply_style, save  # noqa: E402

MODELS = ["qwen3.6-27b", "gemma4-31b", "muse-glimmer-30b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    data = json.load(open(args.json))
    rows = data["rows"]
    apply_style()

    n = len(rows)
    pts = np.zeros((n, 3))
    cis = np.zeros((n, 3, 2))
    for i, r in enumerate(rows):
        for j, m in enumerate(MODELS):
            cell = r["models"].get(m) or {}
            fw = cell.get("fwgs")
            pts[i, j] = fw if fw is not None else np.nan
            ci = cell.get("ci")
            cis[i, j] = ci if ci else [np.nan, np.nan]

    fig, ax = plt.subplots(figsize=(5.4, max(4.6, n * 0.34)))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "fwgs_blue", ["#EAF2FB", "#A9CCE3", "#5499C7", TOL["blue"], "#1A4E7A"])

    im = ax.imshow(pts, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(3):
            pt = pts[i, j]
            if np.isnan(pt):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=10, color="#9AA5B1")
                continue
            lo, hi = cis[i, j]
            txt_color = "white" if pt > 0.55 else "#1F2937"
            ax.text(j, i - 0.14, f"{pt:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt_color)
            if not np.isnan(lo):
                ax.text(j, i + 0.24, f"[{lo:.2f},{hi:.2f}]", ha="center",
                        va="center", fontsize=6,
                        color="white" if pt > 0.55 else "#6B7280")

    ax.set_xticks(range(3))
    ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["short"] for r in rows], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FWGS", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig7_heatmap")
    plt.close(fig)
    print(f"heatmap: {n} events x {len(MODELS)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
