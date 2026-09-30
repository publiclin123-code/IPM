#!/usr/bin/env python3
"""Fig: EDGAR cross-corpus lead-time bar chart.

Horizontal bars = earliest forward-signal lead (days before Chapter 11) for the
12 bankruptcy firms, sorted by lead. Firms with no hit (recall boundary) are
shown at zero / as hollow markers. Companion to tab:edgar.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from figstyle import TOL, apply_style, save  # noqa: E402

# 手填 12 家公司的 lead (来自 aggregate_edgar.py 输出)
FIRMS = [
    ("Lordstown Motors", 174, True),
    ("WeWork", 166, True),
    ("Party City", 162, True),
    ("Tuesday Morning", 151, True),
    ("Bed Bath & Beyond", 108, True),
    ("Audacy", 53, True),
    ("SmileDirectClub", 86, True),
    ("Rite Aid", 11, True),
    ("Yellow Corp", 0, False),
    ("JOANN", 0, False),
    ("Big Lots", 0, False),
    ("Express", 0, False),
]
# 排序: hit 的按 lead 降序, 未命中排最后
FIRMS.sort(key=lambda x: (-x[1], -x[1] if x[2] else -1))
# 实际上: 命中按 lead 降序, 未命中排底部
hit = sorted([f for f in FIRMS if f[2]], key=lambda x: -x[1])
miss = [f for f in FIRMS if not f[2]]
FIRMS = hit + miss


def main():
    apply_style()
    n = len(FIRMS)
    labels = [f[0] for f in FIRMS]
    leads = [f[1] for f in FIRMS]
    is_hit = [f[2] for f in FIRMS]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(n)

    for i, (lead, h) in enumerate(zip(leads, is_hit)):
        if h:
            ax.barh(i, lead, color=TOL["blue"], height=0.6, zorder=3)
            ax.text(lead + 4, i, f"{lead} d", va="center", fontsize=7.5, color="#1F2937")
        else:
            ax.barh(i, 0, color="#E5E7EB", height=0.6, zorder=3)
            ax.text(6, i, "no hit", va="center", fontsize=7.5, color="#9AA5B1", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 195)
    ax.set_xlabel("earliest forward signal (days before Chapter 11)")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"])
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig_edgar_lead")
    plt.close(fig)
    print(f"edgar lead: {n} firms")
    return 0


if __name__ == "__main__":
    main()
