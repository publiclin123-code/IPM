#!/usr/bin/env python3
"""Fig: Foresight map — event a priori type vs ex-post FWGS.

One point per event on a shared days-from... no, on a shared FWGS axis, grouped
by the a priori type (slow-burn / creeping / sudden) that was assigned before
any LLM score was seen. Point size scales with the number of forward/latent
signals on the clock-split corpus; filled circles are protocol hits, open
circles are the designed zeroes (events with no true-precursor hit). The vertical
ordering within each type is by FWGS, so the "slow-burn = readable, sudden = hard"
pattern is immediate. Reads results/v2/_aggregate_18events.json, so it never
drifts from the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from figstyle import TOL, TYPE_COLOR, TYPE_LABEL, apply_style, save  # noqa: E402
from event_meta import EVENT_SHORT, EVENT_TYPES  # noqa: E402

AGG = ROOT / "results" / "v2" / "_aggregate_18events.json"
TYPE_ORDER = ["slow_burn", "creeping", "sudden"]
TYPE_POS = {t: i for i, t in enumerate(TYPE_ORDER)}
MAX_SIG = 35.0   # us_china_tariff_war_2018, for marker-size scaling


def load_events() -> dict[str, dict]:
    with open(AGG, encoding="utf-8") as f:
        data = json.load(f)
    return {e["event_id"]: e for e in data["per_event"]}


def main() -> int:
    apply_style()
    evs = load_events()

    # Assemble rows: (type, short, fwgs, n_sig, hit).
    rows = []
    for eid, typ in EVENT_TYPES.items():
        d = evs.get(eid)
        if d is None:
            continue
        short = EVENT_SHORT.get(eid, eid)
        fwgs = d.get("fwgs")
        rows.append({
            "type": typ, "short": short, "fwgs": fwgs if fwgs is not None else 0.0,
            "n": d.get("n_sig", 0), "hit": d.get("hit", False),
        })
    # Keep the 4 designed zeroes (suez, beirut, renesas quake, renesas fire) marked.
    # A "hit" of False with n>0 is the earthquake (false alarms); n==0 are zeroes.
    for r in rows:
        r["zero"] = (r["n"] == 0)

    # X positions: within each type slot, spread events slightly by FWGS order.
    rng = np.random.default_rng(0)
    # Group by type, sort each group by fwgs descending, assign x with small jitter.
    fig, ax = plt.subplots(figsize=(8.4, 4.5))

    for typ in TYPE_ORDER:
        grp = [r for r in rows if r["type"] == typ]
        grp.sort(key=lambda r: -r["fwgs"])
        # Per-dot x offsets are small, so points in a column stay close together
        # and the leader to each label is short. The label column is placed just
        # right of that cluster.
        xs = [TYPE_POS[typ] + (gi - (len(grp) - 1) / 2) * 0.12
              for gi in range(len(grp))]
        label_x = max(xs) + 0.18
        last_y = None
        for gi, r in enumerate(grp):
            x = xs[gi]
            size = 40 + 260 * (r["n"] / MAX_SIG)
            color = TYPE_COLOR[typ]
            mfc = "white" if not r["hit"] else color
            ax.scatter([x], [r["fwgs"]], s=size, facecolor=mfc,
                       edgecolor=color, linewidth=1.4, zorder=3)
            # Greedy stacking keeps labels apart; the common label_x plus a short
            # dashed leader ties each label unambiguously to its own dot.
            ly = r["fwgs"]
            if last_y is not None and (last_y - ly) < 0.055:
                ly = last_y - 0.055
            last_y = ly
            label = r["short"] + (" (zero)" if r["zero"] else "")
            label_color = "#999999" if r["zero"] else "#333333"
            ax.annotate(label, xy=(x, r["fwgs"]),
                        xytext=(label_x, ly),
                        ha="left", va="center", fontsize=8,
                        color=label_color, zorder=4,
                        arrowprops=dict(arrowstyle="-", lw=0.4,
                                        linestyle="--", color="#AAAAAA",
                                        alpha=0.55, shrinkA=0, shrinkB=0))

    # Type slot separators + labels (with event count under each type name).
    ax.set_xticks(list(TYPE_POS.values()))
    ax.set_xticklabels(
        [f"{TYPE_LABEL[t]}\n(n={sum(1 for r in rows if r['type'] == t)})"
         for t in TYPE_ORDER], fontsize=10)
    for xp, t in zip(TYPE_POS.values(), TYPE_ORDER):
        ax.axvline(xp - 0.42, color="#E5E5E5", lw=0.6, zorder=0)

    ax.set_ylim(-0.22, 1.04)
    ax.set_xlim(-0.62, 3.55)
    ax.set_ylabel("ex-post FWGS (clock-split)", fontsize=11)
    ax.yaxis.grid(True, ls=":", color=TOL["grey"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend: hit / no-hit + size meaning.
    handles = [
        Line2D([], [], marker="o", color="white", mec=TOL["blue"], mfc="white",
               mew=1.4, ms=6, ls="", label="protocol hit"),
        Line2D([], [], marker="o", color="white", mec=TOL["orange"], mfc=TOL["orange"],
               mew=1.4, ms=6, ls="", label="no true-precursor hit"),
        Line2D([], [], marker="o", color="white", mec="#666", mfc="white",
               mew=1.0, ms=4, ls="", label="marker size = # forward signals"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              loc="upper right", bbox_to_anchor=(0.995, 1.0))

    fig.tight_layout()
    save(fig, "fig_foresight_map")
    plt.close(fig)
    print(f"foresight map: {len(rows)} events")
    return 0


if __name__ == "__main__":
    main()
