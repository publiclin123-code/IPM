#!/usr/bin/env python3
"""Generate the §4.3 retrieval-audit precision table and figure from the
final body-round labels (relevant_label_body) in the canonical worksheet.

Emits:
  - LaTeX table body for tab:precision (per-event + pooled, Wilson 95% CI)
  - draft/figures/fig_retrieval_precision.pdf (per-event precision + CI bar)
"""
from __future__ import annotations
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotation"
WS = ANN / "retrieval_audit_sample_body.csv"
GT = json.load(open(ROOT / "validation" / "gt_events.json"))["events"]

# Sampled pools are kept only for events that survive in the ground truth. The
# file is the measuring instrument, so an entry retracted from it cannot supply
# audit evidence about its own pool; `toyota_steel_explosion_2019` was retracted
# (pipeline/prune_gt_toyota.py, validation/gt_excluded.json) and its 25 labelled
# items therefore leave the pooled estimate. Deriving the rule from the ground
# truth rather than hard-coding the event keeps the two files consistent by
# construction.
GT_EVENTS = {e["event_id"] for e in GT}

# event_id -> display name (paper's tab:precision names)
DISPLAY = {
    "red_sea_crisis_2023": "Red Sea crisis",
    "us_chip_export_controls_2022": "U.S.\\ chip export controls",
    "us_china_tariff_war_2018": "U.S.--China tariff war",
    "taiwan_strait_crisis_2022": "Taiwan Strait crisis",
    "russia_ukraine_war_2022": "Russia--Ukraine war",
    "black_sea_grain_exit_2023": "Black Sea grain exit",
    "uaw_auto_strike_2023": "UAW auto strike",
    "india_wheat_export_ban_2022": "India wheat export ban",
    "egg_shortage_birdflu_2025": "Egg shortage / bird flu",
    "port_los_angeles_backlog_2021": "LA/Long Beach backlog",
    "europe_energy_crisis_2022": "European energy crisis",
    "beirut_port_explosion_2020": "Beirut port explosion",
    "renesas_earthquake_2016": "Renesas / Kumamoto",
    "toyota_steel_explosion_2019": "Toyota steel explosion",
    "covid_supply_disruption_2020": "COVID-19 / Wuhan lockdown",
    "hurricane_maria_2017": "Hurricane Maria",
    "renesas_naka_plant_fire_2021": "Renesas / Naka fire",
}

# order matching the current table in the paper
ORDER = [
    "red_sea_crisis_2023",
    "us_chip_export_controls_2022",
    "us_china_tariff_war_2018",
    "taiwan_strait_crisis_2022",
    "russia_ukraine_war_2022",
    "black_sea_grain_exit_2023",
    "uaw_auto_strike_2023",
    "india_wheat_export_ban_2022",
    "egg_shortage_birdflu_2025",
    "port_los_angeles_backlog_2021",
    "europe_energy_crisis_2022",
    "beirut_port_explosion_2020",
    "renesas_earthquake_2016",
    "toyota_steel_explosion_2019",
    "covid_supply_disruption_2020",
    "hurricane_maria_2017",
    "renesas_naka_plant_fire_2021",
]


def wilson(tp: int, n: int, z: float = 1.96):
    if not n:
        return None, None
    p = tp / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def main():
    rows = list(csv.DictReader(open(WS, encoding="utf-8")))
    by_event = OrderedDict()
    dropped = {}
    for r in rows:
        eid = r["event_id"].strip()
        lab = (r.get("relevant_label_body") or "").strip()
        if lab not in ("0", "1"):
            continue
        if eid not in GT_EVENTS:
            dropped[eid] = dropped.get(eid, 0) + 1
            continue
        d = by_event.setdefault(eid, [0, 0])
        d[1] += 1
        d[0] += 1 if lab == "1" else 0
    if dropped:
        for eid, n in sorted(dropped.items()):
            print(f"excluded (not in gt_events.json): {eid}  n={n}")
        print()

    # order + include any missing
    keys = [k for k in ORDER if k in by_event] + [k for k in by_event if k not in ORDER]
    total_n = total_tp = 0
    lines = []
    for eid in keys:
        tp, n = by_event[eid]
        total_n += n
        total_tp += tp
        lo, hi = wilson(tp, n)
        name = DISPLAY.get(eid, eid)
        lines.append(f"{name} & {n} & {tp} & {pct(tp / n)} [{pct(lo):s},{pct(hi):s}] \\\\")

    plo, phi = wilson(total_tp, total_n)
    print("=== LaTeX table body (tab:precision) ===")
    for ln in lines:
        print(ln)
    print(f"\\midrule")
    print(f"Pooled & {total_n} & {total_tp} & {pct(total_tp / total_n)} [{pct(plo)},{pct(phi)}] \\\\")

    print("\n=== headline ===")
    print(f"pooled: {total_tp}/{total_n} = {total_tp/total_n:.4f}  Wilson95 [{plo:.4f},{phi:.4f}]")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import apply_style, FIG_DIR, TOL

    apply_style()
    names = [DISPLAY.get(k, k) for k in keys]
    precs = [by_event[k][0] / by_event[k][1] for k in keys]
    n_vals = [by_event[k][1] for k in keys]
    los, his = zip(*[wilson(by_event[k][0], by_event[k][1]) for k in keys])
    err_lo = [precs[i] - los[i] for i in range(len(keys))]
    err_hi = [his[i] - precs[i] for i in range(len(keys))]

    # sort descending by precision for readability
    idx = sorted(range(len(keys)), key=lambda i: -precs[i])
    names = [names[i] for i in idx]
    precs = [precs[i] for i in idx]
    err_lo = [err_lo[i] for i in idx]
    err_hi = [err_hi[i] for i in idx]
    n_vals = [n_vals[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = range(len(names))
    ax.barh(y, precs, xerr=[err_lo, err_hi], capsize=2.5,
            color=TOL["blue"], alpha=0.85, ecolor=TOL["black"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Retrieval precision (body round)")
    ax.axvline(total_tp / total_n, color=TOL["red"], ls="--", lw=1,
               label=f"Pooled {total_tp/total_n:.2f}")
    ax.legend(frameon=False, fontsize=7)
    for yi, (pr, nv) in enumerate(zip(precs, n_vals)):
        ax.text(pr + 0.02, yi, f"{nv}", va="center", fontsize=6, color=TOL["grey"])
    fig.tight_layout()
    out = FIG_DIR / "fig_retrieval_precision.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
