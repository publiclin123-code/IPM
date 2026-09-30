"""Fig 3 (redesign) — Signal swimlane timeline.

One horizontal lane per event (ordered slow_burn → creeping → sudden, within-type
by onset date). X axis is days-from-onset (relative), shared across lanes so the
2016-2023 event span does not compress. The onset is a red dashed vertical at
x=0; the forward window [-180, 0] is shaded light grey; the post-onset impact
window (0, +45] is shaded light red.

Forward/latent signals are filled circles left of onset, coloured by a priori
event type, marker size ∝ confidence, alpha split TP vs FP (protocol match).
Confirmation signals are small grey ticks right of onset (impact realised).

Reads results/v2/{eid}_signals.jsonl (qwen detector). Self-contained loader.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))

from figstyle import TOL, TYPE_COLOR, TYPE_LABEL, apply_style, save  # noqa: E402
from validate import (  # noqa: E402
    FOREWARD_TEMPORALITIES,
    _parse_date,
    signal_matches_event,
)
from event_meta import EVENT_META  # noqa: E402

RES = ROOT / "results" / "v2"
VAL = ROOT / "validation"
WINDOW = 180
POST_WINDOW = 45


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_signals(eid: str) -> list[dict]:
    p = RES / f"{eid}_signals.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") == "error":
                continue
            out.append(s)
    return out


def dated_signals(gt_by_id: dict[str, dict]) -> list[dict]:
    """Every non-error signal with a parseable date, tagged with dfo + TP flag."""
    rows = []
    for eid, (short, typ) in EVENT_META.items():
        ev = gt_by_id.get(eid)
        if ev is None:
            continue
        onset = _parse_date(ev["gt_onset_date"])
        for s in load_signals(eid):
            sd = s.get("signal_date")
            if not sd:
                continue
            try:
                dfo = (_parse_date(sd) - onset).days
            except (ValueError, TypeError):
                continue
            temp = s.get("temporality")
            is_forward = temp in FOREWARD_TEMPORALITIES
            in_window = -WINDOW <= dfo <= POST_WINDOW
            is_tp = (is_forward and 0 < dfo <= WINDOW
                     and signal_matches_event(s, ev, strict=False))
            rows.append({
                "eid": eid, "short": short, "type": typ,
                "dfo": dfo, "in_window": in_window,
                "temp": temp, "is_forward": is_forward, "is_tp": is_tp,
                "conf": float(s.get("confidence", 0) or 0),
            })
    return rows


def lane_order() -> list[str]:
    """Lane short labels top→bottom: slow_burn, creeping, sudden (onset asc)."""
    order_type = {"slow_burn": 0, "creeping": 1, "sudden": 2}
    gt = {e["event_id"]: _parse_date(e["gt_onset_date"]) for e in load_gt()}
    items = []
    for eid, (short, typ) in EVENT_META.items():
        items.append((order_type[typ], gt.get(eid, datetime(2100, 1, 1)), short))
    items.sort(key=lambda t: (t[0], t[1]))
    return [it[2] for it in items]


def main() -> int:
    apply_style()
    gt_by_id = {e["event_id"]: e for e in load_gt()}
    rows = dated_signals(gt_by_id)
    lanes = lane_order()
    lane_y = {name: i for i, name in enumerate(lanes)}
    n = len(lanes)

    fig, ax = plt.subplots(figsize=(7.1, max(3.8, n * 0.26)))

    # background bands
    ax.axvspan(-WINDOW, 0, color=TOL["grey"], alpha=0.08, lw=0, zorder=0)
    ax.axvspan(0, POST_WINDOW, color=TOL["red"], alpha=0.07, lw=0, zorder=0)
    ax.axvline(0, color=TOL["red"], ls="--", lw=1.1, zorder=1)

    # lane separators + labels
    for name, y in lane_y.items():
        ax.axhline(y, color="#E5E5E5", lw=0.5, zorder=0)
        ax.text(-WINDOW - 8, y, name, ha="right", va="center", fontsize=7.5)

    # confirmation ticks (post-onset, impact realised)
    for r in rows:
        if r["temp"] == "confirmation" and 0 < r["dfo"] <= POST_WINDOW:
            y = lane_y.get(r["short"])
            if y is None:
                continue
            ax.plot([r["dfo"], r["dfo"]], [y - 0.18, y + 0.18],
                    color=TOL["grey"], lw=1.0, alpha=0.55, zorder=2)

    # forward / latent circles (the warnings)
    for r in rows:
        if not r["is_forward"] or not r["in_window"]:
            continue
        y = lane_y.get(r["short"])
        if y is None:
            continue
        color = TYPE_COLOR[r["type"]]
        size = 10 + r["conf"] * 28
        alpha = 0.88 if r["is_tp"] else 0.32
        ax.plot([r["dfo"]], [y], marker="o", ms=size ** 0.5 * 2.0,
                color=color, alpha=alpha, mec="white", mew=0.4, zorder=3)

    # onset annotation
    ax.text(0, n - 0.35, "onset", color=TOL["red"], fontsize=7,
            ha="center", va="bottom", fontweight="bold")
    ax.text(-WINDOW / 2, n - 0.35, "forward window (180 d)",
            color="#888", fontsize=7, ha="center", va="bottom")

    ax.set_xlim(-WINDOW - 55, POST_WINDOW + 6)
    ax.set_ylim(-0.6, n - 0.1)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Days from onset  (negative = pre-event)")
    ax.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30])

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TYPE_COLOR[t], lw=0, marker="o", ms=7,
                      label=TYPE_LABEL[t])
               for t in ("slow_burn", "creeping", "sudden")]
    handles.append(Line2D([], [], color=TOL["grey"], lw=0, marker="|", ms=9,
                          label="Confirmation (impact realised)"))
    handles.append(Line2D([], [], color=TOL["red"], lw=1.1, ls="--",
                          label="Onset"))
    fig.tight_layout()
    fig.legend(handles=handles, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.005), fontsize=7, ncol=5,
               columnspacing=1.3, handletextpad=0.4)
    save(fig, "fig3_swimlane")
    plt.close(fig)
    n_fwd = sum(1 for r in rows if r["is_forward"] and r["in_window"])
    print(f"swimlane: {len(rows)} dated signals, {n_fwd} forward in window, "
          f"{n} lanes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
