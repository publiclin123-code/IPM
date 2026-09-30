#!/usr/bin/env python3
"""Fig: Ontology-version comparison as a dumbbell plot (v1 -> v2 FWGS per event).

Each event = one horizontal dumbbell: left dot = FWGS under the naive (v1)
three-way label, right dot = FWGS under the clock-split (v2) ontology, coloured
by a priori type. A rightward arrow = the split lifted the score (surviving
forward signals are more often true precursors); a leftward arrow (sudden
events) = the split removed hindsight-contaminated long-lead hits.

Reads results/v2 (v2) and results/ (v1 naive) signals, single-event match.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))
from figstyle import TOL, TYPE_COLOR, TYPE_LABEL, apply_style, save  # noqa: E402
from event_meta import EVENT_SHORT, EVENT_TYPES  # noqa: E402
from validate import load_signals, FOREWARD_TEMPORALITIES, signal_matches_event, _parse_date  # noqa: E402
from metrics import load_jsonl  # noqa: E402

GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
RES_V1 = ROOT / "results"
RES_V2 = ROOT / "results" / "v2"
TAU, LAM = 30.0, 0.5


def fwgs_single(signals, news, ev):
    scores = []
    onset = _parse_date(ev["gt_onset_date"])
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        lead = (onset - sd).days
        is_tp = 0 < lead <= 180 and signal_matches_event(s, ev, strict=False)
        w_f = 1 - np.exp(-lead / TAU) if lead > 0 else 0.0
        triggers = s.get("trigger_phrases", []) or []
        art = news.get(s.get("input_id", ""), {})
        text = ((art.get("title", "") or "") + " " + (art.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return sum(scores) / len(scores) if scores else None


def main():
    apply_style()
    gt = json.load(open(GT, encoding="utf-8"))["events"]
    gt_by_id = {e["event_id"]: e for e in gt}

    rows = []
    for ev in gt:
        eid = ev["event_id"]
        news_path = DATA / f"{eid}.jsonl"
        news = {a["id"]: a for a in (load_jsonl(str(news_path)) if news_path.exists() else [])}
        sp1 = RES_V1 / f"{eid}_signals.jsonl"
        sp2 = RES_V2 / f"{eid}_signals.jsonl"
        sigs1 = [s for s in load_signals(str(sp1)) if s.get("status") != "error"] if sp1.exists() else []
        sigs2 = [s for s in load_signals(str(sp2)) if s.get("status") != "error"] if sp2.exists() else []
        f1 = fwgs_single(sigs1, news, ev)
        f2 = fwgs_single(sigs2, news, ev)
        if f1 is not None or f2 is not None:
            rows.append((eid, f1 if f1 is not None else 0.0, f2 if f2 is not None else 0.0))

    # 排序: 按 v2 FWGS 降序
    rows.sort(key=lambda r: -r[2])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(6.0, max(4.0, n * 0.30)))
    y = np.arange(n)

    for i, (eid, f1, f2) in enumerate(rows):
        typ = EVENT_TYPES.get(eid, "sudden")
        color = TYPE_COLOR[typ]
        # 连接线 + 两端点
        ax.plot([f1, f2], [i, i], "-", color="#C9C9C9", lw=1.0, zorder=1)
        ax.plot(f1, i, "o", ms=6, color="white", mec=color, mew=1.6, zorder=3)
        ax.plot(f2, i, "o", ms=8, color=color, mec="white", mew=0.8, zorder=3)
        # 方向箭头标记（涨跌）
        if f2 > f1 + 0.01:
            ax.text((f1 + f2) / 2, i + 0.22, "→", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
        elif f1 > f2 + 0.01:
            ax.text((f1 + f2) / 2, i + 0.22, "←", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([EVENT_SHORT.get(eid, eid) for eid, _, _ in rows], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("FWGS")
    ax.xaxis.grid(True, ls=":", color=TOL["grey"])
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 图例: naive 空心 / clock 实心 (深浅区分) + 类型三色, 单列排列
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="white", marker="o", mec="#6B7280", mew=1.6,
               ms=6, label="naive (v1)"),
        Line2D([], [], color="#6B7280", marker="o", mec="white", mew=0.8,
               ms=8, label="clock-split (v2)"),
    ]
    handles += [Line2D([], [], color=TYPE_COLOR[t], lw=0, marker="o", ms=7,
                       label=TYPE_LABEL[t])
                for t in ("slow_burn", "creeping", "sudden")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right", ncol=1)

    fig.tight_layout()
    save(fig, "fig_v1v2_dumbbell")
    plt.close(fig)
    print(f"dumbbell: {n} events")
    return 0


if __name__ == "__main__":
    main()
