#!/usr/bin/env python3
"""FWGS component ablation, 18-event single-match (no cross-event lead inflation).

Replaces the outdated 8-event, full-GT ablation table. Prints the LaTeX body.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import FOREWARD_TEMPORALITIES, signal_matches_event, _parse_date  # noqa: E402

RES = ROOT / "results" / "v2"
DATA = ROOT / "data" / "by_event"
GT = ROOT / "validation" / "gt_events.json"
TAU, LAM = 30.0, 0.5


def load_signals(eid):
    p = RES / f"{eid}_signals.jsonl"
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        if s.get("status") != "error":
            out.append(s)
    return out


def load_news(eid):
    p = DATA / f"{eid}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r.get("id", "")] = r
    return out


def fwgs_ablation(signals, news, ev, ablate):
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
        w_f = 1 - math.exp(-lead / TAU) if lead > 0 else 0.0
        triggers = s.get("trigger_phrases", []) or []
        art = news.get(s.get("input_id", ""), {})
        text = ((art.get("title", "") or "") + " " + (art.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        if ablate == "full":
            sc = w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf
        elif ablate == "-w_f":
            sc = 1.0 * g * (1 if is_tp else 0) - LAM * (1 - g) * conf
        elif ablate == "-g":
            sc = w_f * 1.0 * (1 if is_tp else 0)
        elif ablate == "-lambda":
            sc = w_f * g * (1 if is_tp else 0)
        scores.append(sc)
    return sum(scores) / len(scores) if scores else None


def main():
    gt = json.load(open(GT, encoding="utf-8"))["events"]
    rows = []
    for ev in gt:
        eid = ev["event_id"]
        sigs = load_signals(eid)
        news = load_news(eid)
        full = fwgs_ablation(sigs, news, ev, "full")
        if full is None:
            continue
        rows.append((eid, full,
                     fwgs_ablation(sigs, news, ev, "-w_f"),
                     fwgs_ablation(sigs, news, ev, "-g"),
                     fwgs_ablation(sigs, news, ev, "-lambda")))
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"event & full & $-w_F$ & $-g$ & $-\lambda$ \\")
    print(r"\midrule")
    for eid, a, b, c, d in rows:
        print(f"{eid} & {a:.3f} & {b:.3f} & {c:.3f} & {d:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
