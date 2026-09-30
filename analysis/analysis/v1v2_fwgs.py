"""Ontology-version comparison: v1 naive three-way vs v2 clock-split FWGS.

Reads v1 (results/{event}_signals.jsonl, naive prompt) and v2
(results/v2/{event}_signals.jsonl, clock-split prompt) over the same 8-event
pre-onset corpus, and reports per-event forward-signal count and FWGS for both.

The split reclassifies post-trigger reports as confirmations, so the
forward-signal count should fall and the surviving forward signals should be
more often true precursors (pooled FWGS should rise).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES_V1 = ROOT / "results"
RES_V2 = ROOT / "results" / "v2"
DATA = ROOT / "data" / "by_event"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023", "us_chip_export_controls_2022",
    "renesas_earthquake_2016", "toyota_steel_explosion_2019",
    "port_los_angeles_backlog_2021", "covid_supply_disruption_2020",
    "europe_energy_crisis_2022", "renesas_naka_plant_fire_2021",
]
WINDOW = 180
TAU, LAM = 30.0, 0.5


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_signals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") != "error":
                out.append(s)
    return out


def load_news(eid: str) -> dict:
    p = DATA / f"{eid}.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("id", "")] = r
    return out


def fwgs_for(signals: list[dict], news: dict, gt: list[dict]) -> dict:
    scores = []
    n_tp = n_fp = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        best_lead = None
        is_tp = False
        for ev in gt:
            try:
                onset = _parse_date(ev["gt_onset_date"])
            except (KeyError, ValueError):
                continue
            lead = (onset - sd).days
            if 0 < lead <= WINDOW and signal_matches_event(s, ev, strict=False):
                is_tp = True
                if best_lead is None or lead > best_lead:
                    best_lead = lead
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        lead = best_lead if best_lead is not None else 0
        w_f = 1 - math.exp(-lead / TAU)
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return {"fwgs": sum(scores) / len(scores) if scores else None,
            "n": len(scores), "n_tp": n_tp, "n_fp": n_fp}


def main() -> int:
    gt = load_gt()
    print("=== Ontology-version comparison (FWGS, tau=30, lambda=0.5) ===")
    print(f"{'event':<36} {'n_v1':>5} {'FWGS_v1':>8} {'n_v2':>5} {'FWGS_v2':>8}")
    rows = []
    tot_n1 = tot_fw1 = tot_n2 = tot_fw2 = 0.0
    for eid in EVENTS:
        news = load_news(eid)
        r1 = fwgs_for(load_signals(RES_V1 / f"{eid}_signals.jsonl"), news, gt)
        r2 = fwgs_for(load_signals(RES_V2 / f"{eid}_signals.jsonl"), news, gt)
        f1 = f"{r1['fwgs']:.3f}" if r1["fwgs"] is not None else "--"
        f2 = f"{r2['fwgs']:.3f}" if r2["fwgs"] is not None else "--"
        print(f"{eid:<36} {r1['n']:>5} {f1:>8} {r2['n']:>5} {f2:>8}")
        if r1["fwgs"] is not None:
            tot_fw1 += r1["fwgs"] * r1["n"]; tot_n1 += r1["n"]
        if r2["fwgs"] is not None:
            tot_fw2 += r2["fwgs"] * r2["n"]; tot_n2 += r2["n"]
        rows.append((eid, r1["n"], r1["fwgs"], r2["n"], r2["fwgs"]))

    p1 = tot_fw1 / tot_n1 if tot_n1 else None
    p2 = tot_fw2 / tot_n2 if tot_n2 else None
    print(f"\nPooled n-weighted: v1 n={int(tot_n1)} FWGS={p1:.3f} | v2 n={int(tot_n2)} FWGS={p2:.3f}")

    print("\n% === tab:v1v2 ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"event & $n_{\text{sig}}^{v1}$ & FWGS$^{v1}$ & $n_{\text{sig}}^{v2}$ & FWGS$^{v2}$ \\")
    print(r"\midrule")
    for eid, n1, f1, n2, f2 in rows:
        e = eid.replace("_", r"\_")
        c1 = f"{f1:.3f}" if f1 is not None else "--"
        c2 = f"{f2:.3f}" if f2 is not None else "--"
        print(f"{e} & {n1} & {c1} & {n2} & {c2} \\\\")
    print(r"\midrule")
    print(f"\\textbf{{Pooled ($n$-weighted)}} & \\textbf{{{int(tot_n1)}}} & $\\mathbf{{{p1:.3f}}}$ & \\textbf{{{int(tot_n2)}}} & $\\mathbf{{{p2:.3f}}}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
