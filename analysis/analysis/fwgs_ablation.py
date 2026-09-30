"""FWGS component ablation.

FWGS = mean_i [ w_F(ell_i) * g_i * 1[TP_i] - lambda * (1-g_i) * conf_i ]

Ablations (each removes ONE component):
  Full   : w_F * g * TP - lambda*(1-g)*conf   (baseline)
  -w_F   : 1   * g * TP - lambda*(1-g)*conf   (no foresight weighting)
  -g     : w_F * 1 * TP - lambda*(1-1)*conf   (no faithfulness: assume all grounded)
  -lambda: w_F * g * TP - 0                    (no hallucination penalty)

Metric: per-event FWGS + ranking. A component matters if removing it
collapses the gap between predictable (red_sea) and sudden (renesas) events,
or flips the ranking.
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results"
DATA = ROOT / "data" / "by_event"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023", "us_chip_export_controls_2022", "renesas_earthquake_2016",
    "global_chip_shortage_2021", "port_los_angeles_backlog_2021",
    "toyota_steel_explosion_2019", "warehouse_collapse_lithium_2019",
]
DATA_OVERRIDE = {"warehouse_collapse_lithium_2019": DATA / "warehouse_sample_100.jsonl"}
WINDOW = 180
TAU, LAM = 30.0, 0.5

ABLATIONS = ["full", "-w_f", "-g", "-lambda"]


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
            if s.get("status") != "error":
                out.append(s)
    return out


def load_news(eid: str) -> dict:
    p = DATA_OVERRIDE.get(eid, DATA / f"{eid}.jsonl")
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


def fwgs_ablation(signals: list[dict], news: dict, gt: list[dict], ablate: str) -> dict:
    """Compute FWGS under an ablation variant."""
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

        if ablate == "full":
            score = w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf
        elif ablate == "-w_f":
            score = 1.0 * g * (1 if is_tp else 0) - LAM * (1 - g) * conf
        elif ablate == "-g":
            score = w_f * 1.0 * (1 if is_tp else 0) - LAM * (1 - 1.0) * conf
        elif ablate == "-lambda":
            score = w_f * g * (1 if is_tp else 0) - 0.0
        else:
            raise ValueError(ablate)
        scores.append(score)
    return {"fwgs": sum(scores) / len(scores) if scores else None, "n": len(scores)}


def main() -> int:
    import argparse
    global RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default=str(RES),
                    help="results dir; default v1, pass results/v2 for clock-split run")
    ap.add_argument("--events", default=None,
                    help="comma-separated event ids (default: all in EVENTS)")
    args = ap.parse_args()
    RES = Path(args.res_dir)
    events = args.events.split(",") if args.events else EVENTS
    gt = load_gt()
    per_event = {}
    for eid in events:
        sigs = load_signals(eid)
        news = load_news(eid)
        per_event[eid] = {a: fwgs_ablation(sigs, news, gt, a) for a in ABLATIONS}

    print("=== FWGS component ablation ===")
    print(f"{'event':<38}" + "".join(f"{a:>10}" for a in ABLATIONS))
    for eid in events:
        cells = []
        for a in ABLATIONS:
            fw = per_event[eid][a]["fwgs"]
            cells.append("--" if fw is None else f"{fw:.3f}")
        print(f"{eid:<38}" + "".join(f"{c:>10}" for c in cells))

    # Gap between red_sea (predictable) and renesas (sudden) under each ablation
    print("\n=== Predictable-vs-sudden gap (red_sea - renesas) ===")
    for a in ABLATIONS:
        rs = per_event["red_sea_crisis_2023"][a]["fwgs"]
        rn = per_event["renesas_earthquake_2016"][a]["fwgs"]
        gap = rs - rn if (rs is not None and rn is not None) else None
        print(f"{a:>8}: red_sea={rs:.3f}, renesas={rn:.3f}, gap={gap:.3f}")

    # Pooled n-weighted per ablation
    print("\n=== Pooled (n-weighted) per ablation ===")
    for a in ABLATIONS:
        tot_fw = tot_n = 0.0
        for eid in events:
            r = per_event[eid][a]
            if r["fwgs"] is not None:
                tot_fw += r["fwgs"] * r["n"]
                tot_n += r["n"]
        print(f"{a:>8}: {tot_fw/tot_n:.3f} (n={int(tot_n)})")

    # LaTeX table
    print("\n% === tab:fwgs_ablation ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"event & full & $-w_F$ & $-g$ & $-\lambda$ \\")
    print(r"\midrule")
    for eid in events:
        cells = []
        for a in ABLATIONS:
            fw = per_event[eid][a]["fwgs"]
            cells.append(f"{fw:.3f}" if fw is not None else "--")
        e = eid.replace("_", r"\_")
        print(f"{e} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
