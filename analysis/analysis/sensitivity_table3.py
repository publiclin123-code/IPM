"""Comprehensive RQ4 analysis: Sensitivity (tau/lambda grid) + Table 3.

Reads per-event signals + GT, reuses validate.signal_matches_event /
metrics.compute_fwgs to compute:
  (A) FWGS sensitivity grid: tau in {15,30,60}, lambda in {0.25,0.5,1.0}
  (B) Table 3: per-event hit / lead time (mean+median) / precision / recall
Outputs LaTeX table bodies for sec:res-rq4 (tab:sensitivity, tab:rq3_hits).
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results"
DATA = ROOT / "data" / "by_event"
GT_PATH = VAL / "gt_events.json"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023",
    "us_chip_export_controls_2022",
    "renesas_earthquake_2016",
    "port_los_angeles_backlog_2021",
    "toyota_steel_explosion_2019",
]

WINDOW = 180
TAUS = [15, 30, 60]
LAMBDAS = [0.25, 0.5, 1.0]

# toyota/warehouse data files (warehouse uses the 100-article sample for faithfulness)
DATA_OVERRIDE = {
    "warehouse_collapse_lithium_2019": DATA / "warehouse_sample_100.jsonl",
}


def load_gt() -> list[dict]:
    with open(GT_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["events"]


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
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("status") == "error":
                continue
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
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r.get("id", "")] = r
    return out


def fwgs_score(signals: list[dict], news: dict, gt: list[dict],
               tau: float, lam: float) -> dict:
    """Inline copy of metrics.compute_fwgs with explicit tau/lambda."""
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
        w_f = 1 - math.exp(-lead / tau)
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = (sum(1 for t in triggers if t.lower() in text) / len(triggers)
             if triggers else 0.0)
        conf = float(s.get("confidence", 0))
        score = w_f * g * (1 if is_tp else 0) - lam * (1 - g) * conf
        scores.append(score)
    return {
        "fwgs": sum(scores) / len(scores) if scores else None,
        "n": len(scores), "n_tp": n_tp, "n_fp": n_fp,
    }


def event_hit_stats(signals: list[dict], gt: list[dict]) -> dict:
    """Per-event: hit, earliest lead time, matched count (strict=False 与正文一致)."""
    out = {}
    for ev in gt:
        eid = ev["event_id"]
        try:
            onset = _parse_date(ev["gt_onset_date"])
        except (KeyError, ValueError):
            continue
        wstart = onset - timedelta(days=WINDOW)
        matches = []
        for s in signals:
            try:
                sd = _parse_date(s["signal_date"])
            except (KeyError, ValueError):
                continue
            if not (wstart <= sd < onset):
                continue
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            if signal_matches_event(s, ev, strict=False):
                matches.append((sd, s))
        if matches:
            matches.sort(key=lambda x: x[0])
            out[eid] = {"hit": True, "lead": (onset - matches[0][0]).days,
                        "n_matched": len(matches)}
        else:
            out[eid] = {"hit": False, "lead": None, "n_matched": 0}
    return out


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
    # ---- (A) Sensitivity grid ----
    print("=== (A) FWGS sensitivity (pooled n-weighted) ===")
    print(f"{'tau':>4} {'lambda':>6} {'FWGS':>7} {'n':>5} {'TP':>4} {'FP':>4}")
    pooled = {}
    for tau in TAUS:
        for lam in LAMBDAS:
            tot_fw = tot_n = 0.0
            for eid in events:
                sigs = load_signals(eid)
                news = load_news(eid)
                r = fwgs_score(sigs, news, gt, tau, lam)
                if r["fwgs"] is not None:
                    tot_fw += r["fwgs"] * r["n"]
                    tot_n += r["n"]
            pooled[(tau, lam)] = (tot_fw / tot_n if tot_n else None, int(tot_n))
            pv = pooled[(tau, lam)][0]
            print(f"{tau:>4} {lam:>6} {pv if pv is None else round(pv,4):>7} "
                  f"{int(tot_n):>5}")

    print("\n=== (B) Per-event hit/lead (Table 3) ===")
    print(f"{'event':<38} {'hit':>4} {'lead_mean':>10} {'lead_med':>9} {'prec':>6} {'recall':>7} {'n_sig':>6}")
    rows = []
    for eid in events:
        sigs = load_signals(eid)
        news = load_news(eid)
        # hit stats per event (aggregate across all GT events, but report per own event)
        hs = event_hit_stats(sigs, gt)
        own = hs.get(eid, {"hit": False, "lead": None, "n_matched": 0})
        # precision: matched / all warning signals
        n_fwd = sum(1 for s in sigs if s.get("temporality") in FOREWARD_TEMPORALITIES)
        prec = own["n_matched"] / n_fwd if n_fwd else None
        # recall: for this event, 1 if hit else 0 (GT is single-event per event set)
        rec = 1.0 if own["hit"] else 0.0
        rows.append((eid, own, n_fwd, prec, rec))
        print(f"{eid:<38} {str(own['hit']):>4} "
              f"{str(own['lead']):>10} {str(own['lead']):>9} "
              f"{prec if prec is None else round(prec,3):>6} {rec:>7} {n_fwd:>6}")

    # ---- LaTeX table bodies ----
    print("\n% === tab:sensitivity === (tau rows, lambda cols)")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"$\tau$ & $\lambda{=}0.25$ & $\lambda{=}0.5$ & $\lambda{=}1.0$ \\")
    print(r"\midrule")
    for tau in TAUS:
        cells = []
        for lam in LAMBDAS:
            v = pooled[(tau, lam)][0]
            cells.append(f"{v:.3f}" if v is not None else "--")
        print(f"{tau} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n% === tab:rq3_hits === (per-event hit/lead/prec/recall)")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"event & hit & lead\,(d) & prec & recall & $n_{\text{sig}}$ \\")
    print(r"\midrule")
    for eid, own, n_fwd, prec, rec in rows:
        h = r"\checkmark" if own["hit"] else "--"
        lead = str(own["lead"]) if own["lead"] is not None else "--"
        pc = f"{prec:.3f}" if prec is not None else "--"
        e = eid.replace("_", r"\_")
        print(f"{e} & {h} & {lead} & {pc} & {rec:.2f} & {n_fwd} \\\\")
    n_hit = sum(1 for _, o, _, _, _ in rows if o["hit"])
    print(r"\midrule")
    print(f"\\textbf{{Aggregate}} & {n_hit}/{len(rows)} & -- & -- & -- & -- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
