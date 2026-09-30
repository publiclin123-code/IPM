#!/usr/bin/env python3
"""Strict-match robustness: loose vs strict entity matching, both inputs.

compute_fwgs in validation/metrics.py hardcodes strict=False, so this script
runs a strict variant alongside it. Outputs results/strict_match_sensitivity.json
and prints the comparison table.

Loose:  commodity OR region OR company
Strict: (commodity AND region) OR company
"""
import json, math, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals, validate, FOREWARD_TEMPORALITIES  # noqa: E402
from metrics import compute_fwgs  # noqa: E402

GT = json.load(open(ROOT / "validation" / "gt_events.json", encoding="utf-8"))["events"]


def fwgs_variant(signals, news_by_id, gt_events, strict: bool, window_days=180,
                 tau=30.0, lambda_penalty=0.5) -> dict:
    from validate import signal_matches_event
    scores = []
    n_tp = n_fp = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        best_lead = None
        is_tp = False
        for ev in gt_events:
            try:
                onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
            except (KeyError, ValueError):
                continue
            lead = (onset - sd).days
            if 0 < lead <= window_days and signal_matches_event(s, ev, strict=strict):
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
        article = news_by_id.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = (sum(1 for t in triggers if t.lower() in text) / len(triggers)
             if triggers else 0.0)
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - lambda_penalty * (1 - g) * conf)
    return {"n_signals": len(scores), "n_tp": n_tp, "n_fp": n_fp,
            "fwgs": round(sum(scores) / len(scores), 4) if scores else None}


def run(res_dir: Path, data_dir: Path, label: str) -> dict:
    out = {}
    for strict in (False, True):
        n = tp = fp = 0
        hits = 0
        fwgs_w = 0.0
        for ev in GT:
            eid = ev["event_id"]
            sp = res_dir / f"{eid}_signals.jsonl"
            if not sp.exists():
                continue
            sigs = load_signals(str(sp))
            news = {a["id"]: a for a in (json.loads(l) for l in
                    open(data_dir / f"{eid}.jsonl", encoding="utf-8"))}
            fw = fwgs_variant(sigs, news, [ev], strict=strict)
            n += fw["n_signals"]; tp += fw["n_tp"]; fp += fw["n_fp"]
            fwgs_w += (fw["fwgs"] or 0.0) * fw["n_signals"]
            val = validate(sigs, [ev], window_days=180, strict=strict)
            per = next((x for x in val["per_event"] if x["event_id"] == eid), None)
            if per and per.get("hit"):
                hits += 1
        prec = tp / (tp + fp) if (tp + fp) else None
        fwgs = fwgs_w / n if n else None
        key = "strict" if strict else "loose"
        out[key] = {"n_signals": n, "tp": tp, "fp": fp,
                    "precision": round(prec, 4) if prec is not None else None,
                    "fwgs": round(fwgs, 4) if fwgs is not None else None,
                    "hit_events": hits, "n_events": len(GT)}
        print(f"{label:6s} {key:6s}: n_sig={n} TP={tp} FP={fp} "
              f"prec={out[key]['precision']} FWGS={out[key]['fwgs']} "
              f"hits={hits}/{len(GT)}")
    return out


res = {
    "title": run(ROOT / "results" / "v2", ROOT / "data" / "by_event", "title"),
    "body": run(ROOT / "results" / "v2_body", ROOT / "data" / "by_event_body", "body"),
}
out = ROOT / "results" / "strict_match_sensitivity.json"
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nwritten {out}")

# headline reading
for k in ("title", "body"):
    l, s = res[k]["loose"], res[k]["strict"]
    print(f"{k}: loose prec {l['precision']} hits {l['hit_events']} -> "
          f"strict prec {s['precision']} hits {s['hit_events']} "
          f"(n_sig {l['n_signals']}->{s['n_signals']})")
