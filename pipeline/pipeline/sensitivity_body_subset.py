#!/usr/bin/env python3
"""Sensitivity: title-vs-body on the SAME rows (body-available subset only).

The headline compare (results/compare_title_body.json) pools over all 2,555
rows; the 218 title_fallback rows get identical title-only input in both
columns. A reviewer may ask whether those 8.5% dilute the body-side gains.
Answer: recompute both columns restricted to the 2,337 rows whose body was
recovered. If direction/magnitude survives, the dilution is immaterial.

Reuses the exact same aggregation math as aggregate_events.py (load_signals
from validation.validate, compute_fwgs from validation.metrics), differing
only in that signals are filtered to body-available article ids on BOTH sides.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))

from validate import load_signals  # noqa: E402
from metrics import compute_fwgs  # noqa: E402

GT_PATH = ROOT / "validation" / "gt_events.json"

# body-available article ids: eid -> set of input_id with a genuine body
avail: dict[str, set[str]] = {}
for p in (ROOT / "data" / "by_event_body").glob("*.jsonl"):
    if "_background" in p.name:
        continue
    eid = p.stem
    s = set()
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("body_status") != "title_fallback" and r.get("text"):
            s.add(r["id"])
    avail[eid] = s

def per_event_rows(res_dir: Path, gt_events: list[dict]) -> list[dict]:
    rows = []
    for ev in gt_events:
        eid = ev["event_id"]
        sig_path = res_dir / f"{eid}_signals.jsonl"
        if not sig_path.exists():
            continue
        ids = avail.get(eid, set())
        signals = [s for s in load_signals(str(sig_path)) if s.get("input_id") in ids]
        news = {a["id"]: a for a in (json.loads(l) for l in
                open(ROOT / "data" / "by_event_body" / f"{eid}.jsonl", encoding="utf-8"))
                if a["id"] in ids}
        fw = compute_fwgs(signals, news, [ev], window_days=180)
        rows.append({"event_id": eid,
                     "n_sig": fw["n_signals"], "tp": fw["n_tp"], "fp": fw["n_fp"],
                     "fwgs": fw["fwgs"]})
    return rows

def pooled(rows: list[dict]) -> tuple[int, int, int, float]:
    n = sum(r["n_sig"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fwgs = (sum((r["fwgs"] or 0.0) * r["n_sig"] for r in rows) / n) if n else None
    return n, tp, fp, fwgs

def main():
    gt = json.load(open(GT_PATH, encoding="utf-8"))["events"]
    rows_t = per_event_rows(ROOT / "results" / "v2", gt)
    rows_b = per_event_rows(ROOT / "results" / "v2_body", gt)
    n_t, tp_t, fp_t, fwgs_t = pooled(rows_t)
    n_b, tp_b, fp_b, fwgs_b = pooled(rows_b)
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else None
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) else None

    print(f"body-available rows: {sum(len(v) for v in avail.values())}")
    if prec_t is not None:
        print(f"SUBSET title : n_sig={n_t} TP={tp_t} FP={fp_t} prec={prec_t:.4f} FWGS={fwgs_t:.4f}")
    if prec_b is not None:
        print(f"SUBSET body  : n_sig={n_b} TP={tp_b} FP={fp_b} prec={prec_b:.4f} FWGS={fwgs_b:.4f}")

    ref = json.load(open(ROOT / "results" / "compare_title_body.json", encoding="utf-8"))
    print("\nFULL-POOL reference (from compare_title_body.json):")
    print(f"title: prec={ref['title_only']['aggregate_precision']:.4f} FWGS={ref['title_only']['pooled_fwgs']:.4f} n={ref['title_only']['total_forward_signals']}")
    print(f"body : prec={ref['body']['aggregate_precision']:.4f} FWGS={ref['body']['pooled_fwgs']:.4f} n={ref['body']['total_forward_signals']}")

if __name__ == "__main__":
    main()
