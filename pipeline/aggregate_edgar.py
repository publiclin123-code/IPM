#!/usr/bin/env python3
"""Phase 2 · EDGAR financial-distress aggregation (per-event single-match).

Mirrors pipeline/aggregate_events.py but for the EDGAR corpus. Computes
per-event forward-signal count, TP/FP, precision, FWGS, and pooled stats,
using single-event matching (no cross-event lead inflation).

Usage: python pipeline/aggregate_edgar.py [--json /tmp/edgar.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals, validate, FOREWARD_TEMPORALITIES  # noqa: E402
from metrics import compute_fwgs, load_jsonl  # noqa: E402

GT_PATH = ROOT / "validation" / "edgar_bankruptcy_events.json"
DATA = ROOT / "data" / "edgar"
RES = ROOT / "results" / "edgar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--res-dir", default="", help="信号目录 (default: results/edgar; DeepSeek 用 results/deepseek/edgar)")
    args = ap.parse_args()

    res_dir = Path(args.res_dir) if args.res_dir else RES

    gt = json.load(open(GT_PATH, encoding="utf-8"))["events"]
    rows = []
    for ev in gt:
        eid = ev["event_id"]
        sig_path = res_dir / f"{eid}_signals.jsonl"
        news_path = DATA / f"{eid}.jsonl"
        n_art = sum(1 for _ in open(news_path, encoding="utf-8")) if news_path.exists() else 0
        if not sig_path.exists():
            rows.append({"event_id": eid, "n_filings": n_art, "n_sig": 0,
                         "tp": 0, "fp": 0, "fwgs": None, "hit": False, "lead": None})
            continue
        signals = [s for s in load_signals(str(sig_path)) if s.get("status") != "error"]
        news = load_jsonl(str(news_path)) if news_path.exists() else []
        news_by_id = {a.get("id", ""): a for a in news}
        # 单事件匹配
        fw = compute_fwgs(signals, news_by_id, [ev], window_days=180)
        n_sig, tp, fp, fwgs = fw["n_signals"], fw["n_tp"], fw["n_fp"], fw["fwgs"]
        val = validate(signals, [ev], window_days=180, strict=False)
        per = next(p for p in val["per_event"] if p["event_id"] == eid)
        rows.append({"event_id": eid, "name": ev["name"], "ticker": ev["ticker"],
                     "onset": ev["gt_onset_date"], "n_filings": n_art,
                     "n_sig": n_sig, "tp": tp, "fp": fp, "fwgs": fwgs,
                     "hit": per.get("hit", False), "lead": per.get("lead_time_days")})

    total_filings = sum(r["n_filings"] for r in rows)
    total_sig = sum(r["n_sig"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    wsum = sum((r["fwgs"] or 0.0) * r["n_sig"] for r in rows)
    pooled_fwgs = wsum / total_sig if total_sig else None

    out = {
        "n_events": len(rows), "total_filings": total_filings,
        "total_forward_signals": total_sig, "total_tp": total_tp, "total_fp": total_fp,
        "aggregate_precision": round(prec, 4) if prec is not None else None,
        "pooled_fwgs": round(pooled_fwgs, 4) if pooled_fwgs is not None else None,
        "per_event": rows,
    }
    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def fnum(x):
        return "--" if x is None else f"{x:.3f}"

    print(f"{'event':30s} {'onset':10s} {'fil':>4s} {'n_sig':>5s} {'TP':>3s} {'FP':>3s} {'prec':>6s} {'FWGS':>6s} {'hit':>4s} {'lead':>5s}")
    for r in rows:
        print(f"{r['event_id']:30s} {r['onset']:10s} {r['n_filings']:4d} {r['n_sig']:5d} "
              f"{r['tp']:3d} {r['fp']:3d} {fnum(r['tp']/(r['tp']+r['fp']) if (r['tp']+r['fp']) else None):>6s} "
              f"{fnum(r['fwgs']):>6s} {'Y' if r['hit'] else '-':>4s} "
              f"{'--' if r['lead'] is None else r['lead']:>5}")
    print("-" * 78)
    print(f"{'POOLED':30s} {'':10s} {total_filings:4d} {total_sig:5d} {total_tp:3d} {total_fp:3d} "
          f"{fnum(prec):>6s} {fnum(pooled_fwgs):>6s}")
    if args.json:
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
