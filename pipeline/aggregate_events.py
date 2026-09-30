#!/usr/bin/env python3
"""聚合全部事件 → 论文所有表格数字 (per-event n_sig/TP/FP/prec/FWGS + pooled).

复用 validate.py / metrics.py 的函数, 用当前 validation/gt_events.json 统一口径。
每事件单独跑 compute_fwgs(全 GT 判断 TP), 再合并所有 forward 信号池化重算 pooled。

用法:
  python pipeline/aggregate_events.py            # 全部 GT 事件
  python pipeline/aggregate_events.py --events a,b,c
  python pipeline/aggregate_events.py --out /tmp/agg.json
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

GT_PATH = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
RES = ROOT / "results" / "v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="", help="comma-sep event_ids (default: all GT)")
    ap.add_argument("--out", default="", help="JSON 输出路径 (default: 打印)")
    ap.add_argument("--res-dir", default="", help="信号目录 (default: results/v2; DeepSeek 用 results/deepseek/gdelt)")
    ap.add_argument("--data-dir", default="", help="news jsonl 目录 (default: data/by_event; body 用 data/by_event_body)")
    args = ap.parse_args()

    res_dir = Path(args.res_dir) if args.res_dir else RES
    data_dir = Path(args.data_dir) if args.data_dir else DATA

    gt = json.load(open(GT_PATH, encoding="utf-8"))
    gt_events = gt["events"]
    only = {e.strip() for e in args.events.split(",") if e.strip()}

    rows = []
    all_signals: list[dict] = []
    all_news: dict[str, dict] = {}

    for ev in gt_events:
        eid = ev["event_id"]
        if only and eid not in only:
            continue
        sig_path = res_dir / f"{eid}_signals.jsonl"
        news_path = data_dir / f"{eid}.jsonl"
        n_art = sum(1 for _ in open(news_path, encoding="utf-8")) if news_path.exists() else 0

        if not sig_path.exists():
            rows.append({"event_id": eid, "n_articles": n_art, "n_sig": 0,
                         "tp": 0, "fp": 0, "fwgs": None, "hit": False,
                         "earliest_lead_days": None, "prec": None})
            continue

        signals = [s for s in load_signals(str(sig_path)) if s.get("status") != "error"]
        news = load_jsonl(str(news_path)) if news_path.exists() else []
        news_by_id = {a.get("id", ""): a for a in news}

        # 关键口径: 每事件只对「自身」匹配 (单事件 GT), 避免跨事件命中污染 precision。
        # 论文旧数字 (红海 TP=7/FP=1) 正是此口径; metrics.py 全 GT 匹配在事件少时未暴露。
        fw = compute_fwgs(signals, news_by_id, [ev], window_days=180)
        n_sig, tp, fp, fwgs = fw["n_signals"], fw["n_tp"], fw["n_fp"], fw["fwgs"]
        prec = (tp / (tp + fp)) if (tp + fp) else None

        val = validate(signals, [ev], window_days=180, strict=False)
        per = next((p for p in val["per_event"] if p["event_id"] == eid), None)
        hit = bool(per.get("hit")) if per else False
        lead = per.get("lead_time_days") if per else None

        rows.append({"event_id": eid, "n_articles": n_art, "n_sig": n_sig,
                     "tp": tp, "fp": fp, "fwgs": fwgs, "hit": hit,
                     "earliest_lead_days": lead, "prec": prec})

    # pooled: 各事件指标按信号数加权聚合 (pooled FWGS = Σ fwgs·n_sig / Σ n_sig)
    total_articles = sum(r["n_articles"] for r in rows)
    total_sig = sum(r["n_sig"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    pooled_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else None
    _wsum = sum((r["fwgs"] or 0.0) * r["n_sig"] for r in rows)
    pooled_fwgs = (_wsum / total_sig) if total_sig else None

    out = {
        "n_events": len(rows),
        "total_articles": total_articles,
        "total_forward_signals": total_sig,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "aggregate_precision": round(pooled_prec, 4) if pooled_prec is not None else None,
        "pooled_fwgs": round(pooled_fwgs, 4) if pooled_fwgs is not None else None,
        "per_event": rows,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印表格
    def fnum(x, nd=3):
        return "--" if x is None else f"{round(x, nd):.3f}"

    print(f"{'event':38s} {'art':>5s} {'n_sig':>5s} {'TP':>4s} {'FP':>4s} {'prec':>6s} {'FWGS':>6s} {'hit':>4s} {'lead':>5s}")
    for r in rows:
        print(f"{r['event_id']:38s} {r['n_articles']:5d} {r['n_sig']:5d} "
              f"{r['tp']:4d} {r['fp']:4d} {fnum(r['prec']):>6} {fnum(r['fwgs']):>6} "
              f"{'Y' if r['hit'] else '-':>4} "
              f"{'--' if r['earliest_lead_days'] is None else r['earliest_lead_days']:>5}")
    print("-" * 78)
    print(f"{'POOLED':38s} {total_articles:5d} {total_sig:5d} {total_tp:4d} {total_fp:4d} "
          f"{fnum(pooled_prec):>6} {fnum(pooled_fwgs):>6}")
    if args.out:
        print(f"\nJSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
