#!/usr/bin/env python3
"""Generate a stratified spot-check annotation sample from the v2 forward signals.

Samples ~50 forward signals across all 18 events (stratified by event, mixing
protocol-hit (TP) and false-alarm (FP) signals), writes a CSV ready for human
semantic spot-checking. Replaces the stale v1-era 105-row annotation file.

Output: data/annotation/forward_spotcheck_v2.csv
"""
from __future__ import annotations
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals, FOREWARD_TEMPORALITIES, signal_matches_event, _parse_date  # noqa: E402
from metrics import load_jsonl  # noqa: E402

GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
RES = ROOT / "results" / "v2"
OUT = ROOT / "data" / "annotation" / "forward_spotcheck_v2.csv"
SEED = 42
TARGET_TOTAL = 60
PER_EVENT_MAX = 5  # 每事件最多抽 5 条（小事件全抽）


def main():
    random.seed(SEED)
    gt = json.load(open(GT, encoding="utf-8"))["events"]
    gt_by_id = {e["event_id"]: e for e in gt}

    sampled = []
    for ev in gt:
        eid = ev["event_id"]
        sp = RES / f"{eid}_signals.jsonl"
        if not sp.exists():
            continue
        sigs = load_signals(str(sp))
        news = {a["id"]: a for a in (load_jsonl(str(DATA / f"{eid}.jsonl"))
                                     if (DATA / f"{eid}.jsonl").exists() else [])}
        fwd = [s for s in sigs if s.get("temporality") in FOREWARD_TEMPORALITIES]
        if not fwd:
            continue
        # 分 TP / FP 两组（分层用于保证样本代表性，不暴露给标注者）
        def is_tp(s):
            try:
                sd = _parse_date(s["signal_date"])
            except (KeyError, ValueError):
                return False
            onset = _parse_date(ev["gt_onset_date"])
            return 0 < (onset - sd).days <= 180 and signal_matches_event(s, ev, strict=False)
        tps = [s for s in fwd if is_tp(s)]
        fps = [s for s in fwd if not is_tp(s)]
        # 优先混合：每条 TP 抽 3、FP 抽 2；小事件全抽
        pick = []
        if tps:
            random.shuffle(tps)
            pick += tps[:min(3, len(tps))]
        if fps:
            random.shuffle(fps)
            pick += fps[:min(2, len(fps))]
        pick = pick[:PER_EVENT_MAX]
        for s in pick:
            art = news.get(s.get("input_id", ""), {})
            sampled.append({
                "event_id": eid,
                "signal_id": s.get("signal_id", ""),
                "signal_date": s.get("signal_date", ""),
                "temporality": s.get("temporality", ""),
                "confidence": s.get("confidence", ""),
                "description": s.get("description", ""),
                "trigger_phrases": " | ".join(s.get("trigger_phrases", [])),
                "commodities": ", ".join(s.get("commodities", [])),
                "companies": ", ".join(c.get("name", "") for c in s.get("companies", [])),
                "geographies": ", ".join(s.get("geographies", [])),
                "article_title": art.get("title", ""),
                "article_url": art.get("url", ""),
                "article_text": (art.get("text", "") or "")[:2000],
                "human_label": "",
            })

    # 如果超 50，随机裁到 50；否则全保留
    if len(sampled) > TARGET_TOTAL:
        random.shuffle(sampled)
        sampled = sampled[:TARGET_TOTAL]

    fields = ["event_id", "signal_id", "signal_date", "temporality", "confidence",
              "description", "trigger_phrases", "commodities",
              "companies", "geographies", "article_title", "article_url",
              "article_text", "human_label"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sampled:
            w.writerow(r)

    # 统计
    from collections import Counter
    ev_cnt = Counter(r["event_id"] for r in sampled)
    print(f"抽检样本: {len(sampled)} 条 → {OUT}")
    print(f"覆盖事件数: {len(ev_cnt)}")
    for eid, n in sorted(ev_cnt.items()):
        print(f"  {eid}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
