#!/usr/bin/env python3
"""Generate a stratified retrieval-audit annotation sample (paper Sec 4.3).

This is a DIFFERENT task from gen_annotation_sample.py (extraction-layer
signal spot-check, Sec 4.4). Here the unit is an ARTICLE (retrieval layer),
not an extracted SIGNAL.

Paper Sec 4.3 "Precision audit": from the 2,555-article keyword pool we draw a
stratified sample of up to 25 articles per event (at most 450 items) and
hand-label each as relevant or irrelevant to the event, yielding per-event
precision estimates P_hat_e with Wilson 95% intervals.

Input:  validation/gt_events.json  (18 ground-truth events, drives the sample)
        data/by_event/{event_id}.jsonl (per-event keyword-filtered article pool)
Output: data/annotation/retrieval_audit_sample.csv

The annotator fills only the `relevant_label` column (1 = relevant, 0 =
irrelevant, ? = unsure) and optionally `note`. Everything else is read-only.
"""
from __future__ import annotations
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from metrics import load_jsonl  # noqa: E402

GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
OUT = ROOT / "data" / "annotation" / "retrieval_audit_sample.csv"

SEED = 42
PER_EVENT_MAX = 25  # paper Sec 4.3: "up to 25 articles per event"
TEXT_CAP = 2000     # article body preview cap, matches spot-check convention


def main() -> int:
    random.seed(SEED)
    gt = json.load(open(GT, encoding="utf-8"))["events"]

    sampled: list[dict] = []
    n_events_with_articles = 0
    for ev in gt:
        eid = ev["event_id"]
        arts_path = DATA / f"{eid}.jsonl"
        if not arts_path.exists():
            continue
        arts = load_jsonl(str(arts_path))
        if not arts:
            continue
        n_events_with_articles += 1
        # Per-event independent uniform draw, capped at 25 (stratified by event).
        if len(arts) > PER_EVENT_MAX:
            arts = random.sample(arts, PER_EVENT_MAX)
        for a in arts:
            sampled.append({
                "event_id": eid,
                "event_name": ev.get("event_name", ""),
                "article_id": a.get("id", ""),
                "article_date": a.get("date", ""),
                "article_title": a.get("title", ""),
                "article_url": a.get("url", ""),
                "article_text": (a.get("text", "") or "")[:TEXT_CAP],
                "relevant_label": "",  # annotator: 1 / 0 / ?
                "note": "",            # annotator: optional
            })

    fields = ["event_id", "event_name", "article_id", "article_date",
              "article_title", "article_url", "article_text",
              "relevant_label", "note"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sampled:
            w.writerow(r)

    ev_cnt = Counter(r["event_id"] for r in sampled)
    print(f"检索审计样本: {len(sampled)} 篇 → {OUT}")
    print(f"有文章的事件: {n_events_with_articles} / 共 {len(gt)} 个 GT 事件")
    for eid, n in sorted(ev_cnt.items()):
        print(f"  {eid}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
