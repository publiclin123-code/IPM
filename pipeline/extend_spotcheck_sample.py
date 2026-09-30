#!/usr/bin/env python3
"""Extend the §4.4 forward-signal spot-check from 52 to ~130 rows.

Keeps the 52 already-adjudicated signals (labels copied from
forward_spotcheck_v2_adjudication.xlsx), then adds a fresh stratified draw of
forward signals from the v2 pool (per-event max raised 5 -> 10, mixing TP/FP),
writing forward_spotcheck_v3.csv with empty human labels for the new rows.

The v2 forward pool has 237 signals; 52 are already labeled, leaving 185.
Target total ~130 keeps manual load bounded while roughly doubling n for the
Wilson interval (n=130 at ~0.94 halves the half-width vs n=51).
"""
import csv, json, random, sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals, FOREWARD_TEMPORALITIES, signal_matches_event, _parse_date  # noqa: E402
from metrics import load_jsonl  # noqa: E402

GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
RES = ROOT / "results" / "v2"
XLSX = ROOT / "data" / "annotation" / "forward_spotcheck_v2_adjudication.xlsx"
OUT = ROOT / "data" / "annotation" / "forward_spotcheck_v3.csv"
SEED = 7
PER_EVENT_MAX = 10
TP_PER_EVENT = 6   # 每事件 TP 最多抽 6、FP 最多抽 4

FIELDS = ["event_id", "signal_id", "signal_date", "temporality", "confidence",
          "description", "trigger_phrases", "commodities", "companies",
          "geographies", "article_title", "article_url", "article_text",
          "human_label", "from_round1"]


def load_existing() -> dict[str, dict]:
    """signal_id -> {row fields} from the adjudicated round-1 xlsx."""
    wb = openpyxl.load_workbook(XLSX)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    cols = [c.value for c in ws[1]]
    idx = {c: i for i, c in enumerate(cols)}
    out = {}
    for r in rows:
        sid = r[idx["signal_id"]]
        if sid is None:
            continue
        out[sid] = {
            "event_id": r[idx["event_id"]],
            "signal_id": sid,
            "signal_date": str(r[idx["signal_date"]])[:10] if r[idx["signal_date"]] else "",
            "temporality": r[idx["temporality"]],
            "confidence": r[idx["confidence"]],
            "description": r[idx["description"]] or "",
            "trigger_phrases": r[idx["trigger_phrases"]] or "",
            "article_title": r[idx["article_title"]] or "",
            "article_url": r[idx["article_url"]] or "",
            # adjudicated label: A wins all 4 disagreements; the one -1 in B is A=1
            "human_label": str(r[idx["label_A"]]),
            "from_round1": "1",
        }
    return out


def main():
    random.seed(SEED)
    existing = load_existing()
    print(f"round-1 adjudicated: {len(existing)}")

    gt = json.load(open(GT, encoding="utf-8"))["events"]
    sampled: dict[str, dict] = dict(existing)

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

        def is_tp(s):
            try:
                sd = _parse_date(s["signal_date"])
            except (KeyError, ValueError):
                return False
            onset = _parse_date(ev["gt_onset_date"])
            return 0 < (onset - sd).days <= 180 and signal_matches_event(s, ev, strict=False)

        tps = [s for s in fwd if is_tp(s)]
        fps = [s for s in fwd if not is_tp(s)]
        have = sum(1 for s in sampled.values() if s["event_id"] == eid)

        # 已有行优先保留; 补抽到 PER_EVENT_MAX
        budget = PER_EVENT_MAX - have
        if budget <= 0:
            continue
        # 排除已抽
        have_ids = {s["signal_id"] for s in sampled.values() if s["event_id"] == eid}
        tps = [s for s in tps if s.get("signal_id") not in have_ids]
        fps = [s for s in fps if s.get("signal_id") not in have_ids]
        random.shuffle(tps); random.shuffle(fps)
        pick = tps[:min(TP_PER_EVENT, len(tps))] + fps[:min(PER_EVENT_MAX - TP_PER_EVENT, len(fps))]
        pick = pick[:budget]
        for s in pick:
            art = news.get(s.get("input_id", ""), {})
            sid = s.get("signal_id", "")
            sampled[sid] = {
                "event_id": eid,
                "signal_id": sid,
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
                "from_round1": "0",
            }

    rows = list(sampled.values())
    rows.sort(key=lambda r: (r["event_id"], r["from_round1"], r["signal_id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    ev_cnt = Counter(r["event_id"] for r in rows)
    new = sum(1 for r in rows if r["from_round1"] == "0")
    print(f"扩容样本: {len(rows)} 条 (round1 保留 {len(existing)}, 新增 {new}) -> {OUT}")
    print(f"覆盖事件: {len(ev_cnt)}")
    for eid, n in sorted(ev_cnt.items()):
        n1 = sum(1 for r in rows if r["event_id"] == eid and r["from_round1"] == "1")
        print(f"  {eid}: {n} (round1 {n1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
