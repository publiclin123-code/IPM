"""Export forward-looking signals as a human-annotation template (CSV).

Reads qwen3.6-27b clock-split (v2) signals for the 8 main-text events, keeps
forward_looking/latent signals (the protocol-precision denominator / FWGS TP
pool), and writes a CSV with one row per signal plus an empty `human_label`
column for annotators.

Columns (annotation-facing):
  signal_id, event, signal_date, temporality, confidence, description,
  trigger_phrases, commodities, companies, geographies, article_title,
  article_url, article_text, human_label

human_label is left empty; annotators fill one of:
  1 (true warning) / 0 (not a warning) / ? (uncertain)
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v2"
DATA = ROOT / "data" / "by_event"
OUT = ROOT / "data" / "annotation"
sys.path.insert(0, str(ROOT / "validation"))
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023", "us_chip_export_controls_2022",
    "renesas_earthquake_2016", "toyota_steel_explosion_2019",
    "port_los_angeles_backlog_2021", "covid_supply_disruption_2020",
    "europe_energy_crisis_2022", "renesas_naka_plant_fire_2021",
]

COLS = [
    "signal_id", "event", "signal_date", "temporality", "confidence",
    "description", "trigger_phrases", "commodities", "companies", "geographies",
    "article_title", "article_url", "article_text", "human_label",
]


def load_news(eid: str) -> dict:
    p = DATA / f"{eid}.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r.get("id", "")] = r
    return out


def load_signals(eid: str) -> list[dict]:
    p = RES / f"{eid}_signals.jsonl"
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        if s.get("status") == "error":
            continue
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        out.append(s)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for eid in EVENTS:
        news = load_news(eid)
        for s in load_signals(eid):
            art = news.get(s.get("input_id", ""), {})
            rows.append({
                "signal_id": s.get("signal_id", ""),
                "event": eid,
                "signal_date": s.get("signal_date", ""),
                "temporality": s.get("temporality", ""),
                "confidence": s.get("confidence", ""),
                "description": s.get("description", ""),
                "trigger_phrases": " | ".join(s.get("trigger_phrases", []) or []),
                "commodities": ", ".join(s.get("commodities", []) or []),
                "companies": ", ".join(
                    (c["name"] if isinstance(c, dict) else str(c))
                    for c in (s.get("companies", []) or [])),
                "geographies": ", ".join(s.get("geographies", []) or []),
                "article_title": art.get("title", ""),
                "article_url": art.get("url", ""),
                "article_text": (art.get("text", "") or "")[:2000],
                "human_label": "",
            })

    out_path = OUT / "forward_signals_annotation.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"[written] {out_path}")
    print(f"  total signals: {len(rows)}")
    # per-event count
    from collections import Counter
    c = Counter(r["event"] for r in rows)
    for eid, n in c.most_common():
        print(f"    {eid:<36} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
