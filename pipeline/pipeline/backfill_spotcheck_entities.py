#!/usr/bin/env python3
"""Spot-check v3 entity backfill + protocol-verdict stratum PPV.

1. Backfill empty commodities/geographies/companies in the canonical
   forward_spotcheck_v3.csv from results/v2/*_signals.jsonl (by signal_id),
   using the CSV's comma-string convention. Exact key join, no fuzzy match.
2. Recompute each signal's protocol verdict (strict entity rule, forward
   temporality, 180-day pre-onset window) and print stratum-conditional
   human-precision for §4.4:
     P(human=1 | protocol hit)   vs  P(human=1 | protocol miss),
   overall and by round1/new subset. Stratum rates only -- the hit/miss mix
   is fixed by the sampling design, never pool it.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path("/home/e/LF_projects/Risk_analysis")
sys.path.insert(0, str(ROOT / "validation"))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

CANON = ROOT / "data" / "annotation" / "forward_spotcheck_v3.csv"
V2 = ROOT / "results" / "v2"
GT_PATH = ROOT / "validation" / "gt_events.json"
WINDOW = 180


def wilson(tp: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = tp / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - hw, 4), round(c + hw, 4)


def load_v2() -> dict:
    m = {}
    for f in glob.glob(str(V2 / "*_signals.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d.get("signal_id"):
                m[d["signal_id"]] = d
    return m


def names(v) -> list[str]:
    """v2 entity field -> flat name list (handles str, list-of-str, role dicts)."""
    if not v:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    out = []
    for x in v:
        if isinstance(x, dict):
            x = x.get("name", "")
        if x:
            out.append(str(x).strip())
    return out


def main() -> None:
    v2 = load_v2()
    gt = {e["event_id"]: e for e in json.load(open(GT_PATH, encoding="utf-8"))["events"]}
    rows = list(csv.DictReader(open(CANON, encoding="utf-8-sig")))
    fields = list(rows[0].keys())

    filled = 0
    for r in rows:
        s = v2[r["signal_id"]]  # exact join; KeyError = data bug, fail loudly
        for col, key in (("commodities", "commodities"),
                         ("geographies", "geographies"),
                         ("companies", "companies")):
            if not (r.get(col) or "").strip():
                r[col] = ", ".join(names(s.get(key)))
                filled += 1
        # protocol verdict (strict rule, same as headline 0.873)
        ev = gt[r["event_id"]]
        sd = _parse_date(s["signal_date"])
        onset = _parse_date(ev["gt_onset_date"])
        date_ok = (onset - timedelta(days=WINDOW)) <= sd < onset
        temp_ok = s.get("temporality") in FOREWARD_TEMPORALITIES
        ent_ok = signal_matches_event(s, ev, strict=True)
        r["protocol_hit"] = "1" if (date_ok and temp_ok and ent_ok) else "0"

    if "protocol_hit" not in fields:
        fields.append("protocol_hit")
    with open(CANON, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {len(rows)} rows -> {CANON}  (entity cells backfilled: {filled}; "
          f"added column protocol_hit)")

    def stratum(sub, label):
        for hit_val, tag in (("1", "hit"), ("0", "miss")):
            g = [r for r in sub if r["protocol_hit"] == hit_val]
            if not g:
                print(f"  {label:8s} {tag:5s} n=0")
                continue
            tp = sum(1 for r in g if r["human_label"].strip() == "1")
            lo, hi = wilson(tp, len(g))
            print(f"  {label:8s} {tag:5s} n={len(g):3d}  human-true={tp:3d}  "
                  f"P(h=1|{tag})={tp/len(g):.4f}  Wilson95 [{lo},{hi}]")

    r1 = [r for r in rows if r["from_round1"].strip() == "1"]
    new = [r for r in rows if r["from_round1"].strip() == "0"]
    print("\n§4.4 stratum-conditional precision (never pool: hit/miss mix is by design):")
    stratum(rows, "ALL")
    stratum(r1, "round1")
    stratum(new, "new-68")


if __name__ == "__main__":
    main()
