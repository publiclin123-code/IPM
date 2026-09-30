"""Re-stratify the human spot-check under the identity matching rule.

Why
---
The spot-check in Section 4 was stratified using the old loose/strict matcher:
66 "strict entity match" hits, of which 59 were human-confirmed. The paper now
reports the identity-aware rule (R4) and precision 0.765, so the spot-check
stratification and the reported rule no longer correspond, and the paragraph
quoting "automatic strict-rule precision 0.582" is stale.

The human labels themselves do not change; only the stratum to which each signal
belongs does. Because the released gold file carries each signal's entities
(companies, geographies, commodities), the identity rule can be reapplied to it
directly.

This gives, for the reported rule:
  - stratum precision among R4 hits  (the P(human-genuine | automatic hit) figure)
  - stratum precision among R4 misses
  - the pooled human-validated precision

Usage
-----
  /home/e/miniconda3/envs/ldl_los/bin/python3 pipeline/restratisfy_spotcheck.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from identity_matcher import rule_match, IDENTITY  # noqa: E402

GOLD = ROOT / "draft" / "forward_spotcheck_v2_round2_gold.csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def parse_entities(cell: str) -> list:
    """The CSV stores entities as a pipe- or comma-joined string."""
    if not cell:
        return []
    cell = cell.strip()
    if cell.startswith("["):
        try:
            return json.loads(cell.replace("'", '"'))
        except Exception:
            pass
    return [p.strip() for p in cell.replace("|", ",").split(",") if p.strip()]


def main() -> int:
    if not GOLD.exists():
        print(f"missing {GOLD}")
        return 1
    rows = list(csv.DictReader(GOLD.open(encoding="utf-8")))
    print(f"gold file: {GOLD.name}   rows = {len(rows)}")
    print(f"columns: {list(rows[0].keys())[:20]}")
    print()

    gt_by_event = {}
    for e in json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]:
        gt_by_event[e["event_id"]] = e

    recomputed = []
    skipped = 0
    for r in rows:
        eid = r.get("event_id", "").strip()
        ev = gt_by_event.get(eid)
        if not ev:
            skipped += 1
            continue
        # rebuild a signal record in the shape the matcher expects
        comps = parse_entities(r.get("companies", ""))
        sig = {
            "companies": [{"name": c} if isinstance(c, str) else c for c in comps],
            "geographies": parse_entities(r.get("geographies", "")),
            "commodities": parse_entities(r.get("commodities", "")),
        }
        try:
            hit = rule_match(sig, ev, "R4_identity")
        except Exception:
            skipped += 1
            continue
        gold = (r.get("gold_label") or "").strip().lower()
        if gold not in ("1", "0", "yes", "no", "true", "false"):
            continue
        genuine = gold in ("1", "yes", "true")
        recomputed.append({"event_id": eid, "hit": hit, "genuine": genuine,
                           "old_proto": (r.get("proto_hit") or "").strip()})

    n = len(recomputed)
    hits = [x for x in recomputed if x["hit"]]
    misses = [x for x in recomputed if not x["hit"]]
    print(f"signals with a usable gold label and recomputable entities: {n}")
    if skipped:
        print(f"  skipped (no ground-truth event or unparsable entities): {skipped}")
    print()

    print("=" * 92)
    print("SPOT-CHECK RE-STRATIFIED UNDER THE IDENTITY RULE (R4)")
    print("=" * 92)
    for label, grp in (("R4 hits", hits), ("R4 misses", misses)):
        k = sum(1 for x in grp if x["genuine"])
        m = len(grp)
        lo, hi = wilson(k, m)
        p = f"{k/m:.3f}" if m else "  --"
        ci = f"[{lo:.3f},{hi:.3f}]" if m else ""
        print(f"  {label:10s} n={m:4d}  human-genuine={k:4d}  precision={p} "
              f"{ci}")
    k = sum(1 for x in recomputed if x["genuine"])
    lo, hi = wilson(k, n)
    print(f"  {'pooled':10s} n={n:4d}  human-genuine={k:4d}  precision={k/n:.3f} "
          f"[{lo:.3f},{hi:.3f}]")
    print()

    print("=" * 92)
    print("CROSS-TAB: old proto_hit (stored) vs R4 (recomputed)")
    print("=" * 92)
    agree = sum(1 for x in recomputed if (x["old_proto"].lower() in ("1", "true", "yes")) == x["hit"])
    print(f"  agree on {agree}/{n} = {agree/n:.1%}")
    moved_in = sum(1 for x in recomputed
                   if x["old_proto"].lower() in ("1", "true", "yes") and not x["hit"])
    moved_out = sum(1 for x in recomputed
                    if x["old_proto"].lower() not in ("1", "true", "yes") and x["hit"])
    print(f"  was a hit under the old rule, is not under R4: {moved_in}")
    print(f"  was not a hit under the old rule, is under R4: {moved_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
