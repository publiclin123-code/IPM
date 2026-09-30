"""Round-2 blind re-annotation template for the 52-signal spot check.

Copies data/annotation/forward_spotcheck_v2.csv (the original v2 spot-check
sample, 52 rows, same signal set as round 1), clears human_label, and adds a
zero_reason column:

  unrelated        — article not about this event's supply-chain domain
  already_occurred — the warned disruption had already happened at report time

Filled only when human_label = 0; blank otherwise.

Output: data/annotation/forward_spotcheck_v2_round2.csv
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "annotation" / "forward_spotcheck_v2.csv"
OUT = ROOT / "data" / "annotation" / "forward_spotcheck_v2_round2.csv"


def main() -> int:
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows, "source template is empty"
    for r in rows:
        r["human_label"] = ""
        r["zero_reason"] = ""
    cols = [c for c in rows[0].keys() if c != "zero_reason"] + ["zero_reason"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[written] {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
