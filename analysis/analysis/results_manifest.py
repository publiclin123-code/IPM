#!/usr/bin/env python3
"""Single authoritative source for every number the manuscript reports.

Why this exists
---------------
Data issue DI-4 was figure-versus-table drift: the same quantity was computed in
two places and the two disagreed.  The fix is not to check the numbers after the
fact but to have exactly one producer.  This script collects every reported
quantity into results/MANIFEST.json and a readable companion report.

It does not recompute anything it can read.  The per-cell metrics come from
recompute_cell.cell (the validated scoring path), the bootstrap intervals from
stats_canonical.py, the paired intervals from paired_v1v2.py, the confidence
diagnostics from confidence_informativeness.py, and the EDGAR aggregate from
aggregate_edgar.py.  Reimplementing any of those here would recreate DI-4.

Prerequisites
-------------
  python3 pipeline/stats_canonical.py --signals-dir results/q38_slug_v2 \\
      --bg-cells q38 --out results/stats_canonical_q38_slug_v2.json
  python3 analysis/paired_v1v2.py --a results/q38_slug_v1 --b results/q38_slug_v2 \\
      --variant title --json-out results/paired_q38_slug.json
  python3 analysis/paired_v1v2.py --a results/q38_body_v1 --b results/q38_body_v2 \\
      --variant body  --json-out results/paired_q38_body.json
  python3 analysis/confidence_informativeness.py --cells ... \\
      --json-out results/confidence_informativeness.json
  python3 pipeline/aggregate_edgar.py --res-dir results/q38_edgar \\
      --json results/q38_edgar/_aggregate.json

Usage
-----
  python3 analysis/results_manifest.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "analysis"))
from recompute_cell import cell, gt  # noqa: E402
from identity_matcher import RULES  # noqa: E402

# Cell definitions: (corpus, schema, directory, body_flag)
CELLS = [
    ("slug", "single-field", "results/q38_slug_v1", False),
    ("slug", "clock-split", "results/q38_slug_v2", False),
    ("body", "single-field", "results/q38_body_v1", True),
    ("body", "clock-split", "results/q38_body_v2", True),
]

MUST_EXIST = {
    "stats_canonical": "results/stats_canonical_q38_slug_v2.json",
    "paired_slug": "results/paired_q38_slug.json",
    "paired_body": "results/paired_q38_body.json",
    "confidence": "results/confidence_informativeness.json",
    "edgar": "results/q38_edgar/_aggregate.json",
}


def read_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    man: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "qwen3.8-27b (Qwen3.8-27B-NVFP4-MTP-Q8attn.gguf)",
        "rule": "R4_identity",
        "events_in_ground_truth": len(gt()),
        "note": ("one ground-truth event, suez_ever_given_2021, has no pre-onset "
                 "corpus because its query keywords exist only post-onset; it is "
                 "therefore absent from every cell and all rates are over 16 "
                 "evaluable events"),
    }

    # ---- missing-artifact report, up front ---------------------------------
    missing = [k for k, v in MUST_EXIST.items() if not (ROOT / v).exists()]
    if missing:
        print("MISSING ARTIFACTS -- run the prerequisite commands first:")
        for k in missing:
            print(f"  {k:16s} {MUST_EXIST[k]}")
        print()
        print("The manifest will contain only the cells it can compute.")

    # ---- per-cell metrics and the matching-rule ladder ----------------------
    cells: dict = {}
    ladder: dict = {}
    for corpus, schema, d, body in CELLS:
        key = f"{corpus}_{schema.replace('-', '_')}"
        base = cell(ROOT / d, body)
        cells[key] = {
            "corpus": corpus, "schema": schema, "dir": d,
            "n_forward": base["n_forward"], "tp": base["tp"], "fp": base["fp"],
            "precision": base["precision"], "hit_events": base["hit_events"],
            "fwgs": base["fwgs"], "per_event_tp": base["per_event"],
        }
        ladder[key] = {}
        for r in RULES:
            c = cell(ROOT / d, body, rule=r)
            ladder[key][r] = {"tp": c["tp"], "precision": c["precision"],
                              "hit_events": c["hit_events"], "fwgs": c["fwgs"]}
        print(f"cell {key:22s} fwd {base['n_forward']:5d}  TP {base['tp']:5d}  "
              f"FP {base['fp']:4d}  prec {base['precision']:.4f}  "
              f"hit {base['hit_events']}/{len(gt())}  FWGS {base['fwgs']:.4f}")
    man["cells"] = cells
    man["match_ladder"] = ladder

    # ---- tolerances-free verbatim copies of the other producers -------------
    for name, rel in MUST_EXIST.items():
        data = read_json(rel)
        man[name] = data
        if data is not None:
            print(f"read {rel}")

    # ---- derived headline statements the paper makes ------------------------
    s1 = man["cells"]["slug_single_field"]
    s2 = man["cells"]["slug_clock_split"]
    b1 = man["cells"]["body_single_field"]
    b2 = man["cells"]["body_clock_split"]
    man["derived"] = {
        "slug_precision_gain": s2["precision"] - s1["precision"],
        "body_precision_gain": b2["precision"] - b1["precision"],
        "slug_fp_reduction": s1["fp"] - s2["fp"],
        "body_fp_reduction": b1["fp"] - b2["fp"],
        "slug_tp_cost": s1["tp"] - s2["tp"],
        "body_tp_cost": b1["tp"] - b2["tp"],
        "slug_tp_cost_relative": (s1["tp"] - s2["tp"]) / s1["tp"],
        "body_tp_cost_relative": (b1["tp"] - b2["tp"]) / b1["tp"],
    }
    d = man["derived"]
    print()
    print("derived")
    print(f"  slug  precision {s1['precision']:.4f} -> {s2['precision']:.4f} "
          f"({d['slug_precision_gain']:+.4f});  FP {s1['fp']} -> {s2['fp']} "
          f"(-{d['slug_fp_reduction']});  TP {s1['tp']} -> {s2['tp']} "
          f"(-{d['slug_tp_cost']}, -{100*d['slug_tp_cost_relative']:.1f}%)")
    print(f"  body  precision {b1['precision']:.4f} -> {b2['precision']:.4f} "
          f"({d['body_precision_gain']:+.4f});  FP {b1['fp']} -> {b2['fp']} "
          f"(-{d['body_fp_reduction']});  TP {b1['tp']} -> {b2['tp']} "
          f"(-{d['body_tp_cost']}, -{100*d['body_tp_cost_relative']:.1f}%)")

    # ---- consistency assertions: catch drift instead of documenting it ------
    errs: list[str] = []
    for key, c in cells.items():
        if c["tp"] + c["fp"] != c["n_forward"]:
            errs.append(f"{key}: tp+fp != n_forward")
        if abs(c["precision"] - c["tp"] / c["n_forward"]) > 1e-12:
            errs.append(f"{key}: precision != tp/n_forward")
    if man.get("paired_slug") and man["paired_slug"]["point"]["tpB"] != s2["tp"]:
        errs.append("paired_slug tpB disagrees with the slug clock-split cell")
    if man.get("paired_body") and man["paired_body"]["point"]["tpB"] != b2["tp"]:
        errs.append("paired_body tpB disagrees with the body clock-split cell")
    if man.get("stats_canonical"):
        pe = man["stats_canonical"]["point"]
        if pe["tp"] != s2["tp"] or pe["n"] != s2["n_forward"]:
            errs.append("stats_canonical point estimates disagree with slug clock-split cell")
    man["consistency_errors"] = errs
    print()
    if errs:
        print("CONSISTENCY ERRORS (the producers disagree):")
        for e in errs:
            print(f"  !! {e}")
    else:
        print("consistency: all producers agree")

    out = ROOT / "results" / "MANIFEST.json"
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
