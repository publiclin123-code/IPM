#!/usr/bin/env python3
"""Cross-model robustness summary.

The paper's cross-model claim is that four models spanning three open-weight
families and one hosted API produce pre-onset precision between 0.67 and 0.81,
and that all four rank the events the same way by type. Checking that claim needs
each model's per-event and pooled numbers, not every individual extracted signal.

This script reduces the per-model extraction directories to those numbers and
writes `results/cross_model_summary.json`. The raw directories it reads are large
(one JSONL per event per model) and are NOT shipped in the release repository;
this summary is shipped instead, so every cross-model figure and sentence remains
checkable without redistributing the full extractions.

Run it only when the raw extraction directories are present:

  python3 analysis/cross_model_summary.py                 # writes the summary
  python3 analysis/cross_model_summary.py --check         # verify against the
                                                          # committed summary

Scoring is the paper's own: pre-onset window, identity-aware match (rule
R4_identity), clock-split ontology, via pipeline/recompute_cell.py, so these
numbers are computed the same way as every other precision in the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))

from recompute_cell import cell  # noqa: E402

RULE = "R4_identity"
OUT = ROOT / "results" / "cross_model_summary.json"

# The four runs the cross-model claim is built from, plus the superseded
# qwen3.6 pass and the full-text qwen3.8 pass for completeness.
MODELS = [
    ("qwen3.8-27b",        "results/q38_slug_v2",    False,
     "main model, slug input (the paper's headline cell)"),
    ("qwen3.8-27b-body",   "results/q38_body_v2",    True,
     "main model, full-text input"),
    ("gemma4-31b",         "results/v2/gemma4",      False,
     "cross-model robustness"),
    ("muse-glimmer-30b",   "results/v2/muse",        False,
     "cross-model robustness"),
    ("deepseek-v4-flash",  "results/deepseek/gdelt", False,
     "cross-model robustness"),
    ("qwen3.6-27b",        "results/v2",             False,
     "SUPERSEDED first pass; the paper states it is not used"),
]

# The body flag selects which corpus the faithfulness term g_i is computed
# against: data/by_event_body for the full-text runs, data/by_event otherwise.
# Getting it wrong silently yields a negative pooled FWGS, because every
# trigger_phrases lookup misses and g_i collapses to 0.


def collect() -> dict:
    models: dict = {}
    missing: list[str] = []
    for name, rel, body, note in MODELS:
        d = ROOT / rel
        if not d.exists():
            missing.append(rel)
            continue
        c = cell(d, body, RULE)
        models[name] = {
            "dir": rel,
            "note": note,
            "rule": RULE,
            "n_forward_pre_onset": c["n_forward"],
            "tp": c["tp"],
            "fp": c["fp"],
            "precision": c["precision"],
            "hit_events": c["hit_events"],
            "pooled_fwgs": c["fwgs"],
            "per_event_tp": c["per_event"],
        }

    # The four models the precision range is quoted over.
    quoted = ["qwen3.8-27b", "gemma4-31b", "muse-glimmer-30b", "deepseek-v4-flash"]
    precs = [models[m]["precision"] for m in quoted if m in models]
    return {
        "rule": RULE,
        "scoring": ("pre-onset window, clock-split ontology, identity-aware match; "
                    "computed by pipeline/recompute_cell.py"),
        "note": ("Raw per-event extraction directories are not distributed in the "
                 "release repository; this summary carries the numbers the "
                 "cross-model figures and claims are built from. Regenerate with "
                 "analysis/cross_model_summary.py when the raw dirs are present."),
        "models": models,
        "quoted_range": {
            "models": quoted,
            "min": min(precs) if precs else None,
            "max": max(precs) if precs else None,
        },
        "missing_dirs": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="cross-model robustness summary")
    ap.add_argument("--check", action="store_true",
                    help="compare a fresh computation against the committed summary")
    args = ap.parse_args()

    fresh = collect()

    if args.check:
        if not OUT.exists():
            print(f"no committed summary at {OUT.relative_to(ROOT)}")
            return 1
        old = json.loads(OUT.read_text(encoding="utf-8"))
        bad = 0
        for name, m in old.get("models", {}).items():
            f = fresh["models"].get(name)
            if f is None:
                print(f"  [skip] {name}: raw dir absent, cannot recompute")
                continue
            for k in ("n_forward_pre_onset", "tp", "fp", "precision", "pooled_fwgs"):
                if abs((m[k] or 0) - (f[k] or 0)) > 1e-9:
                    print(f"  [DIFF] {name}.{k}: committed {m[k]} vs fresh {f[k]}")
                    bad += 1
        print("  summary matches" if not bad else f"  {bad} differences")
        return 1 if bad else 0

    if fresh["missing_dirs"]:
        print("note: these raw dirs are absent, skipped:", ", ".join(fresh["missing_dirs"]))
    OUT.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for name, m in fresh["models"].items():
        print(f"  {name:20s} n={m['n_forward_pre_onset']:5d} TP={m['tp']:5d} "
              f"FP={m['fp']:5d} prec={m['precision']:.4f} FWGS={m['pooled_fwgs']:.4f}")
    q = fresh["quoted_range"]
    print(f"  quoted range over {len(q['models'])} models: "
          f"{q['min']:.4f} to {q['max']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
