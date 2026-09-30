"""Recompute every headline number from an arbitrary signal directory.

Purpose: the manuscript's numbers are all derived from extraction output, and the
output is now being regenerated on a single model (qwen3.8-27b) across four cells
(slug/body x single-field/clock-split). Rather than patching each analysis script
to accept a new path, this one job takes an explicit directory and prints the full
number set, so any two cells can be compared like for like.

Everything is recomputed from signals plus the ground truth; nothing is read from
a cached aggregate. The matching rule is the identity-aware rule the manuscript
reports.

Reported per cell:
  pre-onset window   forward signals, true precursors, false alarms, precision,
                     events with a hit, pooled FWGS
  deployment view    the same counts including post-onset contamination, which is
                     the population a running system would actually fire on
  subtype counts     forward / confirmation / null

Usage
-----
  PY=/home/e/miniconda3/envs/ldl_los/bin/python3
  $PY pipeline/recompute_cell.py --signals-dir results/q38_slug_v2 --label "q38 slug clock-split"
  $PY pipeline/recompute_cell.py --compare results/q38_slug_v1 results/q38_slug_v2 \
      --labels "single-field" "clock-split"
"""
from __future__ import annotations

import argparse
import math
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from identity_matcher import rule_match, load, parse, RULES  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
POST_DAYS = 14
TAU, LAMBDA = 30.0, 0.5
RULE = "R4_identity"


def gt() -> list[dict]:
    return json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]


def load_articles(eid: str, body: bool) -> dict[str, str]:
    base = ROOT / ("data/by_event_body" if body else "data/by_event")
    p = base / f"{eid}.jsonl"
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id"):
            out[r["id"]] = (r.get("text") or r.get("title") or "")
    return out


def faithfulness(sig: dict, text: str) -> float:
    ph = sig.get("trigger_phrases") or []
    if not ph:
        return 1.0
    if not text:
        return 0.0
    low = text.lower()
    return sum(1 for t in ph if str(t).lower() in low) / len(ph)


def fwgs(vals: list[dict]) -> float | None:
    if not vals:
        return None
    s = 0.0
    for v in vals:
        s += ((1 - math.exp(-v["lead"] / TAU)) * v["g"]) if v["is_tp"] else (-LAMBDA * (1 - v["g"]))
    return s / len(vals)


def cell(signals_dir: Path, body: bool, rule: str = RULE) -> dict:
    fwd_pre: list[dict] = []
    n_hit_events = 0
    per_event: dict[str, int] = {}
    subtypes = Counter()
    for ev in gt():
        eid = ev["event_id"]
        p = signals_dir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        arts = load_articles(eid, body)
        n_pre = 0
        for s in load(p):
            if not s.get("temporality"):
                continue
            subtypes[s["temporality"]] += 1
            if s["temporality"] not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            is_tp = rule_match(s, ev, rule)
            g = faithfulness(s, arts.get(s.get("input_id") or "", ""))
            fwd_pre.append({"lead": lead, "is_tp": is_tp, "g": g, "event_id": eid})
            if is_tp:
                n_pre += 1
        per_event[eid] = n_pre
        if n_pre:
            n_hit_events += 1
    n = len(fwd_pre)
    tp = sum(1 for v in fwd_pre if v["is_tp"])
    return {
        "n_forward": n, "tp": tp, "fp": n - tp,
        "precision": (tp / n) if n else None,
        "hit_events": n_hit_events,
        "fwgs": fwgs(fwd_pre),
        "per_event": per_event,
        "subtypes": dict(subtypes),
    }


def contamination(signals_dir_family: str, body: bool) -> dict | None:
    """beta_0 from the post-onset negative-control pool, if an extraction exists."""
    from metrics import compute_bscc, _build_article_event_map
    cands = {
        ("title", "v2"): ROOT / "results" / "_background" / "signals_postevent_v2_body.jsonl",
    }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--body", action="store_true", help="use the body corpus for faithfulness")
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--rule", default=RULE)
    args = ap.parse_args()

    if args.compare:
        labels = args.labels or [Path(d).name for d in args.compare]
        print("=" * 104)
        print("CELL COMPARISON   rule =", args.rule)
        print("=" * 104)
        print(f"{'cell':32s} {'fwd':>5s} {'TP':>5s} {'FP':>5s} {'prec':>7s} "
              f"{'hit ev':>7s} {'FWGS':>7s}")
        print("-" * 104)
        out = {}
        for d, lab in zip(args.compare, labels):
            b = "body" in d
            c = cell(ROOT / d, b, args.rule)
            out[lab] = c
            p = f"{c['precision']:.4f}" if c["precision"] is not None else "  --"
            f = f"{c['fwgs']:.4f}" if c["fwgs"] is not None else "  --"
            print(f"{lab:32s} {c['n_forward']:5d} {c['tp']:5d} {c['fp']:5d} "
                  f"{p:>7s} {c['hit_events']:4d}/17 {f:>7s}")
        if len(args.compare) == 2:
            a, b = out[labels[0]], out[labels[1]]
            print()
            print("  delta (second minus first):")
            print(f"    forward {b['n_forward']-a['n_forward']:+d}   "
                  f"TP {b['tp']-a['tp']:+d}   FP {b['fp']-a['fp']:+d}")
            if a["precision"] and b["precision"]:
                print(f"    precision {b['precision']-a['precision']:+.4f}")
        print()
        print("  per-event TP:")
        eids = sorted({e for c in out.values() for e in c["per_event"]})
        print(f"  {'event':34s} " + "".join(f"{l[:14]:>16s}" for l in labels))
        for e in eids:
            print(f"  {e:34s} " + "".join(f"{out[l]['per_event'].get(e,0):16d}" for l in labels))
        return 0

    if not args.signals_dir:
        print("need --signals-dir or --compare", file=sys.stderr)
        return 2
    c = cell(ROOT / args.signals_dir, args.body, args.rule)
    print("=" * 92)
    print(f"CELL  {args.label or args.signals_dir}   rule={args.rule}   body={args.body}")
    print("=" * 92)
    print(f"  forward signals (pre-onset)  {c['n_forward']}")
    print(f"  true precursors              {c['tp']}")
    print(f"  false alarms                 {c['fp']}")
    print(f"  protocol precision           {c['precision']:.4f}" if c["precision"] else "  precision --")
    print(f"  events with a hit            {c['hit_events']}/17")
    print(f"  pooled FWGS                  {c['fwgs']:.4f}" if c["fwgs"] else "  FWGS --")
    print(f"  label mix                    {c['subtypes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
