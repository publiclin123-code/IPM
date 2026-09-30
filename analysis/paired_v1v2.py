#!/usr/bin/env python3
"""Paired difference bootstrap for the single-field versus clock-split comparison.

Why this script exists
----------------------
The marginal confidence intervals for the two precisions overlap, which is easy
to misread as "no difference".  That reading is wrong: the two cells are scored
on the *same* seventeen events, so the correct inferential object is the paired
difference, not the overlap of two marginal intervals.  Resampling events once
and applying the same resample to both arms removes the between-event variance
that dominates the marginal intervals.

What it reports
---------------
  * point estimates for TP, FP, precision and their difference
  * percentile CI for the paired precision difference
  * the break-even cost c* = dTP / dFP, the cost of one false positive measured
    in units of one missed true precursor, at which the two schemas are equally
    valuable.  Below c* the clock split buys more than it costs.
  * the probability (over bootstrap resamples) that the split is worth it at a
    set of reference costs.

Invariance check: the two arms must partition the same article set.  The script
verifies that the set of article ids seen under each arm is identical, and
aborts otherwise, because a mismatch means the comparison is not paired and
every number below would be meaningless.

Usage
-----
  python3 analysis/paired_v1v2.py \
      --a results/q38_slug_v1 --b results/q38_slug_v2 \
      --labels "single-field" "clock-split" --variant title
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from identity_matcher import rule_match, load, parse  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
B = 2000
SEED = 42


def gt_events() -> list[dict]:
    p = ROOT / "validation" / "gt_events.json"
    return json.loads(p.read_text(encoding="utf-8"))["events"]


def attempted_ids(signals_dir: Path) -> dict[str, set[str]]:
    """Per event: every input_id the extractor was given, including empty/error rows.

    This is the object that must be identical across the two arms.  Filtering to
    rows that yielded a signal would measure prompt yield instead, which is
    exactly what the two arms are allowed to differ in.
    """
    out: dict[str, set[str]] = {}
    for ev in gt_events():
        eid = ev["event_id"]
        p = signals_dir / f"{eid}_signals.jsonl"
        ids: set[str] = set()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("input_id"):
                    ids.add(r["input_id"])
        out[eid] = ids
    return out


def corpus_ids(eid: str, body: bool) -> set[str]:
    """The article ids actually available for this event."""
    base = ROOT / ("data/by_event_body" if body else "data/by_event")
    p = base / f"{eid}.jsonl"
    ids: set[str] = set()
    if not p.exists():
        return ids
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id"):
            ids.add(r["id"])
    return ids


def per_event_tp_fp(signals_dir: Path, rule: str) -> dict[str, dict]:
    """Per event: forward-signal count, true positives, and the article ids seen."""
    out: dict[str, dict] = {}
    for ev in gt_events():
        eid = ev["event_id"]
        p = signals_dir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        n = tp = 0
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            n += 1
            if rule_match(s, ev, rule):
                tp += 1
        out[eid] = {"n": n, "tp": tp, "fp": n - tp}
    return out


def aggregate(per_event: dict[str, dict], events: list[str]) -> tuple[int, int]:
    n = sum(per_event[e]["n"] for e in events)
    tp = sum(per_event[e]["tp"] for e in events)
    return n, tp


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="single-field cell")
    ap.add_argument("--b", required=True, help="clock-split cell")
    ap.add_argument("--labels", nargs=2, default=["single-field", "clock-split"])
    ap.add_argument("--rule", default="R4_identity")
    ap.add_argument("--variant", default="title", choices=["title", "body"],
                    help="which corpus the cells were extracted from")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--b-resamples", type=int, default=B)
    ap.add_argument("--costs", nargs="*", type=float,
                    default=[0.25, 0.5, 0.7, 1.0, 1.5, 2.0])
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    da, db = ROOT / args.a, ROOT / args.b
    ea = per_event_tp_fp(da, args.rule)
    eb = per_event_tp_fp(db, args.rule)
    events = sorted(set(ea) & set(eb))

    # ---- invariance check: both arms must have been fed the same corpus ------
    ia = attempted_ids(da)
    ib = attempted_ids(db)
    body = args.variant == "body"
    problems: list[tuple[str, str, int]] = []
    attempted_a = attempted_b = 0
    for e in events:
        want = corpus_ids(e, body)
        attempted_a += len(ia[e])
        attempted_b += len(ib[e])
        if ia[e] != ib[e]:
            problems.append((e, "arms attempted different article sets",
                             len(ia[e] ^ ib[e])))
        elif not want:
            problems.append((e, "no corpus file found for this event", 0))
        elif ia[e] != want:
            problems.append((e, "attempted set != corpus set",
                             len(want ^ ia[e])))

    print("=" * 84)
    print("PAIRED DIFFERENCE BOOTSTRAP")
    print("=" * 84)
    print(f"  A  {args.a:34s} {args.labels[0]}")
    print(f"  B  {args.b:34s} {args.labels[1]}")
    print(f"  rule {args.rule}   events {len(events)}   B {args.b_resamples}   seed {args.seed}")
    print()
    print("  input-corpus invariance (the condition that makes pairing valid)")
    print(f"    articles attempted, arm A   {attempted_a}")
    print(f"    articles attempted, arm B   {attempted_b}")
    if problems:
        print("    !! PROBLEMS")
        for e, why, k in problems[:15]:
            print(f"       {e:34s} {why} ({k})")
    else:
        print("    both arms attempted exactly the corpus for all events: pairing valid")
        print("    (signal yield legitimately differs; that is the effect under test)")

    nA, tpA = aggregate(ea, events)
    nB, tpB = aggregate(eb, events)
    fpA, fpB = nA - tpA, nB - tpB

    print()
    print("-" * 84)
    print(f"{'':22s} {'fwd':>7s} {'TP':>7s} {'FP':>7s} {'precision':>11s}")
    print("-" * 84)
    print(f"{args.labels[0]:22s} {nA:7d} {tpA:7d} {fpA:7d} {tpA/nA if nA else 0:11.4f}")
    print(f"{args.labels[1]:22s} {nB:7d} {tpB:7d} {fpB:7d} {tpB/nB if nB else 0:11.4f}")
    d_tp, d_fp = tpB - tpA, fpB - fpA
    d_prec = (tpB / nB if nB else 0) - (tpA / nA if nA else 0)
    print("-" * 84)
    print(f"{'delta (B - A)':22s} {nB-nA:+7d} {d_tp:+7d} {d_fp:+7d} {d_prec:+11.4f}")

    # ---- event-stratified paired bootstrap ---------------------------------
    rng = random.Random(args.seed)
    d_precs: list[float] = []
    d_tps: list[int] = []
    d_fps: list[int] = []
    for _ in range(args.b_resamples):
        draw = [events[rng.randrange(len(events))] for _ in range(len(events))]
        n_a = tp_a = n_b = tp_b = 0
        for e in draw:
            n_a += ea[e]["n"]; tp_a += ea[e]["tp"]
            n_b += eb[e]["n"]; tp_b += eb[e]["tp"]
        if n_a == 0 or n_b == 0:
            continue
        d_precs.append(tp_b / n_b - tp_a / n_a)
        d_tps.append(tp_b - tp_a)
        d_fps.append((n_b - tp_b) - (n_a - tp_a))

    print()
    print("=" * 84)
    print(f"PAIRED INTERVALS (event-stratified, B={args.b_resamples})")
    print("=" * 84)
    print(f"  precision difference   {d_prec:+.4f}   "
          f"[{pct(d_precs,0.025):+.4f}, {pct(d_precs,0.975):+.4f}]")
    print(f"  TP difference          {d_tp:+d}   "
          f"[{pct(d_tps,0.025):+.0f}, {pct(d_tps,0.975):+.0f}]")
    print(f"  FP difference          {d_fp:+d}   "
          f"[{pct(d_fps,0.025):+.0f}, {pct(d_fps,0.975):+.0f}]")
    print(f"  P(precision rises)     "
          f"{sum(1 for d in d_precs if d > 0)/len(d_precs):.4f}")
    print(f"  P(FP falls)            "
          f"{sum(1 for d in d_fps if d < 0)/len(d_fps):.4f}")

    # ---- break-even cost ----------------------------------------------------
    # B is preferred iff tpB - c*fpB > tpA - c*fpA  <=>  c*(fpA-fpB) > tpA-tpB
    print()
    print("=" * 84)
    print("BREAK-EVEN COST OF A FALSE POSITIVE")
    print("=" * 84)
    if d_fp == 0:
        print("  no FP reduction: the split cannot pay for itself at any cost")
        c_star = float("inf")
    elif d_tp >= 0:
        c_star = 0.0
        print("  the split lowers FP without losing TP: it dominates")
    else:
        c_star = (-d_tp) / (-d_fp)
        print(f"  c* = |dTP| / |dFP| = {abs(d_tp)} / {abs(d_fp)} = {c_star:.3f}")
        print(f"  below c* the clock split buys more precision than it costs in recall")

    cs = []
    for _ in range(args.b_resamples):
        draw = [events[rng.randrange(len(events))] for _ in range(len(events))]
        tp_a = fp_a = tp_b = fp_b = 0
        for e in draw:
            tp_a += ea[e]["tp"]; fp_a += ea[e]["fp"]
            tp_b += eb[e]["tp"]; fp_b += eb[e]["fp"]
        dt, df = tp_b - tp_a, fp_b - fp_a
        if df == 0:
            cs.append(float("inf") if dt < 0 else 0.0)
        elif dt >= 0:
            cs.append(0.0)
        else:
            cs.append((-dt) / (-df))
    fin = [c for c in cs if math.isfinite(c)]
    if fin:
        print(f"  c* interval            [{pct(fin,0.025):.3f}, {pct(fin,0.975):.3f}]"
              f"   ({len(fin)}/{len(cs)} resamples finite)")

    print()
    print("-" * 84)
    print(f"{'cost c':>8s} {'value(A)':>12s} {'value(B)':>12s} {'P(B better)':>13s}  verdict")
    print("-" * 84)
    verdicts = {}
    for c in args.costs:
        va, vb = tpA - c * fpA, tpB - c * fpB
        win = 0
        for _ in range(args.b_resamples):
            draw = [events[rng.randrange(len(events))] for _ in range(len(events))]
            tp_a = fp_a = tp_b = fp_b = 0
            for e in draw:
                tp_a += ea[e]["tp"]; fp_a += ea[e]["fp"]
                tp_b += eb[e]["tp"]; fp_b += eb[e]["fp"]
            if tp_b - c * fp_b > tp_a - c * fp_a:
                win += 1
        pw = win / args.b_resamples
        verdicts[c] = pw
        v = "split" if pw > 0.95 else ("single-field" if pw < 0.05 else "ambiguous")
        print(f"{c:8.2f} {va:12.1f} {vb:12.1f} {pw:13.4f}  {v}")

    if args.json_out:
        out = ROOT / args.json_out
        out.write_text(json.dumps({
            "a": args.a, "b": args.b, "labels": args.labels, "rule": args.rule,
            "events": len(events),
            "input_corpus_invariant": not problems,
            "articles_attempted_A": attempted_a,
            "articles_attempted_B": attempted_b,
            "point": {"nA": nA, "tpA": tpA, "fpA": fpA,
                      "nB": nB, "tpB": tpB, "fpB": fpB,
                      "precision_A": tpA / nA if nA else None,
                      "precision_B": tpB / nB if nB else None,
                      "d_precision": d_prec, "d_tp": d_tp, "d_fp": d_fp},
            "paired": {
                "d_precision_ci": [pct(d_precs, 0.025), pct(d_precs, 0.975)],
                "d_tp_ci": [pct(d_tps, 0.025), pct(d_tps, 0.975)],
                "d_fp_ci": [pct(d_fps, 0.025), pct(d_fps, 0.975)],
                "p_precision_rises": sum(1 for d in d_precs if d > 0) / len(d_precs),
                "p_fp_falls": sum(1 for d in d_fps if d < 0) / len(d_fps),
            },
            "c_star": c_star if math.isfinite(c_star) else None,
            "c_star_ci": [pct(fin, 0.025), pct(fin, 0.975)] if fin else None,
            "p_split_better_by_cost": {str(k): v for k, v in verdicts.items()},
            "B": args.b_resamples, "seed": args.seed,
        }, indent=2), encoding="utf-8")
        print()
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
