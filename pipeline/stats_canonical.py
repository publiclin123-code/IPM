"""Canonical uncertainty quantification (I5) and recall sensitivity (I4).

Replaces the ad-hoc bootstrap scripts, which hard-coded a 5-event list from an
earlier corpus and were never reconciled with the abstract claims. The manuscript
currently cites bootstrap intervals twice ("the wide bootstrap band at long
leads"; "intervals do not overlap") without stating B, the resampling unit, the
interval type, or the seed. This script fixes all four and produces every
interval the paper needs from one code path.

SPECIFICATION (everything the manuscript must state)
----------------------------------------------------
Statistic        beta_0, protocol precision, pooled FWGS
Estimator        signal-level for beta_0 and precision; mean of per-signal FWGS
Resampling unit  *event*, then *signal within event* (two-stage, event-stratified)
                 Rationale: signals within an event share a stream and are not
                 independent. Resampling signals alone understates the interval
                 width by treating 17 streams as ~200 independent draws.
B                2000
Interval         percentile, 2.5th and 97.5th
Seed             42 (fixed, so the numbers are reproducible)
Degenerate       events with no qualifying signals contribute an empty strand
                 and are dropped from that replicate, matching how the point
                 estimate treats them (n = 0, not n = 0/0)

RECALL SENSITIVITY (I4)
-----------------------
The manuscript states recall is not defined. That is a defensible scope decision
for the internal comparisons, but a measurement paper should bound the effect.
Under an assumed extraction recall r, the number of *true* precursors in a window
is inflated by 1/r while the number of *false* positives is unchanged, so

    precision(r) = TP / (TP + FP + TP*(1/r - 1))
                 = TP / (TP/r + FP)

which is the missing-mass correction. r = 1 recovers the observed precision; low r
is pessimistic. This is a bound, not an estimate, and is reported as such.

Usage
-----
  python3 pipeline/stats_canonical.py
  python3 pipeline/stats_canonical.py --variant body
  python3 pipeline/stats_canonical.py --out results/stats_canonical.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "pipeline"))
from identity_matcher import rule_match, load, parse, IDENTITY  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
TAU = 30.0                     # FWGS foresight scale, days
LAMBDA = 0.5                   # FWGS faithfulness penalty
B = 2000
SEED = 42
MISSING = ROOT / "draft/ipm/figures"


# ------------------------------------------------------------------ FWGS pieces
def w_foresight(lead: float) -> float:
    """1 - exp(-lead/tau). Zero at onset, saturating at long leads."""
    return 1.0 - math.exp(-max(lead, 0.0) / TAU)


def faithfulness(sig: dict, articles: dict) -> float:
    """Share of trigger phrases appearing verbatim in the source text."""
    phrases = sig.get("trigger_phrases") or []
    if not phrases:
        return 0.0
    text = articles.get(sig.get("input_id") or sig.get("id") or "", "")
    if not text:
        return 0.0
    low = text.lower()
    hit = sum(1 for p in phrases if str(p).lower() in low)
    return hit / len(phrases)


def load_articles(eid: str, variant: str) -> dict:
    base = ROOT / ("data/by_event" if variant == "title" else "data/by_event_body")
    p = base / f"{eid}.jsonl"
    out = {}
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
        key = r.get("id")
        if key:
            out[key] = (r.get("text") or r.get("title") or "")
    return out


def _resolve_sdir(variant: str, signals_dir: str = ""):
    """Extraction directory to score; --signals-dir overrides the historical default."""
    if signals_dir:
        p = Path(signals_dir)
        return p if p.is_absolute() else ROOT / p
    return ROOT / ("results/v2" if variant == "title" else "results/v2_body")


# ------------------------------------------------------------------- assemble
def build(variant: str, rule: str, signals_dir: str = "") -> dict:
    sdir = _resolve_sdir(variant, signals_dir)
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]

    streams: dict[str, list[dict]] = {}
    for ev in gt:
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        arts = load_articles(eid, variant)
        sigs = []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            is_tp = rule_match(s, ev, rule)
            g = faithfulness(s, arts)
            sigs.append({"lead": lead, "is_tp": is_tp, "g": g,
                         "event_id": eid, "signal_id": s.get("signal_id", "")})
        streams[eid] = sigs
    return streams


def fwgs_of(sigs: list[dict]) -> float | None:
    if not sigs:
        return None
    vals = [w_foresight(s["lead"]) * s["g"] if s["is_tp"]
            else -LAMBDA * (1 - s["g"]) * float(s["is_tp"] is False and s["g"] > 0)
            for s in sigs]
    # TP signals contribute w_F * g; FP signals contribute -lambda*(1-g)
    vals = []
    for s in sigs:
        if s["is_tp"]:
            vals.append(w_foresight(s["lead"]) * s["g"])
        else:
            vals.append(-LAMBDA * (1.0 - s["g"]))
    return sum(vals) / len(vals)


def pooled_fwgs(streams: dict) -> float | None:
    """Mean of per-signal FWGS across all events, pooled."""
    allsig = [s for v in streams.values() for s in v]
    return fwgs_of(allsig)


def point_estimates(streams: dict) -> dict:
    tp = sum(1 for v in streams.values() for s in v if s["is_tp"])
    n = sum(len(v) for v in streams.values())
    return {"tp": tp, "n": n,
            "precision": tp / n if n else None,
            "fwgs": pooled_fwgs(streams)}


# ------------------------------------------------------------------ bootstrap
def event_stratified(streams: dict) -> dict:
    """Two-stage bootstrap: resample events, then signals within event."""
    rng = random.Random(SEED)
    eids = [e for e, v in streams.items() if v]      # drop empty streams, as the
    precs, fwgs = [], []                              # point estimate does
    for _ in range(B):
        picked = [rng.choice(eids) for _ in eids]
        samp = []
        for e in picked:
            pool = streams[e]
            samp.extend(rng.choice(pool) for _ in range(len(pool)))
        if not samp:
            continue
        tp = sum(1 for s in samp if s["is_tp"])
        precs.append(tp / len(samp))
        f = fwgs_of(samp)
        if f is not None:
            fwgs.append(f)
    def ci(v):
        if not v:
            return (None, None)
        v = sorted(v)
        return (v[int(0.025 * len(v))], v[min(int(0.975 * len(v)), len(v) - 1)])
    return {"B": B, "seed": SEED, "unit": "event-then-signal",
            "interval": "percentile 2.5/97.5",
            "precision_ci": ci(precs), "fwgs_ci": ci(fwgs),
            "n_replicates": len(precs)}


def beta0_bootstrap(post: list[dict], B: int = B, seed: int = SEED) -> dict:
    """Event-stratified bootstrap for beta_0 on the post-onset pool."""
    rng = random.Random(seed)
    by_ev = defaultdict(list)
    for r in post:
        by_ev[r["event_id"]].append(r)
    eids = [e for e, v in by_ev.items() if v]
    vals = []
    for _ in range(B):
        picked = [rng.choice(eids) for _ in eids]
        samp = []
        for e in picked:
            pool = by_ev[e]
            samp.extend(rng.choice(pool) for _ in range(len(pool)))
        if not samp:
            continue
        vals.append(sum(1 for s in samp if s["is_contam"]) / len(samp))
    vals.sort()
    return {"B": B, "seed": seed, "unit": "event-then-signal",
            "interval": "percentile 2.5/97.5", "n": len(post),
            "point": (sum(1 for s in post if s["is_contam"]) / len(post)) if post else None,
            "ci": (vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)])
            if vals else (None, None)}


# Background (post-onset negative-control) pools that produce beta_0.
#
# Original layout: four pre-aggregated files in results/_background/.  Re-extraction
# layout: one _background_signals.jsonl per cell directory.  The re-extracted
# corpora must be used with the re-extracted forward signals, because beta_0 is
# defined as a naive-versus-clock-split contrast and mixing a re-extracted
# numerator with an original denominator would confound the schema change with
# the model change.
BG_LEGACY = {
    "title_naive": ROOT / "results" / "_background" / "signals_postevent_naive_new.jsonl",
    "title_v2": ROOT / "results" / "_background" / "signals_postevent_v2_new.jsonl",
    "body_naive": ROOT / "results" / "_background" / "signals_postevent_naive_body.jsonl",
    "body_v2": ROOT / "results" / "_background" / "signals_postevent_v2_body.jsonl",
}
BG_Q38 = {
    "title_naive": ROOT / "results" / "q38_bg_slug_v1" / "_background_signals.jsonl",
    "title_v2": ROOT / "results" / "q38_bg_slug_v2" / "_background_signals.jsonl",
    "body_naive": ROOT / "results" / "q38_bg_body_v1" / "_background_signals.jsonl",
    "body_v2": ROOT / "results" / "q38_bg_body_v2" / "_background_signals.jsonl",
}
BG_ACTIVE: dict = BG_LEGACY


def load_post(which: str) -> list[dict]:
    """Post-onset signals with their event attribution, for beta_0."""
    from metrics import _build_article_event_map
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    # The two background corpora carry the same 340 article ids, so the slug file
    # supplies a valid article->event map for the body variants too.
    id2e = _build_article_event_map(ROOT / "data" / "by_event" / "_background.jsonl", gt, post_days=60)
    p = BG_ACTIVE[which]
    if not p.exists():
        raise FileNotFoundError(
            f"background pool for {which} missing: {p.relative_to(ROOT)}\n"
            f"  run: bash pipeline/reextract_q38_phases34.sh"
        )
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = id2e.get(s.get("input_id"))
        if not eid:
            continue
        onset = parse(next(e["gt_onset_date"] for e in gt if e["event_id"] == eid))
        try:
            lag = (parse(s["signal_date"]) - onset).days
        except Exception:
            continue
        if not (1 <= lag <= 14):
            continue
        if not s.get("temporality"):
            continue
        out.append({"event_id": eid, "is_contam": s["temporality"] in FORWARD})
    return out


def recall_sensitivity(tp: int, fp: int, rs=(1.0, 0.9, 0.7, 0.5)) -> dict:
    """precision(r) = TP / (TP/r + FP). Missing-mass correction; a bound."""
    out = {}
    for r in rs:
        denom = tp / r + fp
        out[r] = tp / denom if denom else None
    return out


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--rule", default="R4_identity")
    ap.add_argument("--signals-dir", default="",
                    help="override extraction dir (e.g. results/q38_slug_v1)")
    ap.add_argument("--bg-cells", default="legacy", choices=["legacy", "q38"],
                    help="background pools for beta_0: original results/_background "
                         "or the re-extracted results/q38_bg_* cells")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global BG_ACTIVE
    BG_ACTIVE = BG_Q38 if args.bg_cells == "q38" else BG_LEGACY

    streams = build(args.variant, args.rule, args.signals_dir)
    pe = point_estimates(streams)
    ci = event_stratified(streams)
    fp = pe["n"] - pe["tp"]
    rec = recall_sensitivity(pe["tp"], fp)

    print("=" * 100)
    print("SPECIFICATION (this must appear in the manuscript)")
    print("=" * 100)
    print(f"  statistic        beta_0, protocol precision, pooled FWGS")
    print(f"  resampling unit  event, then signal within event (two-stage)")
    print(f"  B                {B}")
    print(f"  interval         percentile, 2.5th and 97.5th")
    print(f"  seed             {SEED}")
    print(f"  rationale        signals within a stream are not independent; resampling")
    print(f"                   signals alone would understate interval width")

    print()
    print("=" * 100)
    print(f"POINT ESTIMATES   variant={args.variant}   rule={args.rule}")
    print("=" * 100)
    print(f"  forward signals        {pe['n']}")
    print(f"  true positives         {pe['tp']}")
    print(f"  protocol precision     {pe['precision']:.4f}")
    print(f"  pooled FWGS            {pe['fwgs']:.4f}")

    print()
    print("=" * 100)
    print("BOOTSTRAP INTERVALS")
    print("=" * 100)
    print(f"  precision  {pe['precision']:.4f}  "
          f"[{ci['precision_ci'][0]:.4f}, {ci['precision_ci'][1]:.4f}]")
    print(f"  FWGS       {pe['fwgs']:.4f}  "
          f"[{ci['fwgs_ci'][0]:.4f}, {ci['fwgs_ci'][1]:.4f}]")

    print()
    print("=" * 100)
    print("BETA_0 WITH INTERVALS (all four variants)")
    print("=" * 100)
    print(f"  {'variant':14s} {'beta_0':>8s} {'95% CI':>22s} {'n':>5s}")
    print("-" * 100)
    beta_rows = {}
    for which in ("title_naive", "title_v2", "body_naive", "body_v2"):
        post = load_post(which)
        b = beta0_bootstrap(post)
        beta_rows[which] = b
        print(f"  {which:14s} {b['point']:8.4f} "
              f"  [{b['ci'][0]:.4f}, {b['ci'][1]:.4f}] {b['n']:5d}")

    print()
    print("=" * 100)
    print("RECALL SENSITIVITY  (upper-bound framing; r=1 is the observed value)")
    print("=" * 100)
    print(f"  Using TP={pe['tp']}, FP={fp}. Precision(r) = TP / (TP/r + FP).")
    print(f"  {'r':>6s} {'precision':>11s} {'implied true positives':>24s}")
    print("-" * 100)
    for r, p in rec.items():
        print(f"  {r:6.2f} {p:11.4f} {int(round(pe['tp'] / r)):24d}")
    print()
    print("  Reading: at r=1 the observed precision holds. At r=0.5, half of the")
    print("  genuine precursors were never emitted, so the same FP count is divided")
    print("  into a larger pool of truth. The statistic is a bound, not an estimate:")
    print("  it assumes missing precursors are indistinguishable from false alarms")

    if args.out:
        op = ROOT / args.out
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps({
            "variant": args.variant, "rule": args.rule,
            "specification": {"B": B, "seed": SEED, "unit": "event-then-signal",
                              "interval": "percentile 2.5/97.5"},
            "point": pe, "bootstrap": ci, "beta0": beta_rows,
            "recall_sensitivity": rec,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten -> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
