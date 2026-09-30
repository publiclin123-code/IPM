#!/usr/bin/env python3
"""Is the model's stated confidence usable as a firing rule?

Three diagnostics, all on forward signals in the pre-onset window:

  quantisation  how many distinct values the confidence field takes, and the
                share carried by the modal value.  A calibrated probability
                should be near-continuous; a short ordinal scale should not.
  separation    median confidence for true positives versus false positives,
                and the AUC = P(random TP scored above random FP), ties at 0.5.
                AUC 0.5 means the field cannot rank a precursor above a false
                alarm; AUC 1.0 means it can do nothing else.
  value         the gap between firing on every signal and firing at the
                oracle-chosen threshold, reported by select_threshold.py.

These three are not independent: quantisation to a handful of levels is the
mechanical reason the medians coincide and the AUC collapses, so the paper
reports them as one finding rather than three.

Note on wording: an AUC near 0.6 is weak ranking power, not zero information.
Earlier drafts claimed confidence "carries no information", which the data do
not support and which a referee can falsify in one line.  The defensible claim
is that the signal is far too weak and too coarse to carry a firing decision.

Usage
-----
  python3 analysis/confidence_informativeness.py \
      --cells results/q38_slug_v1 results/q38_slug_v2 \
              results/q38_body_v1 results/q38_body_v2
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from identity_matcher import rule_match, load, parse  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
RULE = "R4_identity"


def gt_events() -> list[dict]:
    return json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]


def collect(signals_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """Confidence values for true positives, false positives, and everything."""
    tp: list[float] = []
    fp: list[float] = []
    allv: list[float] = []
    for ev in gt_events():
        p = signals_dir / f"{ev['event_id']}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            c = s.get("confidence")
            if c is None:
                continue
            allv.append(float(c))
            (tp if rule_match(s, ev, RULE) else fp).append(float(c))
    return tp, fp, allv


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(random positive outranks random negative), ties counted as 0.5."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="extraction directories to diagnose")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    out: dict[str, dict] = {}
    print("=" * 94)
    print("IS THE CONFIDENCE FIELD USABLE AS A FIRING RULE?")
    print("=" * 94)
    print(f"  rule {RULE}   forward window 0 < lead <= {WINDOW} d")
    print()
    print("QUANTISATION OF THE CONFIDENCE FIELD")
    print("-" * 94)
    print(f"{'cell':18s} {'n':>6s} {'distinct':>9s} {'mode':>6s} {'share':>7s}  top 3 values")
    print("-" * 94)
    quant = {}
    for c in args.cells:
        _, _, allv = collect(ROOT / c)
        if not allv:
            print(f"{Path(c).name:18s} {'--':>6s}  no signals")
            continue
        cnt = Counter(allv)
        top = cnt.most_common(3)
        share = top[0][1] / len(allv)
        desc = "  ".join(f"{v:.2f} x{n} ({100*n/len(allv):.0f}%)" for v, n in top)
        print(f"{Path(c).name:18s} {len(allv):6d} {len(cnt):9d} {top[0][0]:6.2f} "
              f"{100*share:6.1f}%  {desc}")
        quant[Path(c).name] = {"n": len(allv), "distinct": len(cnt),
                               "mode": top[0][0], "mode_share": share,
                               "top3": [[v, n] for v, n in top]}

    print()
    print("SEPARATION BETWEEN TRUE POSITIVES AND FALSE POSITIVES")
    print("-" * 94)
    print(f"{'cell':18s} {'nTP':>5s} {'nFP':>5s} {'medTP':>7s} {'medFP':>7s} {'AUC':>7s}  reading")
    print("-" * 94)
    sep = {}
    for c in args.cells:
        tp, fp, _ = collect(ROOT / c)
        if not tp or not fp:
            print(f"{Path(c).name:18s} {len(tp):5d} {len(fp):5d}  insufficient")
            continue
        a = auc(tp, fp)
        mtp, mfp = stats.median(tp), stats.median(fp)
        if abs(a - 0.5) < 0.05:
            reading = "no ranking power"
        elif abs(a - 0.5) < 0.15:
            reading = "weak ranking power"
        else:
            reading = "moderate ranking power"
        print(f"{Path(c).name:18s} {len(tp):5d} {len(fp):5d} {mtp:7.3f} {mfp:7.3f} "
              f"{a:7.4f}  {reading}")
        sep[Path(c).name] = {"n_tp": len(tp), "n_fp": len(fp),
                             "median_tp": mtp, "median_fp": mfp, "auc": a}

    print()
    print("=" * 94)
    print("READING")
    print("=" * 94)
    shares = [quant[k]["mode_share"] for k in quant if k in quant]
    aucs = [sep[k]["auc"] for k in sep]
    if shares:
        print(f"  the modal confidence value carries {100*min(shares):.0f}-{100*max(shares):.0f}% "
              f"of all signals, over {min(q['distinct'] for q in quant.values())}-"
              f"{max(q['distinct'] for q in quant.values())} distinct levels")
    if aucs:
        print(f"  AUC lies in [{min(aucs):.4f}, {max(aucs):.4f}]")
    print("  the field is a coarse ordinal scale, not a calibrated probability;")
    print("  that single fact produces the coincident medians, the low AUC, and")
    print("  the flat sensitivity of the optimal threshold to the cost ratio.")

    if args.json_out:
        p = ROOT / args.json_out
        p.write_text(json.dumps({"rule": RULE, "window_days": WINDOW,
                                 "quantisation": quant, "separation": sep},
                                indent=2), encoding="utf-8")
        print()
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
