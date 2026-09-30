"""Option B: type-conditional schema selection.

The problem this tests
----------------------
`audit_schema_change.py` established that the clock split is a trade, not a fix:
it removes 166 pre-onset signals of which a substantial number are genuine
precursors (Ukrainian embassy evacuation at lead 11, Wuhan outbreak spread at
lead 14, the Chips for America Act at lead 78). The split performs well on
slow-developing events and badly on abrupt ones, because "a realized trigger is
always confirmation" cannot distinguish "some trigger happened" from "*this*
event's trigger happened".

The hypothesis
--------------
The trade is not uniform across event types, and the type is inferable online.
If so, a system can pick the schema per event and avoid the trade:

    choose the single-field schema for events whose precursors are
    sub-events (geopolitical/regulatory build-ups)
    choose the clock-split schema for events whose precursors are
    conditions (logistics/quality deteriorations)

This is a decision rule, not a new model, which matters for how it can be sold:
it is a *method* in the sense that it prescribes an action, and it can be
evaluated like any classifier.

Two things must be shown for the hypothesis to hold:

  1. The trade-off is type-dependent. The split must help some event types and
     hurt others, and it must hurt substantially -- otherwise there is nothing
     to select on.
  2. The type is recoverable online. It must be inferable from quantities
     available at time t, without the a-priori label and without the onset.
     Otherwise the rule is unimplementable and this is an oracle, not a method.

Both are tested below, and either can fail. Reporting a failure is the point.

Definitions
-----------
For an event, with the a-priori type held back:

  gain(s)  = pre-onset signals that the schema emits and that match the event
  loss(s)  = pre-onset signals that the *other* schema emits, that match the
             event, and that s drops
  net(s)   = gain(s) - loss(s)

"Match" uses the loose entity rule, the manuscript's own, so the comparison is
like-for-like. A `--strict` variant restricts matching to company evidence and
is reported separately because the loose rule is known to over-credit
(DATA_ISSUES.md DI-8).

Usage
-----
  python3 pipeline/cws_type_conditional.py
  python3 pipeline/cws_type_conditional.py --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "analysis"))
from validate import signal_matches_event, _parse_date  # noqa: E402
from event_meta import EVENT_TYPES  # noqa: E402        # only for reporting, never as a feature

FORWARD = {"forward_looking", "latent"}
WINDOW = 180


def company_hit(sig: dict, ev: dict) -> bool:
    def names(v):
        out = []
        for c in v or []:
            out.append((c.get("name", "") if isinstance(c, dict) else str(c)).strip().lower())
        return [x for x in out if x]
    gv = [str(x).strip().lower() for x in (ev.get("companies") or [])]
    if not gv:
        return False
    for a in names(sig.get("companies")):
        for b in gv:
            if a == b or a in b or b in a:
                return True
    return False


def load(p: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if s.get("status") in ("empty", "error") or not s.get("temporality"):
            continue
        k = s.get("input_id") or s.get("id")
        if k:
            out[k] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="count matches only on company evidence")
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    args = ap.parse_args()
    matcher = company_hit if args.strict else (lambda s, e: signal_matches_event(s, e, strict=False))

    v1_dir = ROOT / "results"
    v2_dir = ROOT / ("results/v2" if args.variant == "title" else "results/v2_body")
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]

    per_event = []
    for ev in gt:
        eid = ev["event_id"]
        p1, p2 = v1_dir / f"{eid}_signals.jsonl", v2_dir / f"{eid}_signals.jsonl"
        if not (p1.exists() and p2.exists()):
            continue
        onset = _parse_date(ev["gt_onset_date"])
        s1, s2 = load(p1), load(p2)

        def pre_hits(sigs):
            """article ids of pre-onset matching forward signals"""
            out = set()
            for aid, s in sigs.items():
                if s.get("temporality") not in FORWARD:
                    continue
                try:
                    lead = (onset - _parse_date(s["signal_date"])).days
                except Exception:
                    continue
                if 0 < lead <= WINDOW and matcher(s, ev):
                    out.add(aid)
            return out

        h1, h2 = pre_hits(s1), pre_hits(s2)
        # schema 1 = single field (v1), schema 2 = clock split (v2)
        gain1 = len(h1 - h2)          # v1 finds these, v2 misses them
        gain2 = len(h2 - h1)          # v2 finds these, v1 misses them
        per_event.append({
            "eid": eid,
            "type": EVENT_TYPES.get(eid, "?"),
            "h1": len(h1), "h2": len(h2),
            "gain1": gain1, "gain2": gain2,
            "gain4split": gain2,        # split's gain over single-field
            "gain4single": gain1,       # single-field's gain over split
            "net4split": gain2 - gain1,
        })

    print(f"variant={args.variant}  match={'company-only' if args.strict else 'loose'}")
    print("=" * 104)
    print(f"{'event':34s} {'type':10s} {'h_v1':>5s} {'h_v2':>5s} "
          f"{'v2gain':>7s} {'v1gain':>7s} {'net(v2)':>8s}")
    print("-" * 104)
    for r in sorted(per_event, key=lambda x: (x["type"], x["eid"])):
        print(f"{r['eid']:34s} {r['type']:10s} {r['h1']:5d} {r['h2']:5d} "
              f"{r['gain2']:7d} {r['gain1']:7d} {r['net4split']:8d}")

    print()
    print("=" * 104)
    print("TEST 1 - is the trade-off type-dependent?")
    print("=" * 104)
    by_type = defaultdict(list)
    for r in per_event:
        by_type[r["type"]].append(r)
    print(f"{'type':12s} {'n':>3s} {'sum v2gain':>11s} {'sum v1gain':>11s} "
          f"{'mean net(v2)':>13s} {'events where v2 wins':>21s}")
    print("-" * 104)
    for t, rs in sorted(by_type.items()):
        g2 = sum(r["gain2"] for r in rs)
        g1 = sum(r["gain1"] for r in rs)
        net = np.mean([r["net4split"] for r in rs])
        wins = sum(1 for r in rs if r["net4split"] > 0)
        print(f"{t:12s} {len(rs):3d} {g2:11d} {g1:11d} {net:13.2f} "
              f"{wins:10d}/{len(rs)}")
    # does type explain the sign of net?
    nets = {r["eid"]: r["net4split"] for r in per_event}
    types = {r["eid"]: r["type"] for r in per_event}
    from itertools import combinations
    # rank separation: mean net by type
    means = {t: np.mean([r["net4split"] for r in rs]) for t, rs in by_type.items()}
    order = sorted(means, key=lambda t: means[t])
    print()
    print("  type ranking by mean net(split):")
    for t in order:
        print(f"     {t:12s} {means[t]:+.2f}")
    spread = max(means.values()) - min(means.values())
    print(f"  spread across types: {spread:.2f}")
    if spread > 1.0:
        print("  -> the trade IS type-dependent: there is something to select on.")
    else:
        print("  -> the trade is NOT type-dependent; selection cannot help.")
        print("     Option B fails on this evidence.")

    print()
    print("=" * 104)
    print("TEST 2 - is the type recoverable online (no a-priori label, no onset)?")
    print("=" * 104)
    # Features from the pre-onset signals only, as a system would see them.
    feats, labels, names_g = [], [], []
    for r in per_event:
        eid = r["eid"]
        p1 = v1_dir / f"{eid}_signals.jsonl"
        s1 = load(p1)
        onset = _parse_date(next(e for e in gt if e["event_id"] == eid)["gt_onset_date"])
        vals = []
        for aid, s in s1.items():
            if s.get("temporality") not in FORWARD:
                continue
            try:
                lead = (onset - _parse_date(s["signal_date"])).days
            except Exception:
                continue
            if 0 < lead <= WINDOW:
                vals.append(s)
        if len(vals) < 3:
            continue
        conf = np.mean([float(v.get("confidence") or 0) for v in vals])
        sev = np.mean([float(v.get("severity") or 0) for v in vals])
        ntp = np.mean([len(v.get("trigger_phrases") or []) for v in vals])
        unc = np.mean([1.0 if v.get("impact_uncertainty") else 0.0 for v in vals])
        ets = Counter(str(v.get("event_type") or "none") for v in vals)
        top_et = ets.most_common(1)[0][0]
        feats.append([conf, sev, ntp, unc, float(len(vals)),
                      r["gain2"], r["gain1"]])
        labels.append(r["net4split"])
        names_g.append((eid, r["type"], top_et, r["net4split"], r["gain2"], r["gain1"]))

    # Can we predict which schema is better, out-of-fold, from these features?
    X = np.array([f[:5] for f in feats], dtype=float)
    y = np.array([1 if l > 0 else 0 for l in labels])
    if len(set(y)) < 2:
        print("  only one class present; cannot test recoverability")
        return 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        if len(set(y[tr])) < 2:
            preds[te] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, preds)
    except Exception:
        auc = float("nan")
    acc = float(((preds > 0.5) == y).mean())
    print(f"  predicting 'does the split win on this event' from online features:")
    print(f"     n={len(y)}  base rate={y.mean():.2f}  LOO accuracy={acc:.3f}  AUC={auc:.3f}")
    if auc > 0.7:
        print("     -> recoverable online: the rule is implementable.")
    else:
        print("     -> NOT reliably recoverable: the rule would be an oracle, not a method.")

    print()
    print("  per-event view (online features vs outcome):")
    print(f"     {'event':34s} {'type':10s} {'top event_type':18s} {'net':>5s}")
    for eid, t, tet, net, g2, g1 in names_g:
        print(f"     {eid:34s} {t:10s} {tet:18s} {net:5d}")

    # oracle bound: what if we always picked the better schema?
    oracle_gain = sum(max(r["gain2"], r["gain1"]) for r in per_event)
    best_single = max(sum(r["gain2"] for r in per_event), sum(r["gain1"] for r in per_event))
    print()
    print(f"  oracle (pick better schema per event): {oracle_gain} matched pre-onset signals")
    print(f"  best fixed schema:                     {best_single}")
    print(f"  headroom from selection:               "
          f"{oracle_gain - best_single} signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
