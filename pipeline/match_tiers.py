"""Tiered entity matching, to test which match rule the evidence supports.

Background
----------
The manuscript reports protocol precision 0.873 under the loose rule
(commodity OR region OR company) and discloses 0.582 under the strict rule
(commodity AND region, or company), explaining the gap as follows:

    "because GDELT slug tokens are sparse and single-dimension matches carry
     most of the signal, so requiring both dimensions turns true precursors
     into false positives"

That explanation assumes the single-dimension matches *are* true precursors.
`pipeline/audit_match_evidence.py` showed 69 of the 207 true positives rest on
a generic string with no company-level evidence, and the Toyota event's 27
true positives are steel-tariff articles credited because "steel" appears in
both the signal and the event's commodity list.

So the question is empirical: are the 69 matches that strict discards genuine
precursors, or coincidences? This script answers it by (a) recomputing the
headline at each tier and (b) printing the discarded true positives so they can
be read.

Matching tiers
--------------
  T0 loose_substring   company OR geo OR commodity, `a in b or b in a`   (current headline)
  T1 loose_word        same logic, word-boundary matching
  T2 two_dimension     company OR (geo AND commodity), word-boundary     (current "strict")
  T3 company_only      company match only, word-boundary

T0 reproduces the published number; the others are progressively stricter.

Word-boundary matching matters independently of tier: under substring matching
the name "christopher steele" matches the commodity "steel", and "ever" (from
*Ever Given*) matches "everything" and "never".

Usage
-----
  python3 pipeline/match_tiers.py
  python3 pipeline/match_tiers.py --signals-dir results/v2_body --list-dropped
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

FOREWARD = {"forward_looking", "latent"}
TIERS = ["T0_loose_substring", "T1_loose_word", "T2_two_dimension", "T3_company_only"]


def norm(s: str) -> str:
    return (s or "").strip().lower()


def _words(s):
    return [w for w in re.split(r"[^a-z0-9]+", norm(s)) if w]


def hit_substring(sig_vals, gt_vals) -> bool:
    """Current production rule:双向子串."""
    sv = [norm(v) for v in (sig_vals or []) if v]
    gv = [norm(v) for v in (gt_vals or []) if v]
    for a in sv:
        for b in gv:
            if a == b or a in b or b in a:
                return True
    return False


def hit_word(sig_vals, gt_vals) -> bool:
    """Word-boundary match on whole phrases, no substring fallback.

    A multi-word ground-truth phrase such as "automotive parts" matches only if
    those words appear consecutively in the signal value.
    """
    sv = [norm(v) for v in (sig_vals or []) if v]
    gv = [norm(v) for v in (gt_vals or []) if v]
    for a in sv:
        aw = _words(a)
        if not aw:
            continue
        for b in gv:
            bw = _words(b)
            if not bw:
                continue
            if aw == bw:
                return True
            # consecutive-word containment, either direction
            for (x, y) in ((aw, bw), (bw, aw)):
                if len(x) >= len(y):
                    for i in range(len(x) - len(y) + 1):
                        if x[i:i + len(y)] == y:
                            return True
    return False


def company_names(sig) -> list[str]:
    out = []
    for c in sig.get("companies", []) or []:
        out.append(norm(c.get("name", "")) if isinstance(c, dict) else norm(str(c)))
    return [x for x in out if x]


def matches(sig: dict, ev: dict, tier: str) -> tuple[bool, str]:
    """Return (matched, basis). basis records which criterion fired."""
    comp = (hit_substring if tier == "T0_loose_substring" else hit_word)(
        company_names(sig), ev.get("companies", []))
    geo = (hit_substring if tier == "T0_loose_substring" else hit_word)(
        sig.get("geographies", []), ev.get("geographies", []))
    com = (hit_substring if tier == "T0_loose_substring" else hit_word)(
        sig.get("commodities", []), ev.get("commodities", []))

    if tier == "T0_loose_substring":
        if comp:
            return True, "company"
        if geo and com:
            return True, "geo+commodity"
        if com:
            return True, "commodity_only"
        if geo:
            return True, "geo_only"
        return False, ""
    if tier == "T1_loose_word":
        if comp:
            return True, "company"
        if geo and com:
            return True, "geo+commodity"
        if com:
            return True, "commodity_only"
        if geo:
            return True, "geo_only"
        return False, ""
    if tier == "T2_two_dimension":
        if comp:
            return True, "company"
        if geo and com:
            return True, "geo+commodity"
        return False, ""
    if tier == "T3_company_only":
        if comp:
            return True, "company"
        return False, ""
    raise ValueError(tier)


def parse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def load_jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if s.get("status") == "error":
            continue
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare entity-match tiers")
    ap.add_argument("--signals-dir", default="results/v2")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--list-dropped", action="store_true",
                    help="print the TPs that T2 discards relative to T1")
    args = ap.parse_args()

    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    sdir = ROOT / args.signals_dir

    results = {t: {"tp": 0, "fwd": 0, "hits": 0, "basis": Counter()} for t in TIERS}
    per_event = defaultdict(dict)
    dropped: list[tuple] = []

    for ev in gt:
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        sigs = [s for s in load_jsonl(p)
                if s.get("temporality") in FOREWARD and s.get("signal_date")]
        fwd = []
        for s in sigs:
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if 0 < lead <= args.window:
                fwd.append((s, lead))

        for t in TIERS:
            n_tp = 0
            for s, _lead in fwd:
                ok, basis = matches(s, ev, t)
                if ok:
                    n_tp += 1
            results[t]["fwd"] += len(fwd)
            results[t]["tp"] += n_tp
            results[t]["hits"] += 1 if n_tp else 0
            per_event[eid][t] = n_tp

        # what T2 discards relative to T1
        if args.list_dropped:
            for s, lead in fwd:
                ok1, b1 = matches(s, ev, "T1_loose_word")
                ok2, _ = matches(s, ev, "T2_two_dimension")
                if ok1 and not ok2:
                    dropped.append((eid, lead, b1))
                    dropped[-1] = (eid, lead, b1, s)

    print(f"signals dir: {args.signals_dir}   window: {args.window}d")
    print(f"{'tier':24s} {'forward':>8s} {'TP':>6s} {'precision':>10s} {'events hit':>11s}")
    print("-" * 66)
    base = results["T0_loose_substring"]["tp"]
    for t in TIERS:
        r = results[t]
        prec = r["tp"] / r["fwd"] if r["fwd"] else 0.0
        delta = "" if t == TIERS[0] else f"  ({r['tp'] - base:+d} TP)"
        print(f"{t:24s} {r['fwd']:8d} {r['tp']:6d} {prec:10.4f} "
              f"{r['hits']:6d}/18{delta}")

    print()
    print("bases within T0 (reproduces the published 0.873 rule):")
    for k, v in results["T0_loose_substring"]["basis"].most_common():
        print(f"   {k}: {v}")

    if args.list_dropped and dropped:
        print()
        print("=" * 100)
        print(f"True positives that T1 keeps but T2 discards (n={len(dropped)})")
        print("If these are genuine precursors, the manuscript's defence of the loose rule holds.")
        print("=" * 100)
        by_ev = Counter(d[0] for d in dropped)
        for eid, n in by_ev.most_common():
            print(f"\n--- {eid}  ({n} discarded) ---")
            shown = 0
            for e, lead, b1, s in dropped:
                if e != eid or shown >= 6:
                    continue
                desc = str(s.get("description", ""))[:120]
                print(f"   [{s.get('signal_date')}] lead={lead:3d} basis={b1:15s}")
                print(f"        {desc}")
                print(f"        geos={s.get('geographies')} coms={s.get('commodities')}")
                shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
