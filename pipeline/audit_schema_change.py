"""Is the clock split a defect fix, or a precision-recall trade?

The manuscript's improvement #1 claims that splitting one free-text temporality
field into (event time, impact time) is a fix: contamination beta_0 falls from
0.500 to 0.233, and the pre-onset forward-signal count falls from 363 to 237 while
pooled FWGS rises from 0.470 to 0.535.

The drop from 363 to 237 is doing a lot of work in that sentence and the
manuscript never examines it. If the 126 discarded signals were noise, the change
is a clean fix. If some were genuine precursors -- pre-onset and entity-matching
-- then the change buys precision with recall, which is a trade-off, not a repair,
and calling it an improvement is wrong.

This script answers that directly. For every ground-truth event run under both
prompts, it:

  1. takes the v1 (single-field) forward signals and the v2 (clock-split) ones,
  2. joins them by article (`input_id`), since the two prompts do not produce
     the same signal identifiers,
  3. asks what v2 did to each v1 signal: still forward, relabelled to
     confirmation, or not emitted,
  4. and classifies every discarded signal by whether it was a genuine precursor
     (pre-onset and entity-matching).

Also reports the converse, which the manuscript also does not report: v2 signals
that are forward but that v1 did not flag, i.e. what the split newly admits.

Usage
-----
  python3 pipeline/audit_schema_change.py
  python3 pipeline/audit_schema_change.py --variant body
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import signal_matches_event, _parse_date  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180


def load(p: Path) -> dict[str, dict]:
    """article input_id -> signal record (last wins)."""
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
        if s.get("status") in ("empty", "error"):
            continue
        if not s.get("temporality"):
            continue
        key = s.get("input_id") or s.get("id")
        if key:
            out[key] = s
    return out


def company_hit(sig: dict, ev: dict) -> bool:
    """Company-only match. The strictest defensible definition of a hit.

    Needed because `signal_matches_event(strict=False)` is a substring OR over
    geography and commodity, and `audit_match_evidence.py` shows a third of the
    resulting true positives have no company evidence. A discarded signal that
    only a generic noun matched may not be a genuine precursor at all.
    """
    def names(v):
        out = []
        for c in v or []:
            out.append((c.get("name", "") if isinstance(c, dict) else str(c)).strip().lower())
        return [x for x in out if x]
    gv = [str(x).strip().lower() for x in (ev.get("companies") or [])]
    if not gv:
        return False                     # event has no company identity at all
    for a in names(sig.get("companies")):
        for b in gv:
            if a == b or a in b or b in a:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--strict", action="store_true",
                    help="count a discarded signal as a hit only on company evidence")
    args = ap.parse_args()

    v1_dir = ROOT / "results"
    v2_dir = ROOT / ("results/v2" if args.variant == "title" else "results/v2_body")

    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]

    rows = []
    tot = Counter()
    discarded_detail: list[tuple] = []
    newly: list[tuple] = []

    for ev in gt:
        eid = ev["event_id"]
        p1 = v1_dir / f"{eid}_signals.jsonl"
        p2 = v2_dir / f"{eid}_signals.jsonl"
        if not p1.exists() or not p2.exists():
            continue
        onset = _parse_date(ev["gt_onset_date"])
        s1, s2 = load(p1), load(p2)

        # --- what happened to each v1 forward signal ---
        n1_fwd = n1_fwd_pre = 0
        kept = relabelled = dropped = 0
        disc_tp = disc_fp = 0
        for aid, a in s1.items():
            if a.get("temporality") not in FORWARD:
                continue
            n1_fwd += 1
            try:
                lead = (onset - _parse_date(a["signal_date"])).days
            except Exception:
                continue
            pre = 0 < lead <= WINDOW
            if pre:
                n1_fwd_pre += 1
            b = s2.get(aid)
            if b is None:
                fate = "dropped"
                dropped += 1
            elif b.get("temporality") in FORWARD:
                fate = "kept"
                kept += 1
            else:
                fate = "relabelled"
                relabelled += 1
            if pre:
                is_hit = (company_hit(a, ev) if args.strict
                          else signal_matches_event(a, ev, strict=False))
                if fate in ("dropped", "relabelled"):
                    if is_hit:
                        disc_tp += 1
                    else:
                        disc_fp += 1
                    discarded_detail.append((eid, lead, fate, is_hit,
                                             str(a.get("description", ""))[:110],
                                             a.get("trigger_phrases")))
        # --- what v2 newly admits ---
        n2_fwd = 0
        for aid, b in s2.items():
            if b.get("temporality") not in FORWARD:
                continue
            n2_fwd += 1
            a = s1.get(aid)
            if a is None or a.get("temporality") not in FORWARD:
                try:
                    lead = (onset - _parse_date(b["signal_date"])).days
                except Exception:
                    lead = None
                newly.append((eid, lead, bool(lead and 0 < lead <= WINDOW),
                              str(b.get("description", ""))[:110]))

        rows.append((eid, len(s1), n1_fwd, n1_fwd_pre, len(s2), n2_fwd,
                     kept, relabelled, dropped, disc_tp, disc_fp))
        tot.update({"n1_fwd": n1_fwd, "n1_fwd_pre": n1_fwd_pre, "n2_fwd": n2_fwd,
                    "kept": kept, "relabelled": relabelled, "dropped": dropped,
                    "disc_tp": disc_tp, "disc_fp": disc_fp})

    print(f"variant = {args.variant}   v1 = {v1_dir.name}/   v2 = {v2_dir.name}/")
    print("=" * 108)
    print(f"{'event':36s} {'v1fwd':>6s} {'pre':>5s} {'v2fwd':>6s} "
          f"{'kept':>5s} {'relab':>6s} {'drop':>5s} {'disc_TP':>8s} {'disc_FP':>8s}")
    print("-" * 108)
    for (eid, n1, n1f, n1p, n2, n2f, k, r, d, dtp, dfp) in rows:
        print(f"{eid:36s} {n1f:6d} {n1p:5d} {n2f:6d} {k:5d} {r:6d} {d:5d} "
              f"{dtp:8d} {dfp:8d}")
    print("-" * 108)
    print(f"{'TOTAL':36s} {tot['n1_fwd']:6d} {tot['n1_fwd_pre']:5d} "
          f"{tot['n2_fwd']:6d} {tot['kept']:5d} {tot['relabelled']:6d} "
          f"{tot['dropped']:5d} {tot['disc_tp']:8d} {tot['disc_fp']:8d}")

    print()
    print("=" * 108)
    print("VERDICT")
    print("=" * 108)
    n_disc = tot["disc_tp"] + tot["disc_fp"]
    print(f"  v1 forward signals (all dates)          {tot['n1_fwd']}")
    print(f"  of which pre-onset (the paper's window) {tot['n1_fwd_pre']}")
    print(f"  v2 forward signals                      {tot['n2_fwd']}")
    print()
    print(f"  v1 pre-onset signals that v2 kept       "
          f"{tot['n1_fwd_pre'] - n_disc}")
    print(f"  v1 pre-onset signals that v2 discarded  {n_disc}")
    print(f"      of those, entity-matching (a hit)   {tot['disc_tp']}")
    print(f"      of those, non-matching (noise)      {tot['disc_fp']}")
    if n_disc:
        print(f"      -> {tot['disc_tp'] / n_disc:.1%} of discarded pre-onset "
              f"signals were hits")
    print()
    if tot["disc_tp"] > 0:
        print(f"  THE SPLIT DISCARDS {tot['disc_tp']} GENUINE PRECURSORS.")
        print("  That is a precision-recall trade, not a pure defect fix.")
        print("  The manuscript reports the gain (FWGS 0.470 -> 0.535) and not")
        print("  the loss, and the loss is on the recall side.")
    else:
        print("  The split discards no genuine precursors. On this evidence it is")
        print("  a clean fix rather than a trade-off.")

    print()
    print("=" * 108)
    print(f"WHAT WAS DISCARDED (pre-onset; n={len(discarded_detail)})")
    print("=" * 108)
    by_ev = Counter(d[0] for d in discarded_detail)
    for eid, n in by_ev.most_common():
        hits = sum(1 for d in discarded_detail if d[0] == eid and d[3])
        print(f"\n--- {eid}  discarded {n}  (of which hits: {hits}) ---")
        shown = 0
        for e, lead, fate, is_hit, desc, trig in discarded_detail:
            if e != eid or shown >= 5:
                continue
            mark = "HIT " if is_hit else "    "
            print(f"   {mark} lead={lead:4d} {fate:11s} | {desc}")
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
