"""I3 execution: retract an unverifiable ground-truth event.

Decision
--------
`toyota_steel_explosion_2019` ("Toyota engine plant steel explosion (Japan)",
onset 2019-06-04) cannot be corroborated. Evidence, from
`pipeline/verify_toyota_event.py` and `verify_onsets.py`:

  - the corpus contains no such event: `toyota` appears in 38 of 456 slugs, and
    **no** title contains both `toyota` and (`explosion`|`fire`)
  - ten Wikipedia searches across the English and Japanese editions return no
    dated 2019 Toyota explosion or fire at a production facility
  - the nearest real analogue is the **1997 Aisin fire** (Aisin is a Toyota
    subsidiary), which is a different event in a different year
  - two real 2019 incidents exist and neither matches the entry: a hydrogen
    station explosion in Sandvika, Norway (June 2019, after which Toyota and
    Hyundai suspended fuel-cell car sales), and a fire at Daihatsu's Nakatsu
    plant on 2019-03-14
  - its 27 recorded hits under the loose matcher were US steel-tariff articles
    (DI-8); under identity-aware matching it retains 1

A ground-truth table is the measuring instrument of an audit paper. An entry
that no source documents is an open invitation to challenge the whole table,
and the entry contributes 1 hit, so nothing in the analysis depends on it.

What this script does
---------------------
Retract, do not delete:

  - writes `validation/gt_events.json` without the event
  - writes `validation/gt_excluded.json` with the full entry and the reason
  - backs up the original to `validation/gt_events.json.pre_retraction`

Then recomputes everything the retraction touches and prints the deltas, so the
cost of the change is visible rather than assumed. Reversal is a copy of the
backup.

Usage
-----
  python3 pipeline/prune_gt_toyota.py --dry-run     # show what would change
  python3 pipeline/prune_gt_toyota.py               # apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "analysis"))

GT = ROOT / "validation" / "gt_events.json"
EXC = ROOT / "validation" / "gt_excluded.json"
BAK = ROOT / "validation" / "gt_events.json.pre_retraction"
TARGET = "toyota_steel_explosion_2019"

REASON = {
    "retracted": True,
    "retracted_on": "2026-09-18",
    "reason": (
        "Onset and event itself unverifiable. Ten Wikipedia searches (en+ja) find no "
        "2019 Toyota engine-plant explosion or fire; the corpus contains no title with "
        "both 'toyota' and ('explosion'|'fire'); the nearest real analogue is the 1997 "
        "Aisin fire. Real 2019 incidents do not match: Sandvika hydrogen station "
        "explosion (Norway, June 2019) and a Daihatsu Nakatsu plant fire (2019-03-14). "
        "Retained hits were US steel-tariff articles under loose matching (DI-8)."
    ),
    "evidence_script": "pipeline/verify_toyota_event.py",
    "hits_under_loose_matching": 27,
    "hits_under_identity_matching": 1,
}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = load(GT)
    events = doc["events"]
    if not any(e["event_id"] == TARGET for e in events):
        print(f"{TARGET} not present in {GT} -- already retracted, nothing to do")
        return 0

    kept = [e for e in events if e["event_id"] != TARGET]
    dropped = next(e for e in events if e["event_id"] == TARGET)

    print("=" * 96)
    print("RETRACTION")
    print("=" * 96)
    print(f"  events before : {len(events)}")
    print(f"  events after  : {len(kept)}")
    print(f"  retracting    : {TARGET}")
    print(f"                  {dropped.get('event_name')}  onset {dropped.get('gt_onset_date')}")
    print(f"                  status {dropped.get('onset_status')}")
    print()
    print("  reason:")
    for line in REASON["reason"].split(". "):
        if line.strip():
            print(f"    - {line.strip().rstrip('.')}.")

    if args.dry_run:
        print("\n  [dry run] no files written")
        return 0

    if not BAK.exists():
        shutil.copy(GT, BAK)
        print(f"\n  backup -> {BAK}")
    else:
        print(f"\n  backup already exists, keeping it -> {BAK}")

    doc["events"] = kept
    doc["n_events"] = len(kept)
    doc.setdefault("revision_log", []).append({
        "date": "2026-09-18",
        "action": "retract",
        "event_id": TARGET,
        "reason": REASON["reason"],
    })
    GT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {GT}  ({len(kept)} events)")

    exc_doc = {}
    if EXC.exists():
        exc_doc = load(EXC)
    exc_doc.setdefault("excluded", [])
    exc_doc["excluded"].append({**dropped, **REASON})
    exc_doc["description"] = ("Ground-truth entries removed from gt_events.json, "
                              "with the evidence for removal. Kept for audit trail; "
                              "do not merge back without new sources.")
    EXC.write_text(json.dumps(exc_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  archived -> {EXC}")

    print()
    print("=" * 96)
    print("DOWNSTREAM EFFECTS")
    print("=" * 96)

    # --- background pool attribution, which drives beta_0 ---
    from metrics import compute_bscc, _build_article_event_map
    bg_articles = ROOT / "data" / "by_event" / "_background.jsonl"
    for label, fn in (("title naive", "signals_postevent_naive_new.jsonl"),
                      ("title v2", "signals_postevent_v2_new.jsonl"),
                      ("body naive", "signals_postevent_naive_body.jsonl"),
                      ("body v2", "signals_postevent_v2_body.jsonl")):
        p = ROOT / "results" / "_background" / fn
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        before = None
        # with the event present (original list) then absent
        for name, evs in (("before", events), ("after", kept)):
            id2e = _build_article_event_map(bg_articles, evs, post_days=60)
            m = compute_bscc(rows, evs, post_days=14, id2event=id2e)
            b = m["beta_0_hindsight_rate"]
            if name == "before":
                before = (b, m["n_strictly_post_onset"], m["n_temporal_mismatch"])
            else:
                after = (b, m["n_strictly_post_onset"], m["n_temporal_mismatch"])
        db = after[0] - before[0]
        print(f"  beta_0 {label:12s} {before[0]:.4f} (n={before[1]:3d}) "
              f"-> {after[0]:.4f} (n={after[1]:3d})   {db:+.4f}")

    # --- event counts ---
    print()
    print(f"  events contributing forward signals: "
          f"{sum(1 for e in events if (ROOT / 'results/v2' / (e['event_id'] + '_signals.jsonl')).exists())}"
          f" -> {sum(1 for e in kept if (ROOT / 'results/v2' / (e['event_id'] + '_signals.jsonl')).exists())}")
    print()
    print("  Now re-run:")
    print("    python3 pipeline/identity_matcher.py --variant title")
    print("    python3 pipeline/identity_matcher.py --variant body")
    print("    python3 pipeline/audit_match_evidence.py")
    print()
    print(f"  To reverse: cp {BAK} {GT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
