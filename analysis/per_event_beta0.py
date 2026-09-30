"""Per-event beta_0 breakdown for the post-event BSCC probe.

Re-tags the 41 post-event articles with event_id by URL keyword matching
(fetch_background.py run_post_event dropped this field), then groups the 18
signals by event and computes beta_0 per event + overall.

Outputs:
  - stderr: human-readable per-event table + aggregate
  - stdout: LaTeX table body for sec:res-rq4 (beta_0 column)
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "validation" / "gt_events.json"
KW_PATH = ROOT / "pipeline" / "event_keywords.json"
BG_PATH = ROOT / "data" / "by_event" / "_background.jsonl"
SIG_PATH = ROOT / "results" / "_background" / "signals_postevent.jsonl"

POST_DAYS = 14
FWD_TEMPORALITIES = {"forward_looking", "latent"}


def load_gt_onsets() -> dict[str, str]:
    """event_id -> gt_onset_date (YYYY-MM-DD)."""
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    events = gt.get("events", gt) if isinstance(gt, dict) else gt
    return {ev["event_id"]: ev["gt_onset_date"][:10] for ev in events}


def load_keywords() -> dict[str, list[str]]:
    """event_id -> lowercased url_keywords."""
    with open(KW_PATH, encoding="utf-8") as f:
        kw = json.load(f)
    return {eid: [k.lower() for k in (cfg.get("url_keywords", []) if isinstance(cfg, dict) else [])]
            for eid, cfg in kw.items()}


def tag_articles_by_date() -> dict[str, str]:
    """article_id -> event_id via article-date containment in onset window.

    Reliable attribution: fetch_background.py --mode post-event fetches each
    event's [onset, onset+post_days] window separately, so every article's
    `date` field falls in exactly one event's window. URL-keyword matching is
    NOT reliable (suez/red_sea keyword collision, broad chip terms).
    """
    from datetime import datetime, timedelta
    raw = {}
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    events = gt.get("events", gt) if isinstance(gt, dict) else gt
    windows = []  # (onset_dt, end_dt, event_id)
    for ev in events:
        try:
            onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
            windows.append((onset, onset + timedelta(days=POST_DAYS), ev["event_id"]))
        except (KeyError, ValueError):
            continue

    id2event: dict[str, str] = {}
    with open(BG_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            try:
                ad = datetime.strptime(rec["date"][:10], "%Y-%m-%d")
            except (KeyError, ValueError):
                id2event[rec["id"]] = "UNMATCHED"
                continue
            matched = None
            for onset, end, eid in windows:
                if onset <= ad <= end:
                    matched = eid
                    break
            id2event[rec["id"]] = matched or "UNMATCHED"
    return id2event


def main() -> int:
    onsets = load_gt_onsets()
    id2event = tag_articles_by_date()

    # Article counts per event (sanity check)
    art_counts: dict[str, int] = defaultdict(int)
    for ev in id2event.values():
        art_counts[ev] += 1

    # Load signals, attach event_id via input_id, partition by temporality
    from datetime import datetime, timedelta
    signals = []
    with open(SIG_PATH, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            s["_event_id"] = id2event.get(s.get("input_id", ""), "UNMATCHED")
            signals.append(s)

    # Per-event partition: strictly_post vs onset_day
    per_event: dict[str, dict] = defaultdict(
        lambda: {"n_strictly_post": 0, "mismatch": 0, "correct": 0,
                 "n_onset_day": 0, "onset_mismatch": 0, "unmatched": False}
    )

    for s in signals:
        eid = s["_event_id"]
        cell = per_event[eid]
        if eid == "UNMATCHED":
            cell["unmatched"] = True
            continue
        onset_str = onsets.get(eid)
        if not onset_str:
            continue
        try:
            onset = datetime.strptime(onset_str, "%Y-%m-%d")
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        is_mismatch = s.get("temporality") in FWD_TEMPORALITIES
        if sd > onset and sd <= onset + timedelta(days=POST_DAYS):
            cell["n_strictly_post"] += 1
            if is_mismatch:
                cell["mismatch"] += 1
            else:
                cell["correct"] += 1
        elif sd == onset:
            cell["n_onset_day"] += 1
            if is_mismatch:
                cell["onset_mismatch"] += 1

    # Aggregate
    tot_post = sum(c["n_strictly_post"] for c in per_event.values())
    tot_mismatch = sum(c["mismatch"] for c in per_event.values())
    tot_onset = sum(c["n_onset_day"] for c in per_event.values())
    tot_onset_mismatch = sum(c["onset_mismatch"] for c in per_event.values())
    beta0 = tot_mismatch / tot_post if tot_post else float("nan")

    # stderr report
    print("\n=== Per-event beta_0 (post-event BSCC probe) ===", file=sys.stderr)
    print(f"{'event_id':<38} {'n_post':>6} {'mism':>4} {'corr':>4} {'b0':>5} {'onset':>5}",
          file=sys.stderr)
    for eid in sorted(per_event.keys()):
        c = per_event[eid]
        if c["n_strictly_post"] == 0 and c["n_onset_day"] == 0 and not c["unmatched"]:
            continue
        b0 = c["mismatch"] / c["n_strictly_post"] if c["n_strictly_post"] else float("nan")
        b0_str = f"{b0:.2f}" if c["n_strictly_post"] else "  -"
        print(f"{eid:<38} {c['n_strictly_post']:>6} {c['mismatch']:>4} "
              f"{c['correct']:>4} {b0_str:>5} {c['n_onset_day']:>5}", file=sys.stderr)
    print(f"{'-'*70}", file=sys.stderr)
    print(f"{'AGGREGATE':<38} {tot_post:>6} {tot_mismatch:>4} "
          f"{tot_post-tot_mismatch:>4} {beta0:.3f} {tot_onset:>5}", file=sys.stderr)
    print(f"\nArticle pool: {sum(art_counts.values())} articles across "
          f"{len([k for k in art_counts if k != 'UNMATCHED'])} events", file=sys.stderr)
    if art_counts.get("UNMATCHED"):
        print(f"  WARNING: {art_counts['UNMATCHED']} articles unmatched", file=sys.stderr)

    # stdout: LaTeX table body
    # Columns: event_id & n_articles & n_signals & n_strictly_post & mismatch & beta_0
    print("% Auto-generated by analysis/per_event_beta0.py")
    print("% beta_0 per GT event from post-event BSCC probe (onset+1d .. onset+14d)")
    sig_per_event: dict[str, int] = defaultdict(int)
    for s in signals:
        sig_per_event[s["_event_id"]] += 1
    for eid in sorted(per_event.keys()):
        c = per_event[eid]
        if eid == "UNMATCHED":
            continue
        if c["n_strictly_post"] == 0 and c["n_onset_day"] == 0:
            continue
        b0 = c["mismatch"] / c["n_strictly_post"] if c["n_strictly_post"] else float("nan")
        b0_str = f"{b0:.2f}" if c["n_strictly_post"] else "--"
        # short event label
        label = eid.replace("_", r"\_")
        print(f"{label} & {art_counts.get(eid,0)} & {sig_per_event.get(eid,0)} & "
              f"{c['n_strictly_post']} & {c['mismatch']} & ${b0_str}$ \\\\")
    print(r"\midrule")
    tot_art = sum(art_counts.get(e, 0) for e in per_event if e != "UNMATCHED")
    print(f"{{\\bf Aggregate}} & {tot_art} & {len(signals)} & {tot_post} & "
          f"{tot_mismatch} & $\\mathbf{{{beta0:.3f}}}$ \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
