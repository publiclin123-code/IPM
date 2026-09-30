"""Audit the evidence behind every claimed true positive.

Motivation
----------
`validation/validate.py:signal_matches_event` credits a signal as a true
positive if a company name matches, or, in loose mode, if a geography OR a
commodity string matches. Commodity matching is substring-based
(`a in b or b in a`), so a signal listing "steel" matches an event whose
commodity list contains "steel" -- even when the article is about something else
entirely.

That is not hypothetical. The `toyota_steel_explosion_2019` corpus is dominated
by US steel-tariff coverage, and its forward signals are about tariffs, Brexit
and Indian insolvency proceedings, not about a Japanese engine-plant explosion.
They are still counted as hits, because both the signal and the event list
"steel".

This script reports, per event, which criterion credited each true positive, so
that the soft component of the headline precision is visible. It also flags
events whose positives are overwhelmingly commodity-only.

It does not change any number. It measures how much of the number is supported
by company-level (strong) versus commodity-string (weak) evidence.

Usage
-----
  python3 pipeline/audit_match_evidence.py
  python3 pipeline/audit_match_evidence.py --signals-dir results/v2 --out reports/match_evidence.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import _match_any, _company_names, FOREWARD_TEMPORALITIES, _parse_date  # noqa: E402

WINDOW = 180


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def classify(sig: dict, ev: dict) -> str:
    """Return which criterion justifies calling this signal a true positive."""
    comp = _match_any(_company_names(sig), ev.get("companies", []))
    geo = _match_any(sig.get("geographies", []), ev.get("geographies", []))
    com = _match_any(sig.get("commodities", []), ev.get("commodities", []))
    if comp:
        return "company"
    if geo and com:
        return "geo+commodity"
    if com:
        return "commodity_only"
    if geo:
        return "geo_only"
    return "no_match"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit true-positive evidence strength")
    ap.add_argument("--signals-dir", default="results/v2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gt_all = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    sdir = ROOT / args.signals_dir

    rows = []
    totals = Counter()
    print(f"{'event':34s} {'fwd':>4s} {'TP':>4s} {'comp':>5s} {'g+c':>4s} "
          f"{'com_only':>9s} {'geo_only':>9s} {'weak%':>6s}")
    print("-" * 92)

    for ev in gt_all:
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = _parse_date(ev["gt_onset_date"])
        fw = [s for s in load_jsonl(p)
              if s.get("temporality") in FOREWARD_TEMPORALITIES and s.get("signal_date")]
        c = Counter()
        n_tp = 0
        for s in fw:
            try:
                lead = (onset - _parse_date(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            k = classify(s, ev)
            if k == "no_match":
                continue          # a forward signal before onset that does not
                                  # match is a false positive, not a TP
            n_tp += 1
            c[k] += 1
        weak = c["commodity_only"] + c["geo_only"]
        weak_pct = (weak / n_tp * 100) if n_tp else 0.0
        rows.append((eid, len(fw), n_tp, c, weak_pct))
        totals.update(c)
        print(f"{eid:34s} {len(fw):4d} {n_tp:4d} {c['company']:5d} "
              f"{c['geo+commodity']:4d} {c['commodity_only']:9d} "
              f"{c['geo_only']:9d} {weak_pct:5.0f}%")

    grand_tp = sum(r[2] for r in rows)
    print("-" * 92)
    print(f"TP 合计 {grand_tp}  =  company {totals['company']} + "
          f"geo+commodity {totals['geo+commodity']} + "
          f"commodity_only {totals['commodity_only']} + geo_only {totals['geo_only']}")
    weak_all = totals["commodity_only"] + totals["geo_only"]
    print(f"仅靠商品/地区字符串（无公司级证据）的 TP: {weak_all}/{grand_tp} "
          f"= {weak_all / grand_tp:.1%}" if grand_tp else "")
    print()
    print("最高弱证据占比的事件：")
    for eid, nf, ntp, c, wp in sorted(rows, key=lambda r: -r[4])[:6]:
        print(f"  {eid:34s} TP={ntp:3d}  weak={wp:5.1f}%  {dict(c)}")

    if args.out:
        lines = ["# True-positive evidence audit", "",
                 f"Signals from `{args.signals_dir}`, window {WINDOW} days, loose matcher.", "",
                 "`company` = a company name matched (strong).",
                 "`commodity_only` / `geo_only` = only a generic string matched (weak).", "",
                 "| event | forward | TP | company | geo+commodity | commodity_only | geo_only | weak % |",
                 "|---|---|---|---|---|---|---|---|"]
        for eid, nf, ntp, c, wp in rows:
            lines.append(f"| {eid} | {nf} | {ntp} | {c['company']} | {c['geo+commodity']} | "
                         f"{c['commodity_only']} | {c['geo_only']} | {wp:.0f}% |")
        lines += ["", f"**Total TP {grand_tp}**, of which weak (no company evidence): "
                      f"{weak_all} ({weak_all / grand_tp:.1%})." if grand_tp else ""]
        op = ROOT / args.out
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nreport -> {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
