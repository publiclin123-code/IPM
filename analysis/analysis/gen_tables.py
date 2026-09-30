#!/usr/bin/env python3
"""Emit every per-event table body the manuscript needs, from the new cells.

Writes results/tables_q38.json and prints LaTeX-ready rows, so the .tex files can
be filled from generated text rather than from memory.  This is the DI-4 guard
applied to tables: the numbers are produced once, here.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "analysis"))

from identity_matcher import rule_match, load, parse, RULES  # noqa: E402
from event_meta import EVENT_TYPES, EVENT_SHORT  # noqa: E402
from recompute_cell import cell  # noqa: E402

GT = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
FORWARD = {"forward_looking", "latent"}
WINDOW = 180

CELLS = {
    "slug_v1": "results/q38_slug_v1",
    "slug_v2": "results/q38_slug_v2",
    "body_v1": "results/q38_body_v1",
    "body_v2": "results/q38_body_v2",
}


def per_event(signals_dir: Path, rule: str = "R4_identity") -> dict:
    """Per event: n_forward, tp, fp, precision, earliest lead, fwgs."""
    import math
    TAU, LAM = 30.0, 0.5
    out = {}
    for ev in GT:
        eid = ev["event_id"]
        p = signals_dir / f"{eid}_signals.jsonl"
        if not p.exists():
            out[eid] = None
            continue
        onset = parse(ev["gt_onset_date"])
        leads, tps = [], []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            t = rule_match(s, ev, rule)
            leads.append(lead)
            tps.append(t)
        n = len(leads)
        tp = sum(tps)
        # faithfulness g: share of trigger phrases present in the source text
        arts = {}
        base = ROOT / ("data/by_event_body" if "body" in str(signals_dir) else "data/by_event")
        ap = base / f"{eid}.jsonl"
        if ap.exists():
            for line in ap.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("id"):
                    arts[r["id"]] = (r.get("text") or r.get("title") or "")
        s_vals = []
        for s, lead, t in zip(load(p), leads, tps):
            pass
        # recompute with faithfulness
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
            t = rule_match(s, ev, rule)
            ph = s.get("trigger_phrases") or []
            txt = arts.get(s.get("input_id") or "", "")
            g = 1.0 if not ph else (sum(1 for x in ph if str(x).lower() in txt.lower()) / len(ph) if txt else 0.0)
            sigs.append((lead, t, g))
        fw = None
        if sigs:
            fw = sum(((1 - math.exp(-l / TAU)) * g) if t else (-LAM * (1 - g))
                     for l, t, g in sigs) / len(sigs)
        out[eid] = {
            "n": n, "tp": tp, "fp": n - tp,
            "precision": tp / n if n else None,
            "earliest_lead": max(leads) if leads else None,
            "fwgs": fw,
            "type": EVENT_TYPES.get(eid, "?"),
            "short": EVENT_SHORT.get(eid, eid),
        }
    return out


data = {k: per_event(ROOT / v) for k, v in CELLS.items()}
(ROOT / "results" / "tables_q38.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- tab:rq3_hits
prim = data["body_v2"]
print("%" + "=" * 88)
print("% tab:rq3_hits  (full text, clock-split)")
print("%" + "=" * 88)
for ev in GT:
    eid = ev["event_id"]
    r = prim[eid]
    if r is None or r["n"] == 0:
        print(f"{eid:34s} & {EVENT_TYPES.get(eid,'?'):10s} & -- & -- & -- &  0 \\\\")
    else:
        hit = "\\checkmark" if r["tp"] else "--"
        pr = f"{r['precision']:.3f}"
        print(f"{eid:34s} & {r['type']:10s} & {hit} & {r['earliest_lead']:3d} & "
              f"{pr} & {r['n']:2d} \\\\")

# ------------------------------------------------------------------ tab:fwgs
print()
print("%" + "=" * 88)
print("% tab:fwgs  (full text, clock-split)")
print("%" + "=" * 88)
tot_n = tot_tp = tot_fp = 0
fwvals = []
for ev in GT:
    eid = ev["event_id"]
    r = prim[eid]
    if r is None:
        continue
    tot_n += r["n"]; tot_tp += r["tp"]; tot_fp += r["fp"]
    if r["fwgs"] is not None:
        fwvals.append(r["fwgs"] * r["n"] if r["n"] else 0)
    fs = f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--"
    print(f"{eid:34s} & {r['n']:3d} & {r['tp']:3d} & {r['fp']:3d} & {fs} \\\\")
pooled = sum(fwvals) / tot_n if tot_n else None
print(f"\\textbf{{Pooled}} & \\textbf{{{tot_n}}} & \\textbf{{{tot_tp}}} & "
      f"\\textbf{{{tot_fp}}} & $\\mathbf{{{pooled:.3f}}}$ \\\\")

# --------------------------------------------------------------- tab:match
print()
print("%" + "=" * 88)
print("% tab:match  (five rules)")
print("%" + "=" * 88)
for key in ("slug_v2", "body_v2"):
    base = cell(ROOT / CELLS[key], "body" in key)
    print(f"  -- {key}: forward={base['n_forward']}")
    for r in RULES:
        c = cell(ROOT / CELLS[key], "body" in key, rule=r)
        print(f"     {r:20s} TP {c['tp']:5d}  prec {c['precision']:.4f}  "
              f"hit {c['hit_events']}/17  FWGS {c['fwgs']:.4f}")
    print()

# --------------------------------------------------------- tab:body_robust
print("%" + "=" * 88)
print("% body v1 vs v2 per event (for the input-regime section)")
print("%" + "=" * 88)
for ev in GT:
    eid = ev["event_id"]
    a, b = data["body_v1"][eid], data["body_v2"][eid]
    na = a["n"] if a else 0
    nb = b["n"] if b else 0
    tpa = a["tp"] if a else 0
    tpb = b["tp"] if b else 0
    fa = f"{a['fwgs']:.3f}" if a and a["fwgs"] is not None else "--"
    fb = f"{b['fwgs']:.3f}" if b and b["fwgs"] is not None else "--"
    print(f"{eid:34s} & {na:3d} & {tpa:3d} & {fa} & {nb:3d} & {tpb:3d} & {fb} \\\\")
print()
print(f"wrote results/tables_q38.json")
