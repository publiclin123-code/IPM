#!/usr/bin/env python3
"""Finalize all remaining appendix tables + fig7 data from completed 3-model results.

Single-match per-event FWGS (no cross-event lead inflation), 18 events, three
models. Produces:
  - tab:fwgs_cross  (3-model FWGS)
  - tab:fwgs_ci     (per-event + pooled bootstrap 95% CI, B=2000)
  - tab:v1v2        (naive v1 vs clock-split v2, qwen)
  - tab:sensitivity (tau/lambda grid, qwen pooled)
  - fig7_heatmap DATA (JSON for fig_heatmap.py)

Usage: python analysis/finalize_tables.py [--json /tmp/finalize.json]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
DATA = ROOT / "data" / "by_event"
sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
from event_meta import EVENT_TYPES, EVENT_SHORT  # noqa: E402

RES_V1 = ROOT / "results"
RES_V2 = ROOT / "results" / "v2"
MODELS = [
    ("qwen3.6-27b", RES_V2),
    ("gemma4-31b", RES_V2 / "gemma4"),
    ("muse-glimmer-30b", RES_V2 / "muse"),
]
WINDOW = 180
TAUS = [15, 30, 60]
LAMBDAS = [0.25, 0.5, 1.0]
TAU, LAM = 30.0, 0.5
B = 2000
SEED = 42


def load_gt():
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_signals(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") != "error":
                out.append(s)
    return out


def load_news(eid):
    p = DATA / f"{eid}.jsonl"
    out = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r.get("id", "")] = r
    return out


def fwgs_single(signals, news, ev, tau=TAU, lam=LAM):
    """Per-event single-match FWGS (no cross-event lead inflation)."""
    scores = []
    n_tp = n_fp = 0
    onset = _parse_date(ev["gt_onset_date"])
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        lead = (onset - sd).days
        is_tp = 0 < lead <= WINDOW and signal_matches_event(s, ev, strict=False)
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        w_f = 1 - math.exp(-lead / tau) if lead > 0 else 0.0
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " " +
                (article.get("text", "") or "")).lower()
        g = (sum(1 for t in triggers if t.lower() in text) / len(triggers)
             if triggers else 0.0)
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - lam * (1 - g) * conf)
    return {"fwgs": sum(scores) / len(scores) if scores else None,
            "n": len(scores), "n_tp": n_tp, "n_fp": n_fp}


def bootstrap_fwgs(signals, news, ev, tau=TAU, lam=LAM, b=B, seed=SEED):
    """Bootstrap 95% CI of per-event FWGS by resampling signals with replacement."""
    rng = random.Random(seed)
    forward = [s for s in signals if s.get("temporality") in FOREWARD_TEMPORALITIES]
    ests = []
    if not forward:
        return (None, None)
    for _ in range(b):
        sample = [forward[rng.randrange(len(forward))] for _ in range(len(forward))]
        r = fwgs_single(sample, news, ev, tau=tau, lam=lam)
        ests.append(r["fwgs"] if r["fwgs"] is not None else 0.0)
    ests.sort()
    lo = ests[int(0.025 * b)]
    hi = ests[int(0.975 * b)]
    return (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    gt = load_gt()
    eids = [e["event_id"] for e in gt]
    gt_by_id = {e["event_id"]: e for e in gt}

    # precompute per-event per-model FWGS + news
    table = {}  # eid -> {model -> result}
    ci = {}     # eid -> {model -> (lo, hi)}
    news_cache = {}
    for eid in eids:
        ev = gt_by_id[eid]
        news_cache[eid] = load_news(eid)
        table[eid] = {}
        ci[eid] = {}
        for mname, mdir in MODELS:
            sigs = load_signals(mdir / f"{eid}_signals.jsonl")
            table[eid][mname] = fwgs_single(sigs, news_cache[eid], ev)
            ci[eid][mname] = bootstrap_fwgs(sigs, news_cache[eid], ev)

    # ---- tab:fwgs_cross ----
    print("% === tab:fwgs_cross (18事件) ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"event & qwen3.6-27b & gemma4-31b & muse-glimmer-30b \\")
    print(r"\midrule")
    for eid in eids:
        vals = []
        for mname, _ in MODELS:
            r = table[eid][mname]
            vals.append(f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--")
        print(f"{eid} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- tab:fwgs_ci ----
    print("\n% === tab:fwgs_ci (18事件, qwen bootstrap) ===")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"event & FWGS & 95\% CI \\")
    print(r"\midrule")
    for eid in eids:
        r = table[eid]["qwen3.6-27b"]
        lo, hi = ci[eid]["qwen3.6-27b"]
        f = f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--"
        c = f"[{lo:.3f},\ {hi:.3f}]" if lo is not None else "--"
        print(f"{eid} & {f} & {c} \\\\")
    # pooled (stratified) bootstrap
    def pooled_boot(model):
        rng = random.Random(SEED)
        ests = []
        rows = [eid for eid in eids if table[eid][model]["fwgs"] is not None]
        for _ in range(B):
            wsum = n = 0.0
            for _ in range(len(rows)):
                eid = rows[rng.randrange(len(rows))]
                wsum += table[eid][model]["fwgs"] * table[eid][model]["n"]
                n += table[eid][model]["n"]
            ests.append(wsum / n if n else 0.0)
        ests.sort()
        return ests[int(0.025 * B)], ests[int(0.975 * B)]
    q_lo, q_hi = pooled_boot("qwen3.6-27b")
    q_pooled = (sum(table[e]["qwen3.6-27b"]["fwgs"] * table[e]["qwen3.6-27b"]["n"]
                    for e in eids if table[e]["qwen3.6-27b"]["fwgs"] is not None)
                / sum(table[e]["qwen3.6-27b"]["n"] for e in eids
                      if table[e]["qwen3.6-27b"]["fwgs"] is not None))
    print(r"\midrule")
    print(r"\textbf{Pooled (stratified)} & \textbf{%.3f} & [%.3f,\ %.3f] \\" %
          (q_pooled, q_lo, q_hi))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- tab:v1v2 ----
    print("\n% === tab:v1v2 (18事件, qwen) ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"event & $n_{\text{sig}}^{v1}$ & FWGS$^{v1}$ & $n_{\text{sig}}^{v2}$ & FWGS$^{v2}$ \\")
    print(r"\midrule")
    total_n1 = total_n2 = 0
    wsum1 = wsum2 = 0.0
    for eid in eids:
        ev = gt_by_id[eid]
        news = news_cache[eid]
        r1 = fwgs_single(load_signals(RES_V1 / f"{eid}_signals.jsonl"), news, ev)
        r2 = table[eid]["qwen3.6-27b"]
        total_n1 += r1["n"]; total_n2 += r2["n"]
        if r1["fwgs"] is not None:
            wsum1 += r1["fwgs"] * r1["n"]
        if r2["fwgs"] is not None:
            wsum2 += r2["fwgs"] * r2["n"]
        f1 = f"{r1['fwgs']:.3f}" if r1["fwgs"] is not None else "--"
        f2 = f"{r2['fwgs']:.3f}" if r2["fwgs"] is not None else "--"
        print(f"{eid} & {r1['n']} & {f1} & {r2['n']} & {f2} \\\\")
    p1 = wsum1 / total_n1 if total_n1 else None
    p2 = wsum2 / total_n2 if total_n2 else None
    print(r"\midrule")
    print(r"\textbf{Pooled ($n$-weighted)} & \textbf{%d} & $\mathbf{%.3f}$ & \textbf{%d} & $\mathbf{%.3f}$ \\" %
          (total_n1, p1 or 0, total_n2, p2 or 0))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- tab:sensitivity (qwen pooled) ----
    print("\n% === tab:sensitivity (18事件, qwen pooled) ===")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"$\tau$ & $\lambda{=}0.25$ & $\lambda{=}0.5$ & $\lambda{=}1.0$ \\")
    print(r"\midrule")
    for tau in TAUS:
        vals = []
        for lam in LAMBDAS:
            wsum = n = 0.0
            for eid in eids:
                ev = gt_by_id[eid]
                r = fwgs_single(load_signals(RES_V2 / f"{eid}_signals.jsonl"),
                                news_cache[eid], ev, tau=tau, lam=lam)
                if r["fwgs"] is not None:
                    wsum += r["fwgs"] * r["n"]; n += r["n"]
            vals.append(f"{wsum / n:.3f}" if n else "--")
        print(f"{tau} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- fig7 DATA JSON ----
    fig7 = {
        "models": [m for m, _ in MODELS],
        "rows": [],
    }
    # order by qwen FWGS desc (events with defined FWGS)
    ordered = sorted([e for e in eids if table[e]["qwen3.6-27b"]["fwgs"] is not None],
                     key=lambda e: -table[e]["qwen3.6-27b"]["fwgs"])
    for eid in ordered:
        row = {"short": EVENT_SHORT.get(eid, eid), "eid": eid, "models": {}}
        for mname, _ in MODELS:
            r = table[eid][mname]
            lo, hi = ci[eid][mname]
            row["models"][mname] = {
                "fwgs": r["fwgs"],
                "ci": [lo, hi] if lo is not None else None,
            }
        fig7["rows"].append(row)
    fig7["pooled_qwen"] = {"fwgs": q_pooled, "ci": [q_lo, q_hi]}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(fig7, f, ensure_ascii=False, indent=2)
        print(f"\nfig7 JSON -> {args.json}")

    # console summary
    print("\n% === per-event (qwen) n/TP/FP/FWGS ===")
    for eid in eids:
        r = table[eid]["qwen3.6-27b"]
        f = f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--"
        print(f"{eid:38s} n={r['n']:3d} TP={r['n_tp']:3d} FP={r['n_fp']:3d} FWGS={f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
