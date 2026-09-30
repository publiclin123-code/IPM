#!/usr/bin/env python3
"""修正版聚合: 三模型 cross-model FWGS + v1v2 对比 (每事件单匹配, 18事件).

修复三个问题:
1. three_model_fwgs.py / v1v2_fwgs.py 用全 GT 匹配 → 跨事件 lead 放大 (bug)
2. EVENTS 硬编码 8 事件 → 改为读 GT 全部 18 事件
3. 每事件只对自身匹配, 避免 lead 虚增

用法:
  python analysis/aggregate_cross_model.py            # 打印三模型表 + v1v2 表
  python analysis/aggregate_cross_model.py --out /tmp/xmodel.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
DATA = ROOT / "data" / "by_event"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

WINDOW = 180
TAU, LAM = 30.0, 0.5

MODELS = [
    ("qwen3.6-27b", ROOT / "results" / "v2"),
    ("gemma4-31b", ROOT / "results" / "v2" / "gemma4"),
    ("muse-glimmer-30b", ROOT / "results" / "v2" / "muse"),
]
RES_V1 = ROOT / "results"          # v1 naive 信号 (results/{eid}_signals.jsonl)
RES_V2 = ROOT / "results" / "v2"   # v2 clock-split 信号


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_signals(path: Path) -> list[dict]:
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


def load_news(eid: str) -> dict:
    p = DATA / f"{eid}.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("id", "")] = r
    return out


def fwgs_single(signals: list[dict], news: dict, ev: dict) -> dict:
    """每事件单匹配 (只对自身 ev) 的 FWGS, 修复跨事件 lead 放大."""
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
        w_f = 1 - math.exp(-lead / TAU) if lead > 0 else 0.0
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = (sum(1 for t in triggers if t.lower() in text) / len(triggers)
             if triggers else 0.0)
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return {"fwgs": sum(scores) / len(scores) if scores else None,
            "n": len(scores), "n_tp": n_tp, "n_fp": n_fp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gt = load_gt()
    eids = [e["event_id"] for e in gt]

    # ---- 三模型 cross-model ----
    print("=" * 80)
    print("三模型 FWGS (clock-split v2, 每事件单匹配, 18事件)")
    print("=" * 80)
    table = {}  # eid -> {model -> result}
    for eid in eids:
        ev = next(e for e in gt if e["event_id"] == eid)
        news = load_news(eid)
        table[eid] = {}
        for mname, mdir in MODELS:
            sigs = load_signals(mdir / f"{eid}_signals.jsonl")
            table[eid][mname] = fwgs_single(sigs, news, ev)

    hdr = f"{'event':<36}" + "".join(f"{m:>14}" for m, _ in MODELS)
    print(hdr)
    print("-" * len(hdr))
    for eid in eids:
        cells = []
        for mname, _ in MODELS:
            r = table[eid][mname]
            cells.append(f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--")
        print(f"{eid:<36}" + "".join(f"{c:>14}" for c in cells))

    # pooled
    print("\n=== Pooled (n-weighted) FWGS ===")
    for mname, _ in MODELS:
        total_n = total_tp = total_fp = 0
        for eid in eids:
            r = table[eid][mname]
            total_n += r["n"]
            total_tp += r["n_tp"]
            total_fp += r["n_fp"]
        print(f"{mname:<16} n={total_n} TP={total_tp} FP={total_fp}")

    # LaTeX cross-model 表
    print("\n% === tab:fwgs_cross (18事件) ===")
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

    # ---- v1v2 对比 ----
    print("\n" + "=" * 80)
    print("v1 (naive) vs v2 (clock-split) FWGS 对比 (每事件单匹配, 18事件)")
    print("=" * 80)
    print(f"{'event':<36} {'n_v1':>6} {'FWGS_v1':>9} {'n_v2':>6} {'FWGS_v2':>9}")
    v1v2_rows = []
    total_n1 = total_n2 = 0
    for eid in eids:
        ev = next(e for e in gt if e["event_id"] == eid)
        news = load_news(eid)
        sigs_v1 = load_signals(RES_V1 / f"{eid}_signals.jsonl")
        sigs_v2 = load_signals(RES_V2 / f"{eid}_signals.jsonl")
        r1 = fwgs_single(sigs_v1, news, ev)
        r2 = fwgs_single(sigs_v2, news, ev)
        total_n1 += r1["n"]
        total_n2 += r2["n"]
        v1v2_rows.append((eid, r1, r2))
        f1 = f"{r1['fwgs']:.3f}" if r1["fwgs"] is not None else "--"
        f2 = f"{r2['fwgs']:.3f}" if r2["fwgs"] is not None else "--"
        print(f"{eid:<36} {r1['n']:>6} {f1:>9} {r2['n']:>6} {f2:>9}")

    # pooled v1v2
    def pooled_fwgs(rows, key):
        wsum = sum((r[key]["fwgs"] or 0.0) * r[key]["n"] for _, r1, r2 in rows
                   for r in (r1, r2) if False)  # placeholder, replaced below
        # 实际: rows 是 (eid, r1, r2)
        wsum = 0.0
        n = 0
        for _eid, r1, r2 in rows:
            r = r1 if key == 0 else r2
            if r["fwgs"] is not None:
                wsum += r["fwgs"] * r["n"]
                n += r["n"]
        return wsum / n if n else None
    p1 = pooled_fwgs(v1v2_rows, 0)
    p2 = pooled_fwgs(v1v2_rows, 1)
    f1s = f"{p1:.3f}" if p1 is not None else "--"
    f2s = f"{p2:.3f}" if p2 is not None else "--"
    print(f"\n{'Pooled (n-weighted)':<36} {total_n1:>6} {f1s:>9} {total_n2:>6} {f2s:>9}")

    if args.out:
        out = {"cross_model": {eid: {m: table[eid][m] for m, _ in MODELS}
                               for eid in eids},
               "v1v2": {eid: {"v1": r1, "v2": r2}
                        for eid, r1, r2 in v1v2_rows}}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
