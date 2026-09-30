#!/usr/bin/env python3
"""Plan B todo 8: title-only vs body 提取对比表.

对每个 GT 事件对比 results/v2 (title-only) 与 results/v2_body (body):
  - 信号数 n_sig / TP / FP / precision / FWGS / hit / lead  (来自各自 _aggregate JSON, 没有则现场跑)
  - trigger faithfulness (来自各自 *_metrics.json)
输出: results/compare_title_body.json + 控制台对照表
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
VB = ROOT / "results" / "v2_body"

def ensure_aggregate(res_dir: Path, out_name: str, data_dir: str | None = None) -> dict:
    out = res_dir / out_name
    if not out.exists():
        cmd = [sys.executable, str(ROOT / "pipeline" / "aggregate_events.py"),
               "--res-dir", str(res_dir), "--out", str(out)]
        if data_dir:
            cmd += ["--data-dir", data_dir]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"aggregate failed for {res_dir}")
    return json.load(open(out, encoding="utf-8"))

def load_faith(res_dir: Path) -> dict[str, float | None]:
    """eid -> faithfulness_rate (metrics.json 里可能缺)"""
    out = {}
    for p in res_dir.glob("*_metrics.json"):
        eid = p.name.replace("_metrics.json", "")
        try:
            d = json.load(open(p, encoding="utf-8"))
            f = d.get("faithfulness") or {}
            out[eid] = f.get("faithfulness_rate")
        except Exception:
            out[eid] = None
    return out

def f3(x):
    return "--" if x is None else f"{x:.3f}"

def main():
    A = ensure_aggregate(V2, "_aggregate_18events.json")              # title-only
    B = ensure_aggregate(VB, "_aggregate_18events.json",
                         data_dir="data/by_event_body")                # body (g 需对正文算)
    fa, fb = load_faith(V2), load_faith(VB)

    a_by = {r["event_id"]: r for r in A["per_event"]}
    b_by = {r["event_id"]: r for r in B["per_event"]}

    hdr = f"{'event':34s} | {'sig T/B':>9s} {'prec T/B':>13s} {'FWGS T/B':>13s} {'hit T/B':>7s} {'lead T/B':>9s} {'faith T/B':>13s}"
    print(hdr); print("-" * len(hdr))
    comp = []
    for eid in sorted(a_by):
        a, b = a_by[eid], b_by.get(eid, {})
        row = {
            "event_id": eid,
            "n_sig_title": a["n_sig"], "n_sig_body": b.get("n_sig"),
            "tp_title": a["tp"], "tp_body": b.get("tp"),
            "fp_title": a["fp"], "fp_body": b.get("fp"),
            "prec_title": a["prec"], "prec_body": b.get("prec"),
            "fwgs_title": a["fwgs"], "fwgs_body": b.get("fwgs"),
            "hit_title": a["hit"], "hit_body": b.get("hit"),
            "lead_title": a["earliest_lead_days"], "lead_body": b.get("earliest_lead_days"),
            "faith_title": fa.get(eid), "faith_body": fb.get(eid),
        }
        comp.append(row)
        print(f"{eid:34s} | {a['n_sig']:>4d}/{b.get('n_sig',0):<4d} "
              f"{f3(a['prec']):>6}/{f3(b.get('prec')):<6} "
              f"{f3(a['fwgs']):>6}/{f3(b.get('fwgs')):<6} "
              f"{'Y' if a['hit'] else '-':>2}/{'Y' if b.get('hit') else '-':<2} "
              f"{str(a['earliest_lead_days']):>4}/{str(b.get('earliest_lead_days')):<4} "
              f"{f3(fa.get(eid)):>6}/{f3(fb.get(eid)):<6}")

    print("-" * len(hdr))
    print(f"POOLED prec: {f3(A['aggregate_precision'])} -> {f3(B['aggregate_precision'])} | "
          f"FWGS: {f3(A['pooled_fwgs'])} -> {f3(B['pooled_fwgs'])} | "
          f"n_sig: {A['total_forward_signals']} -> {B['total_forward_signals']} | "
          f"TP {A['total_tp']}->{B['total_tp']} FP {A['total_fp']}->{B['total_fp']}")

    hits_a = sum(1 for r in A["per_event"] if r["hit"])
    hits_b = sum(1 for r in B["per_event"] if r["hit"])
    print(f"hit events: {hits_a}/{A['n_events']} -> {hits_b}/{B['n_events']}")

    out = {"title_only": A, "body": B, "per_event_compare": comp,
           "faith_title": fa, "faith_body": fb}
    dst = ROOT / "results" / "compare_title_body.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nJSON -> {dst}")

if __name__ == "__main__":
    main()
