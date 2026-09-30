#!/usr/bin/env python3
"""4-model cross-model FWGS comparison (qwen/gemma/muse/deepseek), single-match.

Produces the LaTeX body for tab:fwgs_cross (4 columns) and the fig7 heatmap data
(4 models), using single-event matching (no cross-event lead inflation).

Usage: python analysis/cross_model_4way.py [--json /tmp/fwgs_4model.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals  # noqa: E402
from metrics import compute_fwgs, load_jsonl  # noqa: E402

GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
MODELS = [
    ("qwen3.6-27b", ROOT / "results" / "v2"),
    ("gemma4-31b", ROOT / "results" / "v2" / "gemma4"),
    ("muse-glimmer-30b", ROOT / "results" / "v2" / "muse"),
    ("deepseek-v4-flash", ROOT / "results" / "deepseek" / "gdelt"),
]


def main():
    ap = json  # noop to keep argparse-free; use --json via sys.argv
    json_out = None
    if "--json" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json") + 1]

    gt = json.load(open(GT, encoding="utf-8"))["events"]
    table = {}
    for ev in gt:
        eid = ev["event_id"]
        news_path = DATA / f"{eid}.jsonl"
        news = {a["id"]: a for a in (load_jsonl(str(news_path)) if news_path.exists() else [])}
        table[eid] = {}
        for mname, mdir in MODELS:
            sp = mdir / f"{eid}_signals.jsonl"
            if not sp.exists():
                table[eid][mname] = None
                continue
            sigs = [s for s in load_signals(str(sp)) if s.get("status") != "error"]
            fw = compute_fwgs(sigs, news, [ev], window_days=180)
            table[eid][mname] = fw["fwgs"]

    # pooled per model (n-weighted)
    pooled = {}
    for mname, _ in MODELS:
        wsum = n = 0.0
        for ev in gt:
            eid = ev["event_id"]
            fw = table[eid][mname]
            if fw is not None:
                # n = forward signal count; approximate via recompute below
                pass
    # simpler: report per-event only; pooled computed by aggregate_events.py

    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"event & qwen3.6-27b & gemma4-31b & muse-glimmer-30b & deepseek-v4-flash \\")
    print(r"\midrule")
    for ev in gt:
        eid = ev["event_id"]
        vals = []
        for mname, _ in MODELS:
            v = table[eid][mname]
            vals.append(f"{v:.3f}" if v is not None else "--")
        print(f"{eid} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    if json_out:
        out = {"models": [m for m, _ in MODELS],
               "rows": [{"eid": ev["event_id"], "fwgs": {m: table[ev["event_id"]][m] for m, _ in MODELS}}
                        for ev in gt]}
        json.dump(out, open(json_out, "w"), ensure_ascii=False, indent=2)
        print(f"\nJSON -> {json_out}")


if __name__ == "__main__":
    main()
