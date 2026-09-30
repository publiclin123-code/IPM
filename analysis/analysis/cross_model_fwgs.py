"""Cross-model FWGS comparison: qwen3.6-27b vs gemma4-31b (both full corpus).

Both models extract the v2 clock-split ontology over the same full pre-onset
corpus (data/by_event), so the comparison is on identical articles.

Outputs: per-event FWGS for each model + ranking check.
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results"
DATA = ROOT / "data" / "by_event"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023", "us_chip_export_controls_2022",
    "renesas_earthquake_2016", "toyota_steel_explosion_2019",
    "port_los_angeles_backlog_2021",
]
QWEN_DIR = RES / "v2"
GEMMA4_DIR = RES / "v2" / "gemma4"
WINDOW = 180
TAU, LAM = 30.0, 0.5


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


def fwgs_for(signals: list[dict], news: dict, gt: list[dict]) -> dict:
    scores = []
    n_tp = n_fp = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        best_lead = None
        is_tp = False
        for ev in gt:
            try:
                onset = _parse_date(ev["gt_onset_date"])
            except (KeyError, ValueError):
                continue
            lead = (onset - sd).days
            if 0 < lead <= WINDOW and signal_matches_event(s, ev, strict=False):
                is_tp = True
                if best_lead is None or lead > best_lead:
                    best_lead = lead
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        lead = best_lead if best_lead is not None else 0
        w_f = 1 - math.exp(-lead / TAU)
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return {"fwgs": sum(scores) / len(scores) if scores else None,
            "n": len(scores), "n_tp": n_tp, "n_fp": n_fp}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwen-dir", default=str(QWEN_DIR), help="qwen v2 signals dir")
    ap.add_argument("--gemma-dir", default=str(GEMMA4_DIR), help="gemma v2 signals dir")
    ap.add_argument("--events", default=None, help="comma-separated event ids")
    args = ap.parse_args()
    qwen_dir = Path(args.qwen_dir)
    gemma_dir = Path(args.gemma_dir)
    events = args.events.split(",") if args.events else EVENTS

    gt = load_gt()
    print("=== Cross-model FWGS (full corpus) ===")
    print(f"{'event':<38} {'qwen3.6-27b':>10} {'(n,tp,fp)':>15} {'gemma4-31b':>10} {'(n,tp,fp)':>15}")
    rows = []
    for eid in events:
        news = load_news(eid)
        sigs_qwen = load_signals(qwen_dir / f"{eid}_signals.jsonl")
        sigs_gemma = load_signals(gemma_dir / f"{eid}_signals.jsonl")
        r_qwen = fwgs_for(sigs_qwen, news, gt)
        r_gemma = fwgs_for(sigs_gemma, news, gt)
        f_qwen = f"{r_qwen['fwgs']:.3f}" if r_qwen["fwgs"] is not None else "--"
        f_gemma = f"{r_gemma['fwgs']:.3f}" if r_gemma["fwgs"] is not None else "--"
        print(f"{eid:<38} {f_qwen:>10} {str((r_qwen['n'],r_qwen['n_tp'],r_qwen['n_fp'])):>15} "
              f"{f_gemma:>10} {str((r_gemma['n'],r_gemma['n_tp'],r_gemma['n_fp'])):>15}")
        if r_qwen["fwgs"] is not None and r_gemma["fwgs"] is not None:
            rows.append((eid, r_qwen["fwgs"], r_gemma["fwgs"]))

    # Rank agreement (positional match on sorted FWGS)
    if len(rows) >= 3:
        r_qwen_rank = sorted(rows, key=lambda x: -x[1])
        r_gemma_rank = sorted(rows, key=lambda x: -x[2])
        qwen_order = [eid for eid, _, _ in r_qwen_rank]
        gemma_order = [eid for eid, _, _ in r_gemma_rank]
        agree = sum(1 for a, b in zip(qwen_order, gemma_order) if a == b)
        print(f"\nRanking agreement: {agree}/{len(rows)} exact position match")
        print(f"  qwen3.6-27b rank: {qwen_order}")
        print(f"  gemma4-31b rank: {gemma_order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
