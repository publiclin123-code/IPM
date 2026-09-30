"""Innovation 5: Probabilistic warning interpretation.

Maps LLM self-reported confidence to a calibrated posterior P(disruption | warning)
using TS-ECE strata (lead-time-aware reliability), aligning with PRA tradition.

Method:
  1. For each stratum k (lead-time bucket), build the reliability curve
     conf_bin -> empirical P(TP | signal) from label_correctness output.
  2. Since lead_days is only assigned to hits (construction caveat), we compute
     P(TP | conf, lead) by re-running matching for ALL forward/latent signals
     (not just hits), so each signal gets {is_tp, lead} regardless.
  3. Reliability mapping: posterior = P(TP | conf, stratum) via isotonic-free
     binning within each stratum (use bins of width 0.1, pool if sparse).
  4. Report: how confidence translates into posterior per stratum, and the
     gap (over/under-confidence).

Output: LaTeX table (tab:prob_warning).
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
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
    "red_sea_crisis_2023", "us_chip_export_controls_2022", "renesas_earthquake_2016",
    "global_chip_shortage_2021", "port_los_angeles_backlog_2021",
    "toyota_steel_explosion_2019", "warehouse_collapse_lithium_2019",
]
WINDOW = 180
STRATA = [(0, 7), (7, 30), (30, 90), (90, 180)]


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def label_all(signals: list[dict], gt: list[dict]) -> list[dict]:
    """Label EVERY forward/latent signal with is_tp + lead (not just hits)."""
    out = []
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
        out.append({"conf": float(s.get("confidence", 0)), "is_tp": is_tp,
                    "lead": best_lead})
    return out


def stratum_of(lead) -> str:
    if lead is None:
        return "none"
    for lo, hi in STRATA:
        if lo < lead <= hi:
            return f"({lo},{hi}]"
    return "none"


def main() -> int:
    gt = load_gt()
    all_labels = []
    for eid in EVENTS:
        p = RES / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        sigs = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if s.get("status") != "error":
                    sigs.append(s)
        all_labels.extend(label_all(sigs, gt))

    # Stratify by lead bucket
    buckets = defaultdict(list)
    for it in all_labels:
        buckets[stratum_of(it["lead"])].append(it)

    print("=== Reliability mapping (conf -> P(TP)) by lead stratum ===")
    print(f"{'stratum':<12} {'conf>=0.8':>10} {'P(TP)':>7} {'n':>4} | "
          f"{'conf<0.8':>9} {'P(TP)':>7} {'n':>4} | {'conf<0.5':>9} {'P(TP)':>7} {'n':>4}")
    # For latex: per stratum show P(TP | conf>=0.8) etc.
    rows = []
    for key in ["(0,7]", "(7,30]", "(30,90]", "(90,180]", "none"]:
        items = buckets.get(key, [])
        if not items:
            continue
        hi = [i for i in items if i["conf"] >= 0.8]
        lo = [i for i in items if i["conf"] < 0.8]
        lo5 = [i for i in items if i["conf"] < 0.5]
        def pp(lst):
            if not lst:
                return "--", 0
            return f"{sum(1 for i in lst if i['is_tp'])/len(lst):.3f}", len(lst)
        hi_p, hi_n = pp(hi); lo_p, lo_n = pp(lo); lo5_p, lo5_n = pp(lo5)
        print(f"{key:<12} {hi_p:>10} {hi_n:>7} {hi_n:>4} | {lo_p:>9} {lo_n:>7} {lo_n:>4} | {lo5_p:>9} {lo5_n:>7} {lo5_n:>4}")
        rows.append((key, hi_p, hi_n, lo_p, lo_n))

    # LaTeX table
    print("\n% === tab:prob_warning === ")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"stratum & $P(\text{TP}\mid \hat{p}\ge0.8)$ & $n$ & $P(\text{TP}\mid \hat{p}<0.8)$ \\")
    print(r"\midrule")
    for key, hi_p, hi_n, lo_p, lo_n in rows:
        print(f"{key} & {hi_p} & {hi_n} & {lo_p} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Overall gap: mean conf vs mean P(TP)
    n = len(all_labels)
    mean_conf = sum(i["conf"] for i in all_labels) / n
    mean_tp = sum(1 for i in all_labels if i["is_tp"]) / n
    print(f"\nOverall: n={n}, mean_conf={mean_conf:.3f}, P(TP)={mean_tp:.3f}, "
          f"gap={mean_conf - mean_tp:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
