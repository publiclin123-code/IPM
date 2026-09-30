"""Cross-event aggregation for TS-ECE and FWGS (Paper A RQ4).

Reads per-event _metrics.json files, aggregates:
  - TS-ECE: pooled by stratum across events (weighted by n)
  - FWGS: per-event values + simple pooled mean
Outputs LaTeX table body for sec:res-rq4.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Events to include (covid excluded per user directive; suez has 0 articles)
EVENTS = [
    "red_sea_crisis_2023",
    "us_chip_export_controls_2022",
    "renesas_earthquake_2016",
    "global_chip_shortage_2021",
    "port_los_angeles_backlog_2021",
    "toyota_steel_explosion_2019",
    "warehouse_collapse_lithium_2019",
]

STRATA_ORDER = ["(0,7]", "(7,30]", "(30,90]", "(90,180]"]


def load_metrics(eid: str) -> dict:
    p = RES / f"{eid}_metrics.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ts_pooled: dict[str, dict] = defaultdict(lambda: {"n": 0, "ece_sum": 0.0,
                                                      "conf_sum": 0.0, "acc_sum": 0.0,
                                                      "n_with_acc": 0})
    fwgs_rows = []

    print("=== Cross-event aggregation ===")
    print(f"{'event':<38} {'FWGS':>6} {'n_sig':>6} {'TP':>4} {'FP':>4}")
    for eid in EVENTS:
        m = load_metrics(eid)
        fwgs = m.get("fwgs", {})
        if not fwgs:
            print(f"{eid:<38} {'--':>6} {'0':>6} {'0':>4} {'0':>4}  (no metrics)")
            continue
        fw = fwgs.get("fwgs")
        fw_s = f"{fw:.3f}" if fw is not None else "--"
        print(f"{eid:<38} {fw_s:>6} {fwgs.get('n_signals',0):>6} "
              f"{fwgs.get('n_tp',0):>4} {fwgs.get('n_fp',0):>4}")
        if fw is not None:
            fwgs_rows.append((eid, fw, fwgs.get("n_signals", 0)))
        # TS-ECE strata
        curve = m.get("ts_ece", {}).get("ts_ece_curve", [])
        for st in curve:
            key = st.get("stratum")
            if key not in STRATA_ORDER:
                continue
            n = st.get("n") or 0
            ece = st.get("ece")
            ts_pooled[key]["n"] += n
            if ece is not None:
                ts_pooled[key]["ece_sum"] += ece * n
            mc = st.get("mean_conf")
            if mc is not None:
                ts_pooled[key]["conf_sum"] += mc * n
            acc = st.get("accuracy")
            if acc is not None:
                ts_pooled[key]["acc_sum"] += acc * n
                ts_pooled[key]["n_with_acc"] += n

    print("\n=== Pooled TS-ECE by stratum ===")
    print(f"{'stratum':<12} {'n':>5} {'ECE':>6} {'mean_conf':>10} {'acc':>6}")
    for key in STRATA_ORDER:
        c = ts_pooled.get(key)
        if not c or c["n"] == 0:
            print(f"{key:<12} {'0':>5} {'--':>6} {'--':>10} {'--':>6}")
            continue
        ece = c["ece_sum"] / c["n"]
        conf = c["conf_sum"] / c["n"]
        acc = c["acc_sum"] / c["n_with_acc"] if c["n_with_acc"] else None
        acc_s = f"{acc:.3f}" if acc is not None else "--"
        print(f"{key:<12} {c['n']:>5} {ece:.3f} {conf:>10.3f} {acc_s:>6}")

    # FWGS pooled (n-weighted)
    if fwgs_rows:
        tot_n = sum(n for _, _, n in fwgs_rows)
        w_fwgs = sum(fw * n for _, fw, n in fwgs_rows) / tot_n
        print(f"\nPooled FWGS (n-weighted over {tot_n} signals): {w_fwgs:.3f}")

    # ---- LaTeX table body ----
    print("\n% LaTeX table body (tab:rq4_tsfwgs)")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"stratum & n & ECE & mean\,conf & acc \\")
    print(r"\midrule")
    for key in STRATA_ORDER:
        c = ts_pooled.get(key)
        if not c or c["n"] == 0:
            print(f"{key} & 0 & -- & -- & -- \\\\")
            continue
        ece = c["ece_sum"] / c["n"]
        conf = c["conf_sum"] / c["n"]
        acc = c["acc_sum"] / c["n_with_acc"] if c["n_with_acc"] else None
        acc_s = f"{acc:.3f}" if acc is not None else "--"
        print(f"{key} & {c['n']} & {ece:.3f} & {conf:.3f} & {acc_s} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
