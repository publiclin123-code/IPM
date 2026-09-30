"""Three-model FWGS comparison: qwen3.6-27b vs gemma4-31b vs muse-glimmer-30b.

All three models extract the clock-split (v2) ontology over the same full
pre-onset corpus (data/by_event), so the comparison is on identical articles.

8 events = original 5 + covid + europe_energy + renesas_naka.

Outputs: per-event FWGS for each model + pairwise ranking agreement.
"""
from __future__ import annotations
import json
import math
import sys
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
    "port_los_angeles_backlog_2021", "covid_supply_disruption_2020",
    "europe_energy_crisis_2022", "renesas_naka_plant_fire_2021",
]

MODELS = [
    ("qwen3.6-27b", RES / "v2"),
    ("gemma4-31b", RES / "v2" / "gemma4"),
    ("muse-glimmer-30b", RES / "v2" / "muse"),
]

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
    gt = load_gt()

    # per-event FWGS per model
    table = {}  # eid -> {model -> result}
    for eid in EVENTS:
        news = load_news(eid)
        table[eid] = {}
        for mname, mdir in MODELS:
            sigs = load_signals(mdir / f"{eid}_signals.jsonl")
            table[eid][mname] = fwgs_for(sigs, news, gt)

    # console report
    print("=== Three-model FWGS (clock-split v2, full 8-event corpus) ===")
    hdr = f"{'event':<36}" + "".join(f"{m:>14}" for m, _ in MODELS)
    print(hdr)
    for eid in EVENTS:
        cells = []
        for mname, _ in MODELS:
            r = table[eid][mname]
            cells.append(f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--")
        print(f"{eid:<36}" + "".join(f"{c:>14}" for c in cells))
    print()

    # n/TP/FP per model per event (for appendix / table note)
    print("=== (n, TP, FP) per event per model ===")
    for eid in EVENTS:
        parts = []
        for mname, _ in MODELS:
            r = table[eid][mname]
            parts.append(f"{mname}={r['n']}/{r['n_tp']}/{r['n_fp']}")
        print(f"{eid:<36} " + "  ".join(parts))
    print()

    # pooled (n-weighted) FWGS
    print("=== Pooled (n-weighted) FWGS ===")
    for mname, _ in MODELS:
        all_scores = []
        total_n = total_tp = total_fp = 0
        for eid in EVENTS:
            r = table[eid][mname]
            if r["n"]:
                total_n += r["n"]
                total_tp += r["n_tp"]
                total_fp += r["n_fp"]
        print(f"{mname:<16} n={total_n} TP={total_tp} FP={total_fp}")
    print()

    # ranking agreement (events with a defined FWGS, i.e. n>0 and not --)
    def ranked(model):
        rows = [(eid, table[eid][model]["fwgs"]) for eid in EVENTS
                if table[eid][model]["fwgs"] is not None]
        return [eid for eid, _ in sorted(rows, key=lambda x: -x[1])]

    r_qwen = ranked("qwen3.6-27b")
    r_gemma = ranked("gemma4-31b")
    r_muse = ranked("muse-glimmer-30b")
    print("=== FWGS ranking (high to low) ===")
    print(f"  qwen3.6-27b  : {r_qwen}")
    print(f"  gemma4-31b   : {r_gemma}")
    print(f"  muse-glimmer : {r_muse}")
    for a_name, a_rank in [("qwen-gemma", (r_qwen, r_gemma)),
                           ("qwen-muse", (r_qwen, r_muse)),
                           ("gemma-muse", (r_gemma, r_muse))]:
        a, b = a_rank
        n = min(len(a), len(b))
        agree = sum(1 for i in range(n) if a[i] == b[i])
        print(f"  {a_name}: {agree}/{n} exact-position match")

    # LaTeX table body (three models)
    print("\n% === tab:fwgs_3model ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"event & qwen3.6-27b & gemma4-31b & muse-glimmer-30b \\")
    print(r"\midrule")
    for eid in EVENTS:
        vals = []
        for mname, _ in MODELS:
            r = table[eid][mname]
            vals.append(f"{r['fwgs']:.3f}" if r["fwgs"] is not None else "--")
        print(f"{eid} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
