"""Three-model FWGS bootstrap confidence intervals (per-event + pooled).

Resamples signals with replacement (B=2000) per model, per event.
Pooled CI uses event-stratified bootstrap to preserve heterogeneity.
"""
from __future__ import annotations
import json
import math
import random
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
B = 2000
SEED = 42


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


def component(s: dict, news: dict, gt: list[dict]):
    try:
        sd = _parse_date(s["signal_date"])
    except (KeyError, ValueError):
        return None
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
    lead = best_lead if best_lead is not None else 0
    w_f = 1 - math.exp(-lead / TAU)
    triggers = s.get("trigger_phrases", []) or []
    article = news.get(s.get("input_id", ""), {})
    text = ((article.get("title", "") or "") + " "
            + (article.get("text", "") or "")).lower()
    g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
    conf = float(s.get("confidence", 0))
    return w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf


def ci(means, alpha=0.05):
    means = sorted(means)
    lo = means[int(alpha / 2 * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return lo, hi


def main() -> int:
    random.seed(SEED)
    gt = load_gt()

    # per model: per-event component lists
    results = {}  # model -> {eid: [comp...]}
    for mname, mdir in MODELS:
        per_event = {}
        for eid in EVENTS:
            sigs = load_signals(mdir / f"{eid}_signals.jsonl")
            news = load_news(eid)
            comps = []
            for s in sigs:
                if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                    continue
                c = component(s, news, gt)
                if c is not None:
                    comps.append(c)
            if comps:
                per_event[eid] = comps
        results[mname] = per_event

    # console: per-event per-model FWGS + CI
    print("=== Three-model FWGS bootstrap (B=2000, stratified pooled) ===")
    for mname, per_event in results.items():
        print(f"\n--- {mname} ---")
        print(f"{'event':<36} {'FWGS':>7} {'95% CI':>20} {'n':>5}")
        for eid in EVENTS:
            comps = per_event.get(eid)
            if not comps:
                print(f"{eid:<36} {'--':>7} {'--':>20} {0:>5}")
                continue
            obs = sum(comps) / len(comps)
            means = [sum(random.choice(comps) for _ in range(len(comps))) / len(comps)
                     for _ in range(B)]
            lo, hi = ci(means)
            print(f"{eid:<36} {obs:.3f} [{lo:.3f}, {hi:.3f}] {len(comps):>5}")
        # pooled stratified
        eids = list(per_event.keys())
        all_comps = [c for e in eids for c in per_event[e]]
        obs = sum(all_comps) / len(all_comps)
        pm = []
        for _ in range(B):
            boot = []
            for _ in range(len(eids)):
                comps = per_event[random.choice(eids)]
                boot.extend(random.choice(comps) for _ in range(len(comps)))
            pm.append(sum(boot) / len(boot))
        lo, hi = ci(pm)
        print(f"{'POOLED (stratified)':<36} {obs:.3f} [{lo:.3f}, {hi:.3f}] {len(all_comps):>5}")

    # LaTeX table: three-model per-event FWGS [CI]
    print("\n% === tab:fwgs_3model_ci ===")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"event & qwen3.6-27b & gemma4-31b & muse-glimmer-30b \\")
    print(r"\midrule")
    for eid in EVENTS:
        cells = []
        for mname, per_event in results.items():
            comps = per_event.get(eid)
            if not comps:
                cells.append("--")
                continue
            obs = sum(comps) / len(comps)
            means = [sum(random.choice(comps) for _ in range(len(comps))) / len(comps)
                     for _ in range(B)]
            lo, hi = ci(means)
            cells.append(f"${obs:.3f}$ $[{lo:.3f},{hi:.3f}]$")
        e = eid.replace("_", r"\_")
        print(f"{e} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
