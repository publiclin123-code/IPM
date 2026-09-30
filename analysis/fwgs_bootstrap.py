"""FWGS bootstrap confidence intervals (per-event + pooled).

Resamples signals with replacement (B=2000), recomputes per-signal FWGS
components, reports mean and 95% CI. Pooled CI uses event-stratified bootstrap
(resample events, then signals within each event) to preserve heterogeneity.
"""
from __future__ import annotations
import json
import math
import random
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
    "red_sea_crisis_2023", "us_chip_export_controls_2022", "renesas_earthquake_2016",
    "port_los_angeles_backlog_2021", "toyota_steel_explosion_2019",
]
DATA_OVERRIDE = {"warehouse_collapse_lithium_2019": DATA / "warehouse_sample_100.jsonl"}
WINDOW = 180
TAU, LAM = 30.0, 0.5
B = 2000
SEED = 42


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_signals(eid: str) -> list[dict]:
    p = RES / f"{eid}_signals.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") != "error":
                out.append(s)
    return out


def load_news(eid: str) -> dict:
    p = DATA_OVERRIDE.get(eid, DATA / f"{eid}.jsonl")
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


def signal_fwgs_component(s: dict, news: dict, gt: list[dict]) -> float:
    """Per-signal FWGS component (no penalty term separated; returns raw score)."""
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


def ci(means: list[float], alpha: float = 0.05) -> tuple:
    means = sorted(means)
    lo = means[int(alpha / 2 * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return lo, hi


def main() -> int:
    import argparse
    global RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default=str(RES),
                    help="results dir; default v1, pass results/v2 for clock-split run")
    ap.add_argument("--events", default=None,
                    help="comma-separated event ids (default: all in EVENTS)")
    args = ap.parse_args()
    RES = Path(args.res_dir)
    events = args.events.split(",") if args.events else EVENTS
    random.seed(SEED)
    gt = load_gt()
    # Precompute per-event component lists
    per_event_components = {}
    for eid in events:
        sigs = load_signals(eid)
        news = load_news(eid)
        comps = []
        for s in sigs:
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            c = signal_fwgs_component(s, news, gt)
            if c is not None:
                comps.append(c)
        if comps:
            per_event_components[eid] = comps

    print("=== FWGS bootstrap (B=2000) ===")
    print(f"{'event':<38} {'FWGS':>7} {'95% CI':>20} {'n':>5}")

    # Per-event bootstrap
    for eid, comps in per_event_components.items():
        obs = sum(comps) / len(comps)
        means = []
        for _ in range(B):
            sample = [random.choice(comps) for _ in range(len(comps))]
            means.append(sum(sample) / len(sample))
        lo, hi = ci(means)
        print(f"{eid:<38} {obs:.3f} [{lo:.3f}, {hi:.3f}] {len(comps):>5}")

    # Pooled: event-stratified bootstrap
    event_ids = list(per_event_components.keys())
    all_comps = [c for eid in event_ids for c in per_event_components[eid]]
    obs_pool = sum(all_comps) / len(all_comps)
    pool_means = []
    for _ in range(B):
        # resample events with replacement, then signals within
        boot = []
        for _ in range(len(event_ids)):
            eid = random.choice(event_ids)
            comps = per_event_components[eid]
            boot.extend(random.choice(comps) for _ in range(len(comps)))
        pool_means.append(sum(boot) / len(boot))
    lo, hi = ci(pool_means)
    print(f"{'POOLED (stratified)':<38} {obs_pool:.3f} [{lo:.3f}, {hi:.3f}] {len(all_comps):>5}")

    # LaTeX
    print("\n% === tab:fwgs_ci === ")
    for eid, comps in per_event_components.items():
        obs = sum(comps) / len(comps)
        means = []
        for _ in range(200):  # lighter for latex print
            sample = [random.choice(comps) for _ in range(len(comps))]
            means.append(sum(sample) / len(sample))
        lo, hi = ci(means)
        e = eid.replace("_", r"\_")
        print(f"{e} & ${obs:.3f}$ & $[{lo:.3f},\\ {hi:.3f}]$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
