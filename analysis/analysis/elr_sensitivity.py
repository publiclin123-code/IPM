"""ELR parameter sensitivity.

ELR model: mitigation m(ell) = m_max * (1 - exp(-ell/ell_0)); C_FP false-alarm cost.
Grid over m_max in {0.6,0.8,1.0}, ell_0 in {30,60,90}, C_FP in {0.01,0.05,0.10}.
Reports pooled ELR (%) at theta=0.3 under each setting, to show ELR conclusion
is robust to the ad-hoc parameter choices.
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
    "red_sea_crisis_2023", "us_chip_export_controls_2022", "renesas_earthquake_2016",
    "global_chip_shortage_2021", "port_los_angeles_backlog_2021",
    "toyota_steel_explosion_2019", "warehouse_collapse_lithium_2019",
]
DATA_OVERRIDE = {"warehouse_collapse_lithium_2019": DATA / "warehouse_sample_100.jsonl"}
WINDOW = 180
TAU, LAM = 30.0, 0.5
THETA = 0.3

M_MAX_GRID = [0.6, 0.8, 1.0]
ELL0_GRID = [30, 60, 90]
CFP_GRID = [0.01, 0.05, 0.10]


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


def signal_score(s: dict, news: dict, gt: list[dict]) -> dict:
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
    text = ((article.get("title", "") or "") + " " + (article.get("text", "") or "")).lower()
    g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
    conf = float(s.get("confidence", 0))
    return {"score": w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf,
            "is_tp": is_tp, "lead": lead}


def elr_pooled(per_event: dict, m_max: float, ell0: float, cfp: float) -> float:
    def mitig(lead):
        return m_max * (1 - math.exp(-lead / ell0))
    costs = []
    for eid, scored in per_event.items():
        if not scored:
            continue
        fired = [r for r in scored if r["score"] > THETA]
        n_tp = sum(1 for r in fired if r["is_tp"])
        p_hit = n_tp / len(fired) if fired else 0
        avg_lead = sum(r["lead"] for r in fired) / len(fired) if fired else 0
        cost_fire = (1 - p_hit) * cfp + p_hit * (1 - mitig(avg_lead)) if fired else 1.0
        p_fire = len(fired) / max(1, len(scored))
        costs.append(p_fire * cost_fire + (1 - p_fire) * 1.0)
    return (1.0 - sum(costs) / len(costs)) * 100.0


def main() -> int:
    gt = load_gt()
    per_event = {}
    for eid in EVENTS:
        sigs = load_signals(eid)
        news = load_news(eid)
        scored = []
        for s in sigs:
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            r = signal_score(s, news, gt)
            if r is not None:
                scored.append(r)
        per_event[eid] = scored

    print("=== ELR sensitivity (pooled ELR % at theta=0.3) ===")
    print(f"{'m_max':>6} {'ell_0':>6} {'C_FP':>6} {'ELR%':>7}")
    rows = []
    for m_max in M_MAX_GRID:
        for ell0 in ELL0_GRID:
            for cfp in CFP_GRID:
                elr = elr_pooled(per_event, m_max, ell0, cfp)
                rows.append((m_max, ell0, cfp, elr))
                print(f"{m_max:>6} {ell0:>6} {cfp:>6} {elr:>7.1f}")

    # range summary
    vals = [r[3] for r in rows]
    print(f"\nELR range: {min(vals):.1f}% -- {max(vals):.1f}% (baseline m_max=0.8, ell0=60, C_FP=0.05 → 33.6%)")

    # LaTeX: m_max rows x ell_0 cols at C_FP=0.05
    print("\n% === tab:elr_sens (C_FP=0.05) ===")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"$m_{\max}$ & $\ell_0{=}30$ & $\ell_0{=}60$ & $\ell_0{=}90$ \\")
    print(r"\midrule")
    for m_max in M_MAX_GRID:
        cells = []
        for ell0 in ELL0_GRID:
            elr = elr_pooled(per_event, m_max, ell0, 0.05)
            cells.append(f"{elr:.1f}")
        print(f"{m_max} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
