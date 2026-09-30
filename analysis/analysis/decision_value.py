"""Innovation 4: Decision Value Model — Expected Loss Reduction (ELR).

Maps the FWGS/lead-time evidence into a monetized early-warning value curve.

Model:
  For each GT event e with severity s_e (1-5), baseline expected loss L_e.
  An early warning with lead time ell reduces loss by a mitigation factor
  m(ell) = m_max * (1 - exp(-ell / ell_0))   [mitigation saturates with lead]
  The system fires a warning if FWGS score > threshold theta.

  Expected cost of the warning system per event:
    E[cost] = p_fire * [ (1-p_hit) * C_fp + p_hit * L_e * (1 - m(ell)) ]
              + (1 - p_fire) * L_e          [missed -> no mitigation]
  where p_fire = P(FWGS > theta), p_hit = P(TP | fire).

  ELR(theta) = E[cost(no system)] - E[cost(theta)] = L_e - E[cost(theta)]

  L_e is normalized to 1 per severity unit; we report ELR in % of baseline loss.

This is intentionally a transparent arithmetic model (no stock prices needed).
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
TAU = 30.0
ELL0 = 60.0          # mitigation saturates ~2 months of lead
M_MAX = 0.8          # max avoidable fraction of loss
C_FP = 0.05          # false-alarm processing cost as fraction of event loss
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


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
    """Return {score, is_tp, lead} for one signal under FWGS."""
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
    score = w_f * g * (1 if is_tp else 0) - 0.5 * (1 - g) * conf
    return {"score": score, "is_tp": is_tp, "lead": lead}


def mitigation(lead: float) -> float:
    return M_MAX * (1 - math.exp(-lead / ELL0))


def main() -> int:
    gt = load_gt()
    # Per-event: list of signal dicts with score/tp/lead
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

    # Baseline: no system -> full loss 1.0 per event (severity normalized)
    n_events = len([e for e in per_event if per_event[e]])
    print("=== ELR by threshold (fraction of baseline loss, pooled across events) ===")
    print(f"{'theta':>6} {'fire%':>7} {'TP/fire':>8} {'avg_lead':>9} {'ELR%':>7} {'rel_to_no_sys':>12}")

    # Aggregate all signals with event tag for pooled analysis
    pooled = []
    for eid, scored in per_event.items():
        for r in scored:
            r["event"] = eid
            pooled.append(r)

    for theta in THETAS:
        # Per-event expected cost, then average across events (each event is
        # an independent decision instance with baseline loss 1.0)
        event_costs = []
        for eid in EVENTS:
            scored = per_event.get(eid, [])
            if not scored:
                continue
            fired = [r for r in scored if r["score"] > theta]
            n_tp = sum(1 for r in fired if r["is_tp"])
            p_hit = n_tp / len(fired) if fired else 0
            avg_lead = sum(r["lead"] for r in fired) / len(fired) if fired else 0
            cost_fire = (1 - p_hit) * C_FP + p_hit * (1 - mitigation(avg_lead)) if fired else 1.0
            p_fire = len(fired) / max(1, len(scored))
            event_costs.append(p_fire * cost_fire + (1 - p_fire) * 1.0)
        avg_cost = sum(event_costs) / len(event_costs)
        elr = 1.0 - avg_cost
        n_fire_tot = sum(len([r for r in per_event[e] if r["score"] > theta]) for e in EVENTS)
        print(f"{theta:>6} {n_fire_tot:>7} {'--':>8} {'--':>9} {elr*100:>7.1f}% {elr:>12.3f}")

    # Per-event ELR at optimal theta=0.3
    print("\n=== Per-event ELR at theta=0.3 ===")
    print(f"{'event':<38} {'n_sig':>5} {'n_fire':>6} {'TP':>3} {'avg_lead':>8} {'ELR%':>6}")
    for eid in EVENTS:
        scored = per_event.get(eid, [])
        if not scored:
            continue
        fired = [r for r in scored if r["score"] > 0.3]
        n_tp = sum(1 for r in fired if r["is_tp"])
        avg_lead = sum(r["lead"] for r in fired) / len(fired) if fired else 0
        p_hit = n_tp / len(fired) if fired else 0
        cost_fire = (1 - p_hit) * C_FP + p_hit * (1 - mitigation(avg_lead)) if fired else 1.0
        p_fire = len(fired) / max(1, len(scored))
        e_cost = p_fire * cost_fire + (1 - p_fire) * 1.0
        elr = 1.0 - e_cost
        print(f"{eid:<38} {len(scored):>5} {len(fired):>6} {n_tp:>3} {avg_lead:>8.0f} {elr*100:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
