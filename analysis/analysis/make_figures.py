"""Publication figures 3–5 from existing v1 extraction results.

Fig 3: lead-time distribution by a priori event type
Fig 4: BSCC beta0 naive vs disambiguated
Fig 5: ELR illustration under a deployable confidence threshold

Does not re-run the LLM. Captions in the paper must mark ontology version.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "analysis"))

from figstyle import FIG_DIR, TOL, TYPE_COLOR, TYPE_LABEL, apply_style, save  # noqa: E402
from validate import FOREWARD_TEMPORALITIES, _parse_date, signal_matches_event  # noqa: E402

RES = ROOT / "results"
VAL = ROOT / "validation"
WINDOW = 180
C_FP = 0.05
ELL0 = 60.0
M_MAX_GRID = (0.6, 0.8, 1.0)

# Main-text events (8 + suez designed negative)
EVENT_META = {
    "red_sea_crisis_2023": ("Red Sea", "slow_burn"),
    "us_chip_export_controls_2022": ("Chip controls", "slow_burn"),
    "port_los_angeles_backlog_2021": ("LA port", "creeping"),
    "europe_energy_crisis_2022": ("European energy", "creeping"),
    "renesas_earthquake_2016": ("Renesas quake", "sudden"),
    "toyota_steel_explosion_2019": ("Toyota", "sudden"),
    "covid_supply_disruption_2020": ("COVID-19", "sudden"),
    "renesas_naka_plant_fire_2021": ("Renesas fire", "sudden"),
    "suez_ever_given_2021": ("Suez", "sudden"),
}


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


def warning_rows(gt: list[dict]) -> list[dict]:
    """One row per forward/latent signal in a kept event, with protocol TP flag."""
    gt_by_id = {e["event_id"]: e for e in gt}
    rows = []
    for eid, (short, typ) in EVENT_META.items():
        ev = gt_by_id.get(eid)
        if ev is None:
            continue
        onset = _parse_date(ev["gt_onset_date"])
        for s in load_signals(eid):
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            try:
                sd = _parse_date(s["signal_date"])
            except (KeyError, ValueError):
                continue
            lead = (onset - sd).days
            in_window = 0 < lead <= WINDOW
            is_tp = in_window and signal_matches_event(s, ev, strict=False)
            rows.append({
                "event": eid,
                "short": short,
                "type": typ,
                "lead": lead if in_window else None,
                "is_tp": is_tp,
                "conf": float(s.get("confidence", 0) or 0),
                "severity": int(s.get("severity", 0) or 0),
            })
    return rows


def _kde(vals, xs, bw):
    return np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).mean(axis=1)


def fig3_leadtime(rows: list[dict]) -> None:
    apply_style()
    # collect TP leads per event
    event_leads: dict[str, list[float]] = {}
    short2type: dict[str, str] = {}
    for r in rows:
        short2type[r["short"]] = r["type"]
        if r["is_tp"] and r["lead"] is not None:
            event_leads.setdefault(r["short"], []).append(r["lead"])

    type_order = {"slow_burn": 0, "creeping": 1, "sudden": 2}
    events = sorted(event_leads.keys(),
                    key=lambda s: (type_order.get(short2type.get(s, "sudden"), 3), -len(event_leads[s])))

    fig, ax = plt.subplots(figsize=(5.1, 3.9))
    xgrid = np.linspace(0, 190, 240)
    n = len(events)
    for i, name in enumerate(events):
        leads = np.asarray(event_leads[name], dtype=float)
        typ = short2type[name]
        color = TYPE_COLOR[typ]
        y = n - 1 - i
        if len(leads) > 1:
            bw = 1.06 * leads.std(ddof=1) * len(leads) ** -0.2
            bw = max(bw, 5.0)
            dens = _kde(leads, xgrid, bw)
            dens = dens / dens.max() * 0.85
            ax.fill_between(xgrid, y, y + dens, color=color, alpha=0.32, linewidth=0)
            ax.plot(xgrid, y + dens, color=color, lw=1.2)
        else:
            ax.plot([leads[0]], [y], marker="o", ms=4, color=color)
        med = np.median(leads)
        ax.plot([med, med], [y, y + 0.28], color=color, lw=1.4, solid_capstyle="round")
        ax.text(-10, y, name, ha="right", va="center", fontsize=8)

    ax.axvline(30, color=TOL["grey"], ls=":", lw=0.8, zorder=0)
    ax.axvline(90, color=TOL["grey"], ls=":", lw=0.8, zorder=0)
    ax.text(30, n - 0.15, "30 d", fontsize=6.5, color=TOL["grey"], ha="center")
    ax.text(90, n - 0.15, "90 d", fontsize=6.5, color=TOL["grey"], ha="center")

    ax.set_xlim(-60, 198)
    ax.set_ylim(-0.45, n - 0.15)
    ax.set_yticks([])
    ax.set_xlabel("Lead time of protocol hits (days)")

    # legend for types
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TYPE_COLOR[t], lw=2.5, label=TYPE_LABEL[t])
               for t in ("slow_burn", "creeping", "sudden")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_leadtime")
    plt.close(fig)


def fig4_beta0() -> None:
    """β₀ hindsight-bias figure (delegates to fig4_beta0_2panel.main).

    Numbers verified 2026-08-26 against real signal files via
    validation/metrics.compute_bscc (id2event from data/by_event/_background.jsonl):
      sources: results/_background/signals_postevent_naive_new.jsonl (v1)
               results/_background/signals_postevent_v2_new.jsonl   (v2)
      w=14: v1 β₀=0.476 (82 post, 39 mismatch), v2 β₀=0.214 (84 post, 18 mismatch)
      windows w=7/14/30: v1 0.500/0.476/0.483, v2 0.194/0.214/0.231
      label mix (w=14 strictly-post): v1 = 43 confirm/21 latent/18 forward
                                      v2 = 66 confirm/ 0 latent/18 forward
    The old single-run constants (0.729→0.052, −93%) came from the superseded
    first-pass run (signals_postevent.jsonl) and were removed here.
    """
    import fig4_beta0_2panel
    fig4_beta0_2panel.main()


def mitigation(lead: float, m_max: float, ell0: float) -> float:
    return m_max * (1 - math.exp(-lead / ell0))


def elr_for_theta(rows: list[dict], theta: float, m_max: float, ell0: float = ELL0) -> float:
    """Deployable rule: fire if confidence >= theta. Average ELR across events with signals."""
    costs = []
    for eid in EVENT_META:
        ev_rows = [r for r in rows if r["event"] == eid]
        if not ev_rows:
            continue
        fired = [r for r in ev_rows if r["conf"] >= theta]
        if not fired:
            costs.append(1.0)
            continue
        n_tp = sum(1 for r in fired if r["is_tp"])
        p_hit = n_tp / len(fired)
        tp_leads = [r["lead"] for r in fired if r["is_tp"] and r["lead"]]
        avg_lead = sum(tp_leads) / len(tp_leads) if tp_leads else 0.0
        cost_fire = (1 - p_hit) * C_FP + p_hit * (1 - mitigation(avg_lead, m_max, ell0))
        p_fire = len(fired) / len(ev_rows)
        costs.append(p_fire * cost_fire + (1 - p_fire) * 1.0)
    if not costs:
        return 0.0
    return 1.0 - (sum(costs) / len(costs))


def fig5_elr(rows: list[dict]) -> None:
    apply_style()
    thetas = np.round(np.linspace(0.3, 0.95, 14), 2)
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    base = [elr_for_theta(rows, t, 0.8) * 100 for t in thetas]
    lo = [elr_for_theta(rows, t, 0.6) * 100 for t in thetas]
    hi = [elr_for_theta(rows, t, 1.0) * 100 for t in thetas]
    ax.fill_between(thetas, lo, hi, color=TOL["blue"], alpha=0.18, linewidth=0,
                    label=r"$m_{\max}\in[0.6,1.0]$")
    ax.plot(thetas, base, color=TOL["blue"], lw=1.8, marker="o", ms=3.5,
            label=r"$m_{\max}=0.8$, $\ell_0=60$ d")
    ax.set_xlabel("Deployable threshold (confidence)")
    ax.set_ylabel("ELR (% of baseline loss)")
    ax.set_xlim(0.28, 0.97)
    ax.yaxis.grid(True, ls=":", color=TOL["grey"])
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_elr")
    plt.close(fig)


def main() -> int:
    import argparse
    global RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default=str(RES),
                    help="results dir; default v1, pass results/v2 for clock-split run")
    ap.add_argument("--skip-beta0", action="store_true",
                    help="skip fig4 (post-event pool is ontology-independent)")
    args = ap.parse_args()
    RES = Path(args.res_dir)
    gt = load_gt()
    rows = warning_rows(gt)
    n_tp = sum(1 for r in rows if r["is_tp"])
    print(f"[res-dir={RES}] warning signals={len(rows)} protocol TPs={n_tp}")
    fig3_leadtime(rows)
    if not args.skip_beta0:
        fig4_beta0()
    fig5_elr(rows)
    print(f"figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
