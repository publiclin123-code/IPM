"""Select and validate the firing threshold (closes the "threshold never chosen" gap).

Why this exists
---------------
The manuscript derives a condition for the *schema* (Section 5.3, c > 0.87) and
evaluates the expected-loss curve over a sweep of thresholds theta, but it never
selects a theta and never checks one out of sample. A system-improvement paper
that supplies a firing rule should supply a firing rule: a chosen threshold, its
performance, and evidence that the choice is not fitted to the events it is
evaluated on.

What is selected
----------------
The pipeline fires when a signal's self-reported confidence reaches theta. Under
the expected-loss model of Section 3.4.3 the operating cost is

    E[cost(theta)] = p_fire * ( (1 - p_hit) C_FP + p_hit (1 - m(l_bar)) )
                     + (1 - p_fire)

with p_fire the probability of at least one signal crossing theta in an event
window, p_hit the precision among fired signals, l_bar the mean lead among fired
hits, and m(l) = m_max (1 - exp(-l / l0)). The selected threshold minimises this
cost.

Why two protocols
-----------------
Choosing theta on the same events used to report it would inflate the result.
So the script reports three things:

  oracle      theta chosen using all events (an upper bound; reported for reference)
  LOEO        theta chosen on n-1 events, applied to the held-out event, pooled.
              This is the honest number.
  fixed       a single theta from the pooled curve, for practitioners who want one
              number rather than a procedure.

Cost parameters are the manuscript's stated preferences (m_max=0.8, l0=60 d,
C_FP=0.05), and the script sweeps C_FP because that parameter is the one a
reader is least likely to share.

Usage
-----
  # needs the ldl_los interpreter (matplotlib present); numpy/scipy needed here
  /home/e/miniconda3/envs/ldl_los/bin/python3 pipeline/select_threshold.py
  /home/e/miniconda3/envs/ldl_los/bin/python3 pipeline/select_threshold.py --variant body
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from identity_matcher import rule_match, load, parse  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
RULE = "R4_identity"
M_MAX, L0 = 0.8, 60.0
C_FP_DEFAULT = 0.05
THETAS = np.round(np.arange(0.30, 0.96, 0.01), 2)


def _resolve_sdir(variant: str, signals_dir: str = ""):
    """Extraction directory to score; --signals-dir overrides the historical default."""
    if signals_dir:
        p = Path(signals_dir)
        return p if p.is_absolute() else ROOT / p
    return ROOT / ("results/v2" if variant == "title" else "results/v2_body")


def collect(variant: str, signals_dir: str = "") -> dict[str, list[dict]]:
    """Per event: every forward signal in the pre-onset window with its confidence."""
    sdir = _resolve_sdir(variant, signals_dir)
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    out: dict[str, list[dict]] = {}
    for ev in gt:
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        sigs = []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            sigs.append({"conf": float(s.get("confidence") or 0.0),
                         "is_tp": rule_match(s, ev, RULE),
                         "lead": lead})
        if sigs:
            out[eid] = sigs
    return out


def expected_cost(streams: dict[str, list[dict]], theta: float, c_fp: float) -> float:
    """Expected cost at one threshold, pooling events."""
    n_events = len(streams)
    if not n_events:
        return float("nan")
    n_fired_events = 0
    fired: list[dict] = []
    for sigs in streams.values():
        hit = [s for s in sigs if s["conf"] >= theta]
        if hit:
            n_fired_events += 1
            fired.extend(hit)
    if not fired:
        return 1.0                       # never fire: cost 1 (no mitigation benefit)
    p_fire = n_fired_events / n_events
    p_hit = sum(1 for s in fired if s["is_tp"]) / len(fired)
    leads = [s["lead"] for s in fired if s["is_tp"]]
    l_bar = float(np.mean(leads)) if leads else 0.0
    m = M_MAX * (1 - math.exp(-l_bar / L0))
    return p_fire * ((1 - p_hit) * c_fp + p_hit * (1 - m)) + (1 - p_fire) * 1.0


def metrics_at(streams: dict[str, list[dict]], theta: float) -> dict:
    fired = [s for sigs in streams.values() for s in sigs if s["conf"] >= theta]
    n_ev_fire = sum(1 for sigs in streams.values() if any(s["conf"] >= theta for s in sigs))
    tp = sum(1 for s in fired if s["is_tp"])
    n = len(fired)
    leads = [s["lead"] for s in fired if s["is_tp"]]
    return {"theta": float(theta), "n_fired": n, "tp": tp, "fp": n - tp,
            "precision": (tp / n) if n else None,
            "events_firing": n_ev_fire,
            "mean_lead": float(np.mean(leads)) if leads else None}


def best_theta(streams, c_fp: float) -> tuple[float, float]:
    costs = [(expected_cost(streams, t, c_fp), t) for t in THETAS]
    cost, theta = min(costs)
    return float(theta), float(cost)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--signals-dir", default="",
                    help="override extraction dir (e.g. results/q38_slug_v1)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    streams = collect(args.variant, args.signals_dir)
    n_sig = sum(len(v) for v in streams.values())
    print("=" * 100)
    print(f"FIRING THRESHOLD SELECTION   variant={args.variant}   rule={RULE}")
    print("=" * 100)
    print(f"  events with signals {len(streams)}   forward signals {n_sig}")
    print(f"  cost model: m_max={M_MAX}, l0={L0:.0f}d, C_FP default={C_FP_DEFAULT}")
    print(f"  theta grid: {THETAS[0]:.2f} .. {THETAS[-1]:.2f} step 0.01 "
          f"({len(THETAS)} values)")
    print(f"  events without signals contribute p_fire=0 and are excluded from streams")
    print()

    # ---- pooled curve ----
    print("=" * 100)
    print("POOLED COST CURVE (all events)")
    print("=" * 100)
    print(f"  {'theta':>6s} {'fired':>6s} {'TP':>5s} {'FP':>5s} {'precision':>10s} "
          f"{'ev fires':>9s} {'mean lead':>10s} {'cost':>8s}")
    curve = []
    for t in THETAS:
        m = metrics_at(streams, float(t))
        c = expected_cost(streams, float(t), C_FP_DEFAULT)
        curve.append({**m, "cost": c})
        if abs((t * 100) % 10) < 0.5 or t in (THETAS[0], THETAS[-1]):
            p = f"{m['precision']:.3f}" if m["precision"] is not None else "  --"
            ml = f"{m['mean_lead']:.1f}" if m["mean_lead"] is not None else "  --"
            print(f"  {t:6.2f} {m['n_fired']:6d} {m['tp']:5d} {m['fp']:5d} {p:>10s} "
                  f"{m['events_firing']:9d} {ml:>10s} {c:8.4f}")

    theta_oracle, cost_oracle = best_theta(streams, C_FP_DEFAULT)
    print()
    print(f"  oracle theta (chosen on all events) = {theta_oracle:.2f}   cost {cost_oracle:.4f}")
    print("  NOTE: this is an upper bound, not a result. It is fitted on the same")
    print("        events it is reported for.")

    # ---- leave-one-event-out ----
    print()
    print("=" * 100)
    print("LEAVE-ONE-EVENT-OUT (the honest number)")
    print("=" * 100)
    print("  For each held-out event: choose theta on the other events, apply it to")
    print("  the held-out one. Pool the fired signals across folds.")
    eids = sorted(streams)
    loeo_fired: list[dict] = []
    chosen = []
    for held in eids:
        rest = {e: s for e, s in streams.items() if e != held}
        if len(rest) < 3:
            continue
        th, _ = best_theta(rest, C_FP_DEFAULT)
        chosen.append(th)
        loeo_fired.extend(s for s in streams[held] if s["conf"] >= th)
    if loeo_fired:
        tp = sum(1 for s in loeo_fired if s["is_tp"])
        n = len(loeo_fired)
        leads = [s["lead"] for s in loeo_fired if s["is_tp"]]
        print()
        print(f"  thresholds chosen per fold: median {np.median(chosen):.2f}, "
              f"range {min(chosen):.2f}..{max(chosen):.2f}")
        print(f"  pooled out-of-fold: fired {n}, TP {tp}, FP {n-tp}, "
              f"precision {tp/n:.3f}" if n else "  no signals fired out of fold")
        if leads:
            print(f"  mean lead among fired hits: {np.mean(leads):.1f} d")
    else:
        print("  no signals fired out of fold")

    # ---- C_FP sensitivity ----
    print()
    print("=" * 100)
    print("SENSITIVITY TO THE FALSE-ALARM COST RATIO")
    print("=" * 100)
    print(f"  {'C_FP':>6s} {'theta*':>7s} {'fired':>6s} {'precision':>10s}")
    sens = {}
    for c in (0.01, 0.02, 0.05, 0.10, 0.20):
        th, _ = best_theta(streams, c)
        m = metrics_at(streams, th)
        sens[c] = {"theta": th, **{k: m[k] for k in ("n_fired", "tp", "fp", "precision")}}
        p = f"{m['precision']:.3f}" if m["precision"] is not None else "  --"
        print(f"  {c:6.2f} {th:7.2f} {m['n_fired']:6d} {p:>10s}")

    # ---- compare with "fire on every signal" and "fire on all forward labels" ----
    print()
    print("=" * 100)
    print("WHAT THE THRESHOLD IS WORTH")
    print("=" * 100)
    allm = metrics_at(streams, 0.0)
    print(f"  fire on every extracted signal (theta=0): fired {allm['n_fired']}, "
          f"precision {allm['precision']:.3f}")
    om = metrics_at(streams, theta_oracle)
    print(f"  oracle theta {theta_oracle:.2f}: fired {om['n_fired']}, "
          f"precision {om['precision']:.3f}")

    if args.out:
        op = ROOT / args.out
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps({
            "variant": args.variant, "rule": RULE,
            "cost_model": {"m_max": M_MAX, "l0": L0, "C_FP": C_FP_DEFAULT},
            "theta_grid": [float(t) for t in THETAS],
            "curve": curve,
            "oracle": {"theta": theta_oracle, "cost": cost_oracle},
            "loeo": {"median_theta": float(np.median(chosen)) if chosen else None,
                     "range": [float(min(chosen)), float(max(chosen))] if chosen else None,
                     "n_fired": len(loeo_fired),
                     "tp": sum(1 for s in loeo_fired if s["is_tp"])},
            "sensitivity": {str(k): v for k, v in sens.items()},
        }, indent=2), encoding="utf-8")
        print(f"\nwritten -> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
