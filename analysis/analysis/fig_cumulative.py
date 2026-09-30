"""Fig 8 (new) — Cumulative forward-signal detection curve.

For each event, the empirical CDF of forward-signal report dates (days from
onset). Pooled curve = mean of per-event CDFs (events weighted equally), with a
bootstrap band (B=2000 event resamples). Style mirrors fig5: line + band.

Answers: "how many days before onset has the detector accumulated X% of its
warnings?"  Read qwen detector (results/v2/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))

from figstyle import TOL, apply_style, save  # noqa: E402
from validate import FOREWARD_TEMPORALITIES, _parse_date  # noqa: E402
from event_meta import EVENT_TYPES  # noqa: E402

RES = ROOT / "results" / "v2"
VAL = ROOT / "validation"
WINDOW = 180
B = 2000
SEED = 42

EVENT_META = EVENT_TYPES


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
            if s.get("status") == "error":
                continue
            out.append(s)
    return out


def per_event_dfos(gt_by_id: dict[str, dict]) -> dict[str, np.ndarray]:
    """Forward-signal days-from-onset per event (within [-WINDOW, 0])."""
    out = {}
    for eid in EVENT_META:
        ev = gt_by_id.get(eid)
        if ev is None:
            continue
        onset = _parse_date(ev["gt_onset_date"])
        dfos = []
        for s in load_signals(eid):
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            sd = s.get("signal_date")
            if not sd:
                continue
            try:
                dfo = (_parse_date(sd) - onset).days
            except (ValueError, TypeError):
                continue
            if -WINDOW <= dfo <= 0:
                dfos.append(dfo)
        if dfos:
            out[eid] = np.asarray(sorted(dfos))
    return out


def cdf_at(dfos: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Step CDF: fraction of dfos <= x, evaluated on grid xs."""
    if len(dfos) == 0:
        return np.zeros_like(xs, dtype=float)
    counts = np.searchsorted(dfos, xs, side="right")
    return counts / len(dfos)


def main() -> int:
    apply_style()
    gt_by_id = {e["event_id"]: e for e in load_gt()}
    per_event = per_event_dfos(gt_by_id)
    xs = np.arange(-WINDOW, 1)

    fig, ax = plt.subplots(figsize=(3.7, 3.3))

    # per-event faint step curves
    for eid, dfos in per_event.items():
        ax.plot(xs, cdf_at(dfos, xs), color=TOL["grey"], lw=0.7, alpha=0.5)

    # pooled = mean of per-event CDFs, with bootstrap band
    keys = list(per_event.keys())
    cdfs = np.array([cdf_at(per_event[k], xs) for k in keys])
    pooled = cdfs.mean(axis=0)

    rng = np.random.default_rng(SEED)
    boot = np.empty((B, len(xs)), dtype=float)
    for b in range(B):
        idx = rng.integers(0, len(keys), size=len(keys))
        boot[b] = cdfs[idx].mean(axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    ax.fill_between(xs, lo, hi, color=TOL["blue"], alpha=0.20, lw=0,
                    label="Pooled 95% CI")
    ax.plot(xs, pooled, color=TOL["blue"], lw=1.9, label="Pooled (event-avg)")

    # reference lines
    ax.axhline(0.5, color=TOL["grey"], ls=":", lw=0.8)
    ax.axhline(0.8, color=TOL["grey"], ls=":", lw=0.8)
    # median detection day
    med_idx = int(np.searchsorted(pooled, 0.5))
    med_day = xs[med_idx] if med_idx < len(xs) else xs[-1]
    ax.plot([med_day], [0.5], marker="o", ms=4.5, color=TOL["red"],
            mec="white", mew=0.6, zorder=5)
    ax.annotate(f"50% detected\n{med_day} d", xy=(med_day, 0.5),
                xytext=(med_day + 25, 0.30), fontsize=7, color=TOL["red"],
                arrowprops=dict(arrowstyle="-", color=TOL["red"], lw=0.7))

    ax.set_xlim(-WINDOW, 2)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Days from onset  (negative = pre-event)")
    ax.set_ylabel("Cumulative share of forward signals")
    ax.set_xticks([-180, -150, -120, -90, -60, -30, 0])
    ax.yaxis.grid(True, ls=":", color=TOL["grey"])
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig8_cumulative")
    plt.close(fig)
    print(f"cumulative: {len(keys)} events with forward signals, "
          f"pooled 50% at {med_day} d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
