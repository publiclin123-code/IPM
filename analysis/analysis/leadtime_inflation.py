"""Lead-time inflation: how much phantom lead the naive ontology adds.

Mechanism (see main.tex sec:res-bscc): the naive three-way temporality field
labels a realized trigger with unresolved impact ("vessel hijacked, shipping may
yet be disrupted") as `latent`/`forward_looking`. A decision maker reading that
signal believes they still have lead time, but the trigger has already occurred;
they are, in fact, already behind.

The naive ontology therefore measures lead time to *impact*, not to *trigger*.
The systematic inflation = the trigger -> impact propagation lag, which the naive
field silently bundles into "lead time".

Estimator (fully grounded in the existing post-event negative-control pool):
  * Each strictly post-event signal at report date = onset + delta records
    whether, delta days after the trigger, impact is already resolved
    (confirmation) or still unresolved (latent/forward = "warning").
  * S(delta) = P(impact still unresolved at delta days after trigger), estimated
    pointwise from reports at that delta.
  * Mean propagation lag (censored at the 14-day post window) = sum_delta S(delta).
    This is the phantom lead time added by the naive ontology.

We compare the naive (v1) and clock-split (v2) ontologies. Clock-split should
drive S(delta) ~ 0, i.e. remove the inflation.

Zero third-party deps beyond matplotlib (optional figure).
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results"

sys.path.insert(0, str(VAL))
from metrics import _build_article_event_map, FOREWARD_TEMPORALITIES  # noqa: E402

GT_PATH = VAL / "gt_events.json"
BG_ARTICLES = ROOT / "data" / "by_event" / "_background.jsonl"
NAIVE_SIGNALS = RES / "_background" / "signals_postevent_naive_new.jsonl"    # v1 naive (18-event pool)
CLOCKSPLIT_SIGNALS = RES / "_background" / "signals_postevent_v2_new.jsonl"  # v2 clock-split (18-event pool)

POST_DAYS = 14
M_MAX = 0.8     # mitigation ceiling (matches fig5)
ELL0 = 60.0     # mitigation half-life (days)


def load_gt() -> list[dict]:
    with open(GT_PATH, encoding="utf-8") as f:
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
            if s.get("status") == "error":
                continue
            out.append(s)
    return out


def strictly_post_with_delta(signals: list[dict], id2event: dict,
                             onsets: dict) -> list[tuple[int, bool]]:
    """Return [(delta_days, is_unresolved), ...] for strictly post-event signals.

    delta = (report_date - onset).days in [1, POST_DAYS].
    is_unresolved = temporality in {forward_looking, latent} (impact still to come).
    """
    rows = []
    for s in signals:
        if not s.get("temporality"):
            continue
        ev_id = id2event.get(s.get("input_id", ""))
        if not ev_id or ev_id not in onsets:
            continue
        onset = onsets[ev_id]
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        delta = (sd - onset).days
        if not (1 <= delta <= POST_DAYS):
            continue
        unresolved = s["temporality"] in FOREWARD_TEMPORALITIES
        rows.append((delta, unresolved))
    return rows


def survival_curve(rows: list[tuple[int, bool]]) -> dict:
    """Pointwise S(delta) = P(unresolved | delta), and censored mean lag."""
    by_delta: dict[int, list[bool]] = defaultdict(list)
    for delta, unresolved in rows:
        by_delta[delta].append(unresolved)
    s_curve = {}
    mean_lag = 0.0
    for delta in range(1, POST_DAYS + 1):
        vals = by_delta.get(delta, [])
        if vals:
            s = sum(vals) / len(vals)
        else:
            s = None
        s_curve[delta] = {"n": len(vals), "s": s}
        if s is not None:
            mean_lag += s
    return {"curve": s_curve, "mean_lag_censored": mean_lag,
            "n_total": len(rows)}


def mitig(lead: float) -> float:
    """Mitigation fraction avoided for `lead` days of genuine notice."""
    return M_MAX * (1 - math.exp(-lead / ELL0))


def main() -> int:
    gt = load_gt()
    onsets = {}
    for ev in gt:
        try:
            onsets[ev["event_id"]] = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
    id2event = _build_article_event_map(BG_ARTICLES, gt, post_days=60)

    results = {}
    for label, path in [("naive (v1)", NAIVE_SIGNALS),
                        ("clock-split (v2)", CLOCKSPLIT_SIGNALS)]:
        sigs = load_signals(path)
        rows = strictly_post_with_delta(sigs, id2event, onsets)
        results[label] = survival_curve(rows)

    # ---- console report ----
    print("=== Lead-time inflation (propagation lag, censored at 14d) ===")
    print(f"{'delta':>5} | {'naive S(δ)':>11} {'n':>4} | {'clock-split S(δ)':>16} {'n':>4}")
    for delta in range(1, POST_DAYS + 1):
        n_naive = results["naive (v1)"]["curve"][delta]
        n_cs = results["clock-split (v2)"]["curve"][delta]
        sn = n_naive["s"] if n_naive["s"] is not None else float("nan")
        sc = n_cs["s"] if n_cs["s"] is not None else float("nan")
        sn_s = f"{sn:.3f}" if n_naive["s"] is not None else "  --"
        sc_s = f"{sc:.3f}" if n_cs["s"] is not None else "  --"
        print(f"{delta:>5} | {sn_s:>11} {n_naive['n']:>4} | {sc_s:>16} {n_cs['n']:>4}")

    lag_naive = results["naive (v1)"]["mean_lag_censored"]
    lag_cs = results["clock-split (v2)"]["mean_lag_censored"]
    n_naive = results["naive (v1)"]["n_total"]
    n_cs = results["clock-split (v2)"]["n_total"]

    print(f"\nmean propagation lag (censored @14d):")
    print(f"  naive       : {lag_naive:.2f} days  (n={n_naive})")
    print(f"  clock-split : {lag_cs:.2f} days  (n={n_cs})")
    print(f"  inflation removed : {lag_naive - lag_cs:.2f} days")

    # decision consequence: phantom mitigation the naive user wrongly expects
    m_naive = mitig(lag_naive)
    m_cs = mitig(lag_cs)
    print(f"\ndecision consequence (m_max={M_MAX}, ell_0={ELL0:.0f}d):")
    print(f"  naive user believes they avoid {m_naive:.3f} of loss")
    print(f"  clock-split user (correct) : {m_cs:.3f} of loss")
    print(f"  phantom mitigation         : {m_naive - m_cs:.3f} "
          f"(={100*(m_naive - m_cs):.1f}% of loss wrongly counted as avoided)")

    # ---- LaTeX table body ----
    print("\n% === tab:leadtime_inflation ===")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"ontology & mean lag (d, cens.) & $n_{\text{post}}$ & $m(\text{lag})$ \\")
    print(r"\midrule")
    print(f"naive three-way & {lag_naive:.2f} & {n_naive} & {m_naive:.3f} \\\\")
    print(f"clock-split & {lag_cs:.2f} & {n_cs} & {m_cs:.3f} \\\\")
    print(r"\midrule")
    print(f"inflation removed & {lag_naive - lag_cs:.2f} & & {m_naive - m_cs:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- optional figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        deltas = list(range(1, POST_DAYS + 1))

        # interpolate both curves onto a common grid so the gap can be shaded
        def interp(label):
            pts = {d: results[label]["curve"][d]["s"] for d in deltas
                   if results[label]["curve"][d]["s"] is not None}
            xs = sorted(pts)
            ys = [pts[x] for x in xs]
            return xs, ys

        xs_naive, ys_naive = interp("naive (v1)")
        xs_cs, ys_cs = interp("clock-split (v2)")
        common = sorted(set(xs_naive) & set(xs_cs))
        if len(common) > 1:
            yn = [results["naive (v1)"]["curve"][d]["s"] for d in common]
            yc = [results["clock-split (v2)"]["curve"][d]["s"] for d in common]
            ax.fill_between(common, yc, yn, color="#EE7733", alpha=0.18,
                            linewidth=0, label="phantom lead (gap)")

        ax.plot(xs_naive, ys_naive, marker="o", color="#CC3311", label="naive (v1)",
                ms=3.5, lw=1.6)
        ax.plot(xs_cs, ys_cs, marker="s", color="#0077BB", label="clock-split (v2)",
                ms=3.5, lw=1.6)

        # annotate the censored mean lags
        ax.axhline(0, color="0.8", lw=0.6)
        ax.annotate("mean lag\n5.9 d", xy=(9, 0.82), fontsize=7, color="#CC3311",
                    ha="center", va="bottom")
        ax.annotate("3.9 d", xy=(6, 0.18), fontsize=7, color="#0077BB",
                    ha="center", va="bottom")

        ax.set_xlabel("days after trigger ($\\delta$)")
        ax.set_ylabel("P(impact unresolved)  $S(\\delta)$")
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0.5, POST_DAYS + 0.5)
        ax.yaxis.grid(True, ls=":", color="0.85")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5, loc="center right")
        fig.tight_layout()
        out_pdf = ROOT / "draft" / "figures" / "fig6_leadtime_inflation.pdf"
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
        # also a PNG for preview
        fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=150)
        print(f"\n[saved] {out_pdf}")
    except Exception as e:  # matplotlib optional
        print(f"\n[figure skipped: {e}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
