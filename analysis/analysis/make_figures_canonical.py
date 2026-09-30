"""Canonical figure generation for the corrected 17-event dataset.

Replaces the twelve scattered `fig*.py` scripts, which between them contained
hard-coded beta_0 constants, a renamed cache path, and references to the
retracted `toyota_steel_explosion_2019` event. Generating every figure from one
code path that reads the current ground truth is the only way to guarantee the
figures and the tables cannot drift apart -- drift between the two was itself a
finding (DATA_ISSUES.md DI-4).

Every figure reads:
  - validation/gt_events.json          (17 events, post-retraction)
  - the four extraction cells listed in CELLS below
  - pipeline/identity_matcher.py       (the hit rule actually reported)
  - the canonical bootstrap output listed in STATS_JSON

Extraction cells
----------------
The 2026-09-19 re-extraction on qwen3.8-27b replaced the original qwen3.6-27b
runs.  It had to: the original single-field (`v1`) baseline covered only 5-33% of
the corpus for four events (us_chip 34/673, renesas 50/185, red_sea 36/108,
port_la 12/53), and those events supplied 35% of all v1 forward signals.  The
headline v1-versus-v2 comparison was therefore made against an incomplete
baseline and every number derived from it is void.

CELLS holds the four cells that must be compared on identical input: two
schemas (single-field, clock-split) crossed with two corpora (slug, body).  Every
figure and table must read through cell(); a literal path here is a bug waiting
to happen, because the earlier scripts had the invalid `results/` directory
baked into the headline figure.

Figures produced (draft/ipm/figures/):
  fig_leadtime_by_type     signals on a days-from-onset axis, grouped by type
  fig_beta0                beta_0 naive vs clock-split, with bootstrap CIs
  fig_fwgs_by_type         pooled FWGS per event, ordered by a priori type
  fig_schema_trade         NEW headline: what the clock split gains and loses
  fig_match_sensitivity    precision across the five matching rules
  fig_recall_sensitivity   precision as a function of assumed extraction recall
  fig_crosscorpus          GDELT news vs SEC EDGAR

Interpreter: needs matplotlib. On this machine
/home/e/miniconda3/envs/ldl_los/bin/python3 has it; base does not.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "pipeline"))
from figstyle import TOL, apply_style, save  # noqa: E402
from event_meta import EVENT_TYPES, EVENT_SHORT  # noqa: E402
from identity_matcher import rule_match, load, parse, RULES  # noqa: E402

FORWARD = {"forward_looking", "latent"}
WINDOW = 180
TAU, LAMBDA = 30.0, 0.5
RULE = "R4_identity"
FIG_DIR = ROOT / "draft" / "ipm" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data sources
# The four cells.  (corpus, schema) -> directory holding <event>_signals.jsonl.
CELLS = {
    ("title", "v1"): ROOT / "results" / "q38_slug_v1",
    ("title", "v2"): ROOT / "results" / "q38_slug_v2",
    ("body", "v1"): ROOT / "results" / "q38_body_v1",
    ("body", "v2"): ROOT / "results" / "q38_body_v2",
}
# Canonical bootstrap output for the cell used by the beta_0 and recall figures.
STATS_JSON = ROOT / "results" / "stats_canonical_q38_slug_v2.json"
# EDGAR cross-corpus extraction.
EDGAR_DIR = ROOT / "results" / "q38_edgar"


def cell(variant: str, schema: str) -> Path:
    """Directory for one (corpus, schema) cell.  Fails loudly, never silently."""
    key = (variant, schema)
    if key not in CELLS:
        raise KeyError(f"unknown cell {key}; known: {sorted(CELLS)}")
    return CELLS[key]


def stats_json() -> dict:
    p = STATS_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"{p.relative_to(ROOT)} missing. Run:\n"
            f"  python3 pipeline/stats_canonical.py --signals-dir "
            f"{CELLS[('title','v2')].relative_to(ROOT)} --out {p.relative_to(ROOT)}"
        )
    return json.loads(p.read_text(encoding="utf-8"))

TYPE_ORDER = ["slow_burn", "creeping", "sudden"]
TYPE_COLOR = {"slow_burn": TOL["blue"], "creeping": TOL["teal"], "sudden": TOL["red"]}
TYPE_LABEL = {"slow_burn": "slow-burn", "creeping": "creeping", "sudden": "sudden"}


def savef(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def gt() -> list[dict]:
    return json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]


def collect(variant: str) -> dict:
    """Per event: every forward signal in the window, with lead, is_tp, faithfulness."""
    sdir = cell(variant, "v2")
    abase = ROOT / ("data/by_event" if variant == "title" else "data/by_event_body")
    out = {}
    for ev in gt():
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            out[eid] = []
            continue
        onset = parse(ev["gt_onset_date"])
        arts = {}
        ap_ = abase / f"{eid}.jsonl"
        if ap_.exists():
            for line in ap_.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("id"):
                    arts[r["id"]] = (r.get("text") or r.get("title") or "")
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
            ph = s.get("trigger_phrases") or []
            txt = arts.get(s.get("input_id") or "", "").lower()
            g = (sum(1 for t in ph if str(t).lower() in txt) / len(ph)) if (ph and txt) else (1.0 if not ph else 0.0)
            sigs.append({"lead": lead, "is_tp": rule_match(s, ev, RULE), "g": g,
                         "signal_id": s.get("signal_id", "")})
        out[eid] = sigs
    return out


def pooled_fwgs(sigs: list[dict]) -> float | None:
    if not sigs:
        return None
    vals = []
    for s in sigs:
        if s["is_tp"]:
            vals.append((1 - math.exp(-s["lead"] / TAU)) * s["g"])
        else:
            vals.append(-LAMBDA * (1 - s["g"]))
    return sum(vals) / len(vals)


def fig_leadtime(data: dict, variant: str) -> None:
    evs = gt()
    order = [e["event_id"] for t in TYPE_ORDER for e in evs if EVENT_TYPES.get(e["event_id"]) == t]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, eid in enumerate(order):
        t = EVENT_TYPES.get(eid, "?")
        sigs = data.get(eid, [])
        if not sigs:
            ax.text(2, i, "no usable corpus", va="center", fontsize=7.5,
                    color=TOL["grey"], style="italic")
            continue
        for s in sigs:
            ax.scatter(s["lead"], i, s=26 if s["is_tp"] else 16,
                       color=TYPE_COLOR[t] if s["is_tp"] else "white",
                       edgecolor=TYPE_COLOR[t], linewidth=0.9,
                       alpha=0.95 if s["is_tp"] else 0.65, zorder=3)
    ax.axvline(0, color=TOL["black"], lw=1.4, zorder=2)
    ax.text(1, len(order) - 0.2, "onset", fontsize=8, color=TOL["black"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([EVENT_SHORT.get(e, e) for e in order], fontsize=8)
    ax.set_xlabel("days before onset", fontsize=9)
    ax.set_xlim(-4, WINDOW + 8)
    ax.set_ylim(-0.8, len(order) - 0.2)
    for t in TYPE_ORDER:
        idx = [i for i, e in enumerate(order) if EVENT_TYPES.get(e) == t]
        if idx:
            ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color=TYPE_COLOR[t], alpha=0.05, zorder=0)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", color=TOL["blue"], label="true precursor"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=TOL["grey"], label="false alarm")]
    ax.legend(handles=leg, fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title(f"Forward signals by days before onset ({variant} input, identity match)",
                 fontsize=9.5)
    savef(fig, "fig_leadtime_by_type")


def fig_beta0() -> None:
    st = stats_json()
    rows = [("slug\n(naive)", "title_naive"), ("slug\n(clock-split)", "title_v2"),
            ("body\n(naive)", "body_naive"), ("body\n(clock-split)", "body_v2")]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = np.arange(len(rows))
    pts = [st["beta0"][k]["point"] for _, k in rows]
    los = [st["beta0"][k]["ci"][0] for _, k in rows]
    his = [st["beta0"][k]["ci"][1] for _, k in rows]
    cols = [TOL["red"], TOL["teal"], TOL["red"], TOL["teal"]]
    ax.bar(xs, pts, color=cols, width=0.6, zorder=2)
    ax.errorbar(xs, pts, yerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
                fmt="none", ecolor=TOL["black"], capsize=4, lw=1.2, zorder=3)
    for x, p, n in zip(xs, pts, [st["beta0"][k]["n"] for _, k in rows]):
        ax.text(x, p + 0.035, f"{p:.3f}\nn={n}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel(r"contamination rate $\beta_0$", fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_title("Hindsight contamination on the post-onset pool\n"
                 "(bars: point estimate; whiskers: event-stratified bootstrap, B=2000)",
                 fontsize=9)
    savef(fig, "fig_beta0")


def fig_fwgs(data: dict, variant: str) -> None:
    evs = gt()
    order = [e["event_id"] for t in TYPE_ORDER for e in evs if EVENT_TYPES.get(e["event_id"]) == t]
    vals, cols, labs = [], [], []
    for eid in order:
        f = pooled_fwgs(data.get(eid, []))
        vals.append(f if f is not None else 0.0)
        cols.append(TYPE_COLOR[EVENT_TYPES.get(eid, "?")])
        labs.append(EVENT_SHORT.get(eid, eid))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=TOL["black"], lw=0.9)
    ax.set_xlabel("pooled FWGS (ex-post foresight score)", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t]) for t in TYPE_ORDER],
              fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Ex-post foresight score by event type ({variant} input)", fontsize=9.5)
    savef(fig, "fig_fwgs_by_type")


def fig_schema_trade(variant: str) -> None:
    """The headline finding: what the clock split gains and loses, per event."""
    v1_dir = cell(variant, "v1")
    v2_dir = cell(variant, "v2")
    rows = []
    for ev in gt():
        eid = ev["event_id"]
        p1, p2 = v1_dir / f"{eid}_signals.jsonl", v2_dir / f"{eid}_signals.jsonl"
        if not (p1.exists() and p2.exists()):
            continue
        onset = parse(ev["gt_onset_date"])

        def hits(path):
            out = set()
            for s in load(path):
                if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                    continue
                try:
                    lead = (onset - parse(s["signal_date"])).days
                except Exception:
                    continue
                if 0 < lead <= WINDOW and rule_match(s, ev, RULE):
                    out.add(s.get("input_id") or s.get("signal_id"))
            return out
        h1, h2 = hits(p1), hits(p2)
        rows.append({"eid": eid, "type": EVENT_TYPES.get(eid, "?"),
                     "v2_gain": len(h2 - h1), "v1_gain": len(h1 - h2)})
    rows.sort(key=lambda r: (TYPE_ORDER.index(r["type"]) if r["type"] in TYPE_ORDER else 9, r["eid"]))
    fig, ax = plt.subplots(figsize=(8.8, 6))
    y = np.arange(len(rows))
    v1g = np.array([r["v1_gain"] for r in rows], dtype=float)
    v2g = np.array([-r["v2_gain"] for r in rows], dtype=float)
    ax.barh(y, v1g, color=TOL["grey"], label="precursors the clock split discards", zorder=2)
    ax.barh(y, v2g, color=TOL["teal"], label="precursors only the clock split finds", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([EVENT_SHORT.get(r["eid"], r["eid"]) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=TOL["black"], lw=1.0)
    ax.set_xlabel("pre-onset matching signals gained (positive: single-field) or lost (negative)",
                  fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    n_win = sum(1 for r in rows if r["v2_gain"] > r["v1_gain"])
    ax.set_title("The intuitive repair is not free: clock split vs single field\n"
                 f"({n_win} of {len(rows)} events improve under the split)",
                 fontsize=9.5)
    savef(fig, "fig_schema_trade")


def fig_match_sensitivity(variant: str) -> None:
    sdir = cell(variant, "v2")
    per = defaultdict(dict)
    tot = Counter()
    n_fwd = 0
    for ev in gt():
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        fwd = []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if 0 < lead <= WINDOW:
                fwd.append(s)
        n_fwd += len(fwd)
        for r in RULES:
            n = sum(1 for s in fwd if rule_match(s, ev, r))
            per[eid][r] = n
            tot[r] += n
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(RULES))
    prec = [tot[r] / n_fwd for r in RULES]
    cols = [TOL["grey"]] * len(RULES)
    cols[RULES.index(RULE)] = TOL["blue"]
    ax.bar(xs, prec, color=cols, width=0.62, zorder=2)
    for x, p_, r in zip(xs, prec, RULES):
        ax.text(x, p_ + 0.02, f"{p_:.3f}", ha="center", fontsize=8)
        ax.text(x, -0.06, f"{tot[r]} TP", ha="center", fontsize=7, color=TOL["grey"])
    ax.set_xticks(xs)
    ax.set_xticklabels([r.split("_", 1)[1].replace("_", "\n") for r in RULES], fontsize=7.5)
    ax.set_ylabel("protocol precision", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Measured precision depends on the matching rule\n"
                 "(the reported rule is highlighted)", fontsize=9.5)
    savef(fig, "fig_match_sensitivity")


def fig_recall_sensitivity() -> None:
    st = stats_json()
    rec = st["recall_sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    rs = sorted((float(k) for k in rec), reverse=True)
    ys = [rec[str(r)] if str(r) in rec else rec[r] for r in rs]
    ax.plot(rs, ys, "o-", color=TOL["blue"], zorder=3)
    for r, y in zip(rs, ys):
        ax.annotate(f"{y:.3f}", (r, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5)
    ax.set_xlabel("assumed extraction recall $r$", fontsize=9)
    ax.set_ylabel("implied protocol precision", fontsize=9)
    ax.set_xlim(0.42, 1.05)
    ax.set_ylim(0, 0.9)
    ax.invert_xaxis()
    ax.set_title("Bounding the effect of unmeasured recall\n"
                 r"precision($r$) $=$ TP$/($TP$/r + $FP$)", fontsize=9.5)
    savef(fig, "fig_recall_sensitivity")


def fig_crosscorpus() -> None:
    ed = ROOT / "results" / "edgar"
    agg_p = ed / "_aggregate.json"
    if not agg_p.exists():
        print("  [skip] fig_crosscorpus: no EDGAR aggregate")
        return
    agg = json.loads(agg_p.read_text(encoding="utf-8"))
    news = stats_json()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    precs = [news["point"]["precision"], agg.get("aggregate_precision")]
    labs = ["GDELT news\n(identity match)", "SEC EDGAR"]
    axes[0].bar([0, 1], precs, color=[TOL["blue"], TOL["teal"]], width=0.55, zorder=2)
    ci = news["bootstrap"]["precision_ci"]
    axes[0].errorbar([0], [precs[0]], yerr=[[precs[0] - ci[0]], [ci[1] - precs[0]]],
                     fmt="none", ecolor=TOL["black"], capsize=4, zorder=3)
    for x, p_ in zip([0, 1], precs):
        axes[0].text(x, p_ + 0.03, f"{p_:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(labs, fontsize=8)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("protocol precision", fontsize=9)
    axes[0].set_title("Precision", fontsize=9)
    fw = [news["point"]["fwgs"], agg.get("pooled_fwgs")]
    axes[1].bar([0, 1], fw, color=[TOL["blue"], TOL["teal"]], width=0.55, zorder=2)
    for x, p_ in zip([0, 1], fw):
        axes[1].text(x, p_ + 0.02, f"{p_:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labs, fontsize=8)
    axes[1].set_ylabel("pooled FWGS", fontsize=9)
    axes[1].set_title("Ex-post foresight", fontsize=9)
    fig.suptitle("Second-hand news versus first-hand filings", fontsize=9.5)
    savef(fig, "fig_crosscorpus")


def fig_elr(variant: str) -> None:
    """Expected-loss reduction when the system fires on confidence.

    PROMOTED 2026-09-19: this was slated for deletion as an "illustration with
    stated-preference parameters". It is now load-bearing, because the paper
    reports the improvement condition in terms of the false-positive cost c and
    ELR is the apparatus that expresses c and locates the crossover. Keep the
    parameters visible.

    The crossover moved when the extraction was repeated on complete corpora:
    c* = 0.70 on slug input and c* = 1.42 on full text, against 0.87 reported
    from the invalid baseline.  The figure reads the break-even point from the
    data rather than printing a constant, so it cannot drift again.
    """
    sdir = cell(variant, "v2")
    pts = []
    for ev in gt():
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if not (0 < lead <= WINDOW):
                continue
            pts.append({"conf": float(s.get("confidence") or 0.0),
                        "is_tp": rule_match(s, ev, RULE), "lead": lead,
                        "event_id": eid})
    if not pts:
        print("  [skip] fig_elr: no signals")
        return
    M_MAX, L0, C_FP = 0.8, 60.0, 0.05
    thetas = np.linspace(0.3, 0.95, 14)
    n_events = len({p["event_id"] for p in pts})
    elr = []
    for th in thetas:
        fired = [p for p in pts if p["conf"] >= th]
        if not fired:
            elr.append(0.0)
            continue
        # p_fire is the probability that an EVENT fires at least one signal, not
        # the fraction of signals that fire. Using the signal fraction (an earlier
        # bug here) understates p_fire and makes the figure disagree with the
        # threshold table in the manuscript.
        n_fired_events = len({p["event_id"] for p in fired})
        p_fire = n_fired_events / n_events
        hits = [p for p in fired if p["is_tp"]]
        p_hit = len(hits) / len(fired)
        lbar = float(np.mean([p["lead"] for p in hits])) if hits else 0.0
        m = M_MAX * (1 - math.exp(-lbar / L0))
        cost = p_fire * ((1 - p_hit) * C_FP + p_hit * (1 - m)) + (1 - p_fire) * 1.0
        elr.append(1 - cost)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(thetas, elr, "o-", color=TOL["blue"], zorder=3)
    best = int(np.argmax(elr))
    ax.scatter([thetas[best]], [elr[best]], s=90, facecolor="none",
               edgecolor=TOL["red"], lw=1.6, zorder=4)
    ax.annotate(f"max {elr[best]:.3f}\n$\\theta$={thetas[best]:.2f}",
                (thetas[best], elr[best]), textcoords="offset points",
                xytext=(10, -14), fontsize=7.5)
    ax.set_xlabel(r"confidence threshold $\theta$", fontsize=9)
    ax.set_ylabel("expected loss reduction", fontsize=9)
    ax.set_title("Firing on confidence, and what the improvement condition buys\n"
                 f"($m_{{\\max}}$={M_MAX}, $\\ell_0$={L0:.0f}d, $C_{{FP}}$={C_FP}; "
                 "stated preferences, not estimated damages)", fontsize=8.5)
    savef(fig, "fig_elr")


def fig_cumulative(variant: str) -> None:
    """Share of a stream's forward signals already on file, by days before onset."""
    sdir = cell(variant, "v2")
    curves = {}
    for ev in gt():
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        leads = []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if 0 < lead <= WINDOW and rule_match(s, ev, RULE):
                leads.append(lead)
        if leads:
            curves[eid] = sorted(leads)
    if not curves:
        print("  [skip] fig_cumulative: no streams")
        return
    xs = np.linspace(0, WINDOW, 120)
    pooled = []
    for x in xs:
        shares = [sum(1 for l in ls if l <= x) / len(ls) for ls in curves.values()]
        pooled.append(float(np.mean(shares)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for eid, ls in curves.items():
        t = EVENT_TYPES.get(eid, "?")
        ax.plot(xs, [sum(1 for l in ls if l <= x) / len(ls) for x in xs],
                color=TYPE_COLOR[t], alpha=0.28, lw=1.0, zorder=2)
    ax.plot(xs, pooled, color=TOL["black"], lw=2.0, zorder=3, label="pooled mean")
    half = next((x for x, p_ in zip(xs, pooled) if p_ >= 0.5), None)
    if half is not None:
        ax.axvline(half, color=TOL["grey"], ls="--", lw=1.0, zorder=1)
        ax.annotate(f"half the signal mass\non file {half:.0f} d before onset",
                    (half, 0.5), textcoords="offset points", xytext=(8, -30),
                    fontsize=7.5)
    ax.set_xlabel("days before onset", fontsize=9)
    ax.set_ylabel("share of the event's forward signals on file", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, WINDOW)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t], alpha=0.6)
                       for t in TYPE_ORDER] + [plt.Line2D([], [], color=TOL["black"], label="pooled mean")],
              fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Cumulative detection of forward signals ({variant} input)", fontsize=9.5)
    savef(fig, "fig_cumulative")


def fig_retrieval_precision() -> None:
    """Per-event retrieval precision from the blind audit worksheet."""
    import csv
    ws = ROOT / "data" / "annotation" / "retrieval_audit_sample_body.csv"
    if not ws.exists():
        print("  [skip] fig_retrieval_precision: no worksheet")
        return
    rows = list(csv.DictReader(ws.open(encoding="utf-8")))
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        eid = r.get("event_id") or ""
        lab = (r.get("relevant_label_body") or "").strip().lower()
        if not eid or not lab:
            continue
        per[eid][1] += 1
        if lab in ("1", "yes", "relevant", "true"):
            per[eid][0] += 1
    if not per:
        print("  [skip] fig_retrieval_precision: nothing labelled")
        return
    items = sorted(((e, h / n if n else 0.0, n) for e, (h, n) in per.items()),
                   key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.26 * len(items))))
    ys = np.arange(len(items))
    ax.barh(ys, [i[1] for i in items], color=TOL["grey"], zorder=2)
    tot_h = sum(per[e][0] for e in per)
    tot_n = sum(per[e][1] for e in per)
    ax.axvline(tot_h / tot_n, color=TOL["red"], ls="--", lw=1.2, zorder=3,
               label=f"pooled {tot_h/tot_n:.3f} ({tot_h}/{tot_n})")
    ax.set_yticks(ys)
    ax.set_yticklabels([EVENT_SHORT.get(e, e) for e, _v, _n in items], fontsize=7.5)
    ax.set_xlabel("retrieval precision (blind audit, body text)", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("The keyword stage is high-coverage and low-precision\n"
                 "and it is the audit's binding constraint", fontsize=9.5)
    savef(fig, "fig_retrieval_precision")


def fig_model_agreement(variant: str) -> None:
    """Per-event FWGS across the four checkpoints, to test checkpoint dependence.

    Replaces the earlier four-model heatmap. Reads the cross-model result
    directories, which carry the clock-split extraction for gemma4-31b,
    muse-glimmer-30b and the DeepSeek API. Events absent from the ground truth
    are excluded, so the retracted event cannot leak back in.
    """
    models = {
        "qwen3.8-27b": cell(variant, "v2"),
        "gemma4-31b": ROOT / "results" / "v2" / "gemma4",
        "muse-glimmer-30b": ROOT / "results" / "v2" / "muse",
        "DeepSeek V4": ROOT / "results" / "deepseek" / "gdelt",
    }
    evs = gt()
    order = [e["event_id"] for t in TYPE_ORDER for e in evs if EVENT_TYPES.get(e["event_id"]) == t]
    grid = np.full((len(order), len(models)), np.nan)
    for j, (mname, d) in enumerate(models.items()):
        for i, eid in enumerate(order):
            p = d / f"{eid}_signals.jsonl"
            if not p.exists():
                continue
            ev = next(e for e in evs if e["event_id"] == eid)
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
                ph = s.get("trigger_phrases") or []
                sigs.append({"lead": lead, "is_tp": rule_match(s, ev, RULE),
                             "g": 1.0 if not ph else 0.0})
            f = pooled_fwgs(sigs)
            if f is not None:
                grid[i, j] = f
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(order) + 1.9))
    im = ax.imshow(grid, cmap="RdYlBu", vmin=-0.2, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(list(models), fontsize=7.5, rotation=20, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([EVENT_SHORT.get(e, e) for e in order], fontsize=7.5)
    for i in range(len(order)):
        for j in range(len(models)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color=TOL["black"])
    fig.colorbar(im, ax=ax, shrink=0.7, label="pooled FWGS")
    ax.set_title("Foresight score by checkpoint\n"
                 "(blank: no signal in the window)", fontsize=9)
    savef(fig, "fig_model_agreement")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--only", default="", help="comma-separated figure stems")
    args = ap.parse_args()

    apply_style()
    want = {s.strip() for s in args.only.split(",") if s.strip()}

    def do(name, fn):
        if want and name not in want:
            return
        print(f"[{name}]")
        fn()

    data = collect(args.variant)
    n = sum(len(v) for v in data.values())
    print(f"variant={args.variant}  rule={RULE}  forward signals (pre-onset) = {n}")
    print(f"events with signals: {sum(1 for v in data.values() if v)} / {len(data)}")
    print(f"output: {FIG_DIR}\n")

    do("fig_leadtime_by_type", lambda: fig_leadtime(data, args.variant))
    do("fig_beta0", fig_beta0)
    do("fig_fwgs_by_type", lambda: fig_fwgs(data, args.variant))
    do("fig_schema_trade", lambda: fig_schema_trade(args.variant))
    do("fig_match_sensitivity", lambda: fig_match_sensitivity(args.variant))
    do("fig_recall_sensitivity", fig_recall_sensitivity)
    do("fig_elr", lambda: fig_elr(args.variant))
    do("fig_cumulative", lambda: fig_cumulative(args.variant))
    do("fig_retrieval_precision", fig_retrieval_precision)
    do("fig_model_agreement", lambda: fig_model_agreement(args.variant))
    do("fig_crosscorpus", fig_crosscorpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
