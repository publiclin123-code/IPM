"""CWS prototype: can an onset-blind score beat self-reported confidence?

This is the decisive test for the paper's framing (see draft/eaai/README.md,
"Risk" section). Three findings have been raised about the manuscript:

  1. It is an evaluation study, and every journal it has been sent to wants a
     different artifact type.
  2. Its strongest repair is a schema change, which an editor can read as a
     prompt revision.
  3. There is no onset-blind firing statistic: FWGS needs the future onset and,
     in the manuscript's own words, "cannot be the system's firing statistic"
     (main.tex L165), so the current illustration fires on raw confidence.

If a causal score can beat raw confidence at deciding whether a warning should
have fired, the system story is real and the paper has a method. If it cannot,
the honest framing is an evaluation paper and the target is a venue that
publishes evaluation.

WHAT IS PREDICTED
-----------------
Among signals the extractor labelled forward-looking or latent -- i.e. signals
that *fire* a warning -- which ones were justified?

  y = 1   fired before onset and the record matches the event   (a real precursor)
  y = 0   fired before onset but does not match the event        (false alarm)
  y = 0   fired after onset                                      (hindsight contamination)

This is the deployment question. A running system emits a warning whenever it
outputs one of these labels, so the population below is exactly the population
its alerts would come from.

At the current labels the naive pipeline's precision is not the published 0.873,
which examines only the pre-onset window, but 207 / (237 + 42) = 0.742. Both are
reported. CWS must beat the baselines on the deployment figure.

CAUSALITY
---------
Every feature is observable at the signal's own timestamp. Forbidden: the onset,
the lead time, the a-priori event type (slow-burn / creeping / sudden) and,
crucially, whether the record is post-onset -- that is not knowable at time t,
which is the whole difficulty.

Note the asymmetry that makes this non-trivial: nothing in a record's content
says whether its trigger has already occurred, and the label that would say so
is the one under test.

EVALUATION
----------
Leave-one-event-out. Each fold holds out one stream entirely, so the model
cannot memorise an event's idiosyncrasies. Scores are pooled out-of-fold.

Usage
-----
  python3 pipeline/cws_prototype.py
  python3 pipeline/cws_prototype.py --variant body
  python3 pipeline/cws_prototype.py --company-labels     # strict y definition
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from validate import signal_matches_event, FOREWARD_TEMPORALITIES, _parse_date  # noqa: E402
from metrics import _build_article_event_map  # noqa: E402

WINDOW = 180
POST_DAYS = 14          # contamination window, matching the manuscript
FUZZ = 1e-9


# ---------------------------------------------------------------- label table
def company_only(sig: dict, ev: dict) -> bool:
    """Stricter ground-truth definition: company evidence required.

    Used only to test whether the prototype's conclusion depends on how loosely
    a hit is defined (see DATA_ISSUES.md DI-8).
    """
    def names(v):
        out = []
        for c in v or []:
            out.append((c.get("name", "") if isinstance(c, dict) else str(c)).strip().lower())
        return [x for x in out if x]
    gv = [str(x).strip().lower() for x in (ev.get("companies") or [])]
    for a in names(sig.get("companies")):
        for b in gv:
            if a == b or a in b or b in a:
                return True
    return False


# Negative-control pool, by input variant. These must match the variant used for
# the positives, or the classifier can separate the two frames on input format
# alone rather than on anything evidential. See cws_leak_check.py TEST C.
BACKGROUND_FILE = {
    "title": "signals_postevent_v2_new.jsonl",
    "body": "signals_postevent_v2_body.jsonl",
}


def build_rows(signals_dir: Path, gt: list[dict], company_labels: bool,
               variant: str = "title") -> list[dict]:
    """Assemble one row per fired warning: pre-onset (labelled) and post-onset."""
    matcher = company_only if company_labels else (lambda s, e: signal_matches_event(s, e, strict=False))
    rows: list[dict] = []

    # --- pre-onset fired warnings, from the event corpora ---
    for ev in gt:
        eid = ev["event_id"]
        p = signals_dir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = _parse_date(ev["gt_onset_date"])
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("status") == "error" or s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            if not s.get("signal_date"):
                continue
            try:
                sd = _parse_date(s["signal_date"])
            except Exception:
                continue
            lead = (onset - sd).days
            if not (0 < lead <= WINDOW):
                continue
            y = 1 if matcher(s, ev) else 0
            rows.append({**s, "event_id": eid, "onset": ev["gt_onset_date"],
                         "lead": lead, "y": y, "kind": "pre_tp" if y else "pre_fp"})

    # --- post-onset fired warnings: hindsight contamination ---
    bg = ROOT / "results" / "_background" / BACKGROUND_FILE[variant]
    id2event = _build_article_event_map(ROOT / "data" / "by_event" / "_background.jsonl", gt, post_days=60)
    ev_by_id = {e["event_id"]: e for e in gt}
    for line in bg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        eid = id2event.get(s.get("input_id"))
        if not eid or eid not in ev_by_id:
            continue
        try:
            sd = _parse_date(s["signal_date"])
            onset = _parse_date(ev_by_id[eid]["gt_onset_date"])
        except Exception:
            continue
        lag = (sd - onset).days
        if not (1 <= lag <= POST_DAYS):
            continue          # strictly post-onset, matching the paper's β₀ window
        rows.append({**s, "event_id": eid, "onset": ev_by_id[eid]["gt_onset_date"],
                     "lead": -lag, "y": 0, "kind": "post_onset"})
    return rows


# ------------------------------------------------------------------- features
def featurize(rows: list[dict]) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[dict]]:
    """Causal features only.

    Returns X, names, y, groups and the rows **reordered to match X**. The
    reordering matters: rows are grouped by stream and sorted by date so that
    stream-history features are causal, and any caller that pairs X with the
    original row list will silently mis-align the two.
    """
    # stable ordering: by event then by date, so stream history is causal
    by_event: dict[str, list[dict]] = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)
    for eid in by_event:
        by_event[eid].sort(key=lambda r: (r["signal_date"], str(r.get("signal_id", ""))))

    ets = sorted({str(r.get("event_type") or "none") for r in rows})
    et_idx = {e: i for i, e in enumerate(ets)}

    X, names, y, groups, ordered = [], [], [], [], []
    names += ["confidence", "severity", "impact_uncertainty",
              "n_trigger_phrases", "mean_trigger_len",
              "n_commodities", "n_companies", "n_geographies",
              "prior_count", "days_since_prev", "prior_mean_conf", "prior_mean_sev"]
    names += [f"event_type={e}" for e in ets]

    for eid, group in by_event.items():
        prior: list[dict] = []
        for r in group:
            conf = float(r.get("confidence") or 0.0)
            sev = float(r.get("severity") or 0.0)
            tp = r.get("trigger_phrases") or []
            tp_len = [len(str(t)) for t in tp]
            comps = r.get("companies") or []

            # stream history: strictly prior signals in the same stream
            prior_count = len(prior)
            if prior:
                try:
                    d = (_parse_date(r["signal_date"]) - _parse_date(prior[-1]["signal_date"])).days
                except Exception:
                    d = 0.0
                pmc = float(np.mean([float(p.get("confidence") or 0) for p in prior]))
                pms = float(np.mean([float(p.get("severity") or 0) for p in prior]))
            else:
                d, pmc, pms = 0.0, 0.0, 0.0

            f = [conf, sev,
                 1.0 if r.get("impact_uncertainty") else 0.0,
                 float(len(tp)),
                 float(np.mean(tp_len)) if tp_len else 0.0,
                 float(len(r.get("commodities") or [])),
                 float(len(comps)),
                 float(len(r.get("geographies") or [])),
                 float(prior_count), float(d), pmc, pms]
            hot = [0.0] * len(ets)
            hot[et_idx[str(r.get("event_type") or "none")]] = 1.0
            f += hot

            X.append(f)
            y.append(int(r["y"]))
            groups.append(eid)
            ordered.append(r)
            prior.append(r)

    return np.array(X, dtype=float), names, np.array(y, dtype=int), groups, ordered


# --------------------------------------------------------------------- models
def loeo_scores(X, y, groups) -> np.ndarray:
    """Out-of-fold predictions with leave-one-event-out."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    events = sorted(set(groups))
    out = np.zeros(len(y))
    for held in events:
        te = np.array([g == held for g in groups])
        tr = ~te
        if tr.sum() < 10 or len(set(y[tr])) < 2:
            out[te] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def prf(y, score, k) -> tuple[float, float, int]:
    """Precision / recall firing the top-k by score."""
    if k <= 0:
        return 0.0, 0.0, 0
    order = np.argsort(-score)[:k]
    tp = int(y[order].sum())
    return tp / k, tp / max(int(y.sum()), 1), tp


def aucs(y, score) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, score), average_precision_score(y, score)


def main() -> int:
    ap = argparse.ArgumentParser(description="CWS prototype vs baselines")
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--company-labels", action="store_true",
                    help="require company evidence for y=1 (strict ground truth)")
    ap.add_argument("--signals-dir", default="",
                    help="override extraction dir (e.g. results/q38_slug_v2)")
    args = ap.parse_args()

    if args.signals_dir:
        _p = Path(args.signals_dir)
        signals_dir = _p if _p.is_absolute() else ROOT / _p
    else:
        signals_dir = ROOT / ("results/v2" if args.variant == "title" else "results/v2_body")
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]

    rows = build_rows(signals_dir, gt, args.company_labels, args.variant)
    kinds = Counter(r["kind"] for r in rows)
    n_fire = len(rows)
    n_pos = sum(r["y"] for r in rows)

    print("=" * 96)
    print(f"CWS prototype   variant={args.variant}   labels="
          f"{'company-only' if args.company_labels else 'loose (current)'}")
    print("=" * 96)
    print(f"fired warnings (forward_looking/latent), n = {n_fire}")
    print(f"   real precursors (y=1)   {kinds.get('pre_tp', 0)}")
    print(f"   pre-onset false alarms  {kinds.get('pre_fp', 0)}")
    print(f"   post-onset contamination{kinds.get('post_onset', 0):>4}")
    print(f"   base rate               {n_pos}/{n_fire} = {n_pos / n_fire:.4f}")
    print()
    print(f"reference: published pre-onset-only precision = "
          f"{kinds.get('pre_tp', 0) / max(kinds.get('pre_tp', 0) + kinds.get('pre_fp', 0), 1):.4f}")
    print(f"reference: deployment precision (this table) = {n_pos / n_fire:.4f}")

    X, names, y, groups, _ordered = featurize(rows)
    print(f"\nfeatures: {len(names)}   streams: {len(set(groups))}")

    cws = loeo_scores(X, y, groups)
    conf = X[:, names.index("confidence")]
    sev = X[:, names.index("severity")]

    k = n_pos                      # fire as many alerts as there are real precursors
    print()
    print("=" * 96)
    print(f"{'method':28s} {'ROC-AUC':>9s} {'PR-AUC':>9s} "
          f"{'P@'+str(k):>9s} {'R@'+str(k):>9s} {'TP@'+str(k):>6s}")
    print("-" * 96)
    for label, score in (("baseline: confidence", conf),
                         ("baseline: label (random)", np.random.RandomState(0).rand(len(y))),
                         ("baseline: severity", sev),
                         ("CWS (logistic, causal)", cws)):
        roc, prauc = aucs(y, score)
        p, r, tp = prf(y, score, k)
        print(f"{label:28s} {roc:9.3f} {prauc:9.3f} {p:9.4f} {r:9.4f} {tp:6d}")

    # how well can confidence alone rank, as the manuscript currently assumes?
    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    roc_c, pr_c = aucs(y, conf)
    roc_w, pr_w = aucs(y, cws)
    p_c, _, _ = prf(y, conf, k)
    p_w, _, _ = prf(y, cws, k)
    print(f"  CWS      ROC-AUC {roc_w:.3f}  PR-AUC {pr_w:.3f}  P@{k} {p_w:.4f}")
    print(f"  confid.  ROC-AUC {roc_c:.3f}  PR-AUC {pr_c:.3f}  P@{k} {p_c:.4f}")
    gain = p_w - p_c
    print(f"  precision gain over confidence: {gain:+.4f}")
    if roc_w > roc_c + 0.02 and p_w > p_c + 0.01:
        print("  -> CWS beats confidence. The system story has a foothold.")
    elif p_w > p_c - 0.005:
        print("  -> CWS is not better than confidence. Framing as a system is NOT supported.")
        print("     The current illustration (firing on confidence) already captures what is available.")
    else:
        print("  -> CWS is WORSE than confidence. Do not build the system.")

    # which features carry the signal
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    coef = clf.coef_[0]
    order = np.argsort(-np.abs(coef))
    print()
    print("  strongest coefficients (full-data fit, for interpretation only):")
    for i in order[:8]:
        print(f"     {names[i]:26s} {coef[i]:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
