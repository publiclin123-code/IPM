#!/usr/bin/env python3
"""Post-event pool breakdown for the beta_0 defect section.

Reports, per cell: signals extracted, strictly post-onset count, beta_0, the
confidence separation between mislabelled and correctly-labelled post-onset
signals, and beta_0 under 7/14/30-day windows (the robustness claim).

Uses metrics.compute_bscc so the beta_0 definition is not reimplemented here.

On the confidence question: this pool shows a REAL mean gap (mislabelled lower
than correct), which an earlier draft's blanket "confidence cannot filter them
out" obscures.  The honest object is the AUC between mislabelled and correct
signals on this pool, reported here, and it should be read together with
analysis/confidence_informativeness.py, which measures the pre-onset
TP-versus-FP separation.  The two pools answer different questions.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))

from metrics import compute_bscc, _build_article_event_map  # noqa: E402
from identity_matcher import load  # noqa: E402

GT = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
CELLS = [
    ("slug x single-field", "results/q38_bg_slug_v1"),
    ("slug x clock-split", "results/q38_bg_slug_v2"),
    ("body x single-field", "results/q38_bg_body_v1"),
    ("body x clock-split", "results/q38_bg_body_v2"),
]
FORWARD = {"forward_looking", "latent"}
ONSETS = {e["event_id"]: datetime.strptime(e["gt_onset_date"][:10], "%Y-%m-%d")
          for e in GT}
id2e = _build_article_event_map(ROOT / "data" / "by_event" / "_background.jsonl", GT,
                               post_days=60)


def auc(pos, neg):
    """P(random positive outranks random negative), ties at 0.5."""
    if not pos or not neg:
        return None
    w = t = 0
    for a in pos:
        for b in neg:
            if a > b:
                w += 1
            elif a == b:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def split_conf(sigs):
    """Confidence of mislabelled vs correctly-labelled strictly post-onset signals."""
    mis, ok = [], []
    for s in sigs:
        c, tmp = s.get("confidence"), s.get("temporality")
        if c is None or not tmp:
            continue
        eid = id2e.get(s.get("input_id", ""))
        if not eid or eid not in ONSETS:
            continue
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except Exception:
            continue
        lag = (sd - ONSETS[eid]).days
        if not (1 <= lag <= 14):
            continue
        (mis if tmp in FORWARD else ok).append(float(c))
    return mis, ok


print("=" * 104)
print("POST-EVENT NEGATIVE-CONTROL POOL")
print("=" * 104)
print(f"{'cell':22s} {'sigs':>5s} {'post':>5s} {'mism':>5s} {'beta0':>7s}  "
      f"{'conf_mis':>8s} {'conf_ok':>8s} {'AUC':>6s}   beta0 at 7/14/30d")
print("-" * 104)
rows = {}
for name, d in CELLS:
    p = ROOT / d / "_background_signals.jsonl"
    if not p.exists():
        print(f"{name:22s}  missing")
        continue
    sigs = load(p)
    by_w = {w: compute_bscc(sigs, GT, post_days=w, id2event=id2e) for w in (7, 14, 30)}
    r = by_w[14]
    mis, ok = split_conf(sigs)
    a = auc(mis, ok)
    mc = r.get("mean_confidence_mismatch")
    mo = r.get("mean_confidence_correct")
    b0 = [by_w[w].get("beta_0_hindsight_rate") for w in (7, 14, 30)]
    print(f"{name:22s} {len(sigs):5d} {r.get('n_strictly_post_onset'):5d} "
          f"{r.get('n_temporal_mismatch'):5d} {r.get('beta_0_hindsight_rate'):7.4f}  "
          f"{mc:8.4f} {mo:8.4f} {a:6.4f}   " + "  ".join(f"{x:.3f}" for x in b0))
    rows[name] = {
        "n_signals_extracted": len(sigs),
        "n_strictly_post_onset_14d": r.get("n_strictly_post_onset"),
        "n_temporal_mismatch": r.get("n_temporal_mismatch"),
        "n_confirmation_correct": r.get("n_confirmation_correct"),
        "n_onset_day_ambiguous": r.get("n_onset_day_ambiguous"),
        "beta0_14d": r.get("beta_0_hindsight_rate"),
        "mean_conf_mislabelled": mc,
        "mean_conf_correct": mo,
        "auc_mislabelled_vs_correct": a,
        "beta0_by_window": {str(w): by_w[w].get("beta_0_hindsight_rate") for w in (7, 14, 30)},
        "temporal_distribution": r.get("temporal_distribution"),
    }

print()
for name, r in rows.items():
    print(f"--- {name}")
    print(f"    extracted {r['n_signals_extracted']}   strictly post-onset "
          f"{r['n_strictly_post_onset_14d']}   onset-day held out {r['n_onset_day_ambiguous']}")
    print(f"    beta_0 {r['beta0_14d']:.4f}   ({r['n_temporal_mismatch']} mislabelled / "
          f"{r['n_confirmation_correct']} correct)")
    print(f"    mean conf  mislabelled {r['mean_conf_mislabelled']:.4f}   correct "
          f"{r['mean_conf_correct']:.4f}   AUC {r['auc_mislabelled_vs_correct']:.4f}")
    print(f"    windows 7/14/30 d: " + ", ".join(f"{v:.3f}" for v in r["beta0_by_window"].values()))
    print()

out = ROOT / "results" / "postevent_breakdown.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)}")
