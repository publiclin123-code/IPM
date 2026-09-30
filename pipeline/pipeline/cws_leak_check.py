"""Diagnose whether the CWS prototype is learning the phenomenon or the corpus.

The prototype's negative class mixes two populations drawn from *different
sampling frames*:

  pre-onset false alarms   from the event corpora (keyword-filtered, 180-day window)
  post-onset contamination from the background pool (340 articles, post-onset only)

If those two frames differ systematically in signal characteristics, then a
classifier can separate them by "which corpus is this from" rather than by
"should this warning have fired". That would look like a win and mean nothing.

The prototype's fitted coefficients make this a live worry: the largest weights
land on mean_trigger_len (-1.83), n_trigger_phrases (-0.98) and prior_count
(-0.61) -- surface properties, not evidential ones.

Three tests:

  A. Pre-onset only. Positives vs pre-onset false alarms, same corpus, same
     window. The only difference is whether the record's entities match the
     event. No sampling-frame confound is possible.
  B. Feature drift. Compare the distributions of every feature between the
     pre-onset and post-onset groups. Large gaps mean the classifier can cheat.
  C. Corpus-only classifier. Predict post-onset vs pre-onset *ignoring the
     label entirely*. If this is highly accurate, the two frames are separable
     and test A is the only trustworthy one.

Usage
-----
  python3 pipeline/cws_leak_check.py
  python3 pipeline/cws_leak_check.py --variant body
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from cws_prototype import build_rows, featurize, loeo_scores, aucs, prf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Leak check for the CWS prototype")
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--company-labels", action="store_true")
    args = ap.parse_args()

    signals_dir = ROOT / ("results/v2" if args.variant == "title" else "results/v2_body")
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    rows = build_rows(signals_dir, gt, args.company_labels, args.variant)

    X, names, y, groups, ordered = featurize(rows)
    # kind must be read off the *reordered* rows, not the original list
    pre = np.array([r["kind"].startswith("pre") for r in ordered])
    post = np.array([r["kind"] == "post_onset" for r in ordered])

    print("=" * 100)
    print("TEST B - feature drift between the two sampling frames")
    print("=" * 100)
    print(f"{'feature':26s} {'pre-onset mean':>15s} {'post-onset mean':>16s} {'gap (SD units)':>15s}")
    print("-" * 100)
    worst = []
    for i, nm in enumerate(names):
        a, b = X[pre, i], X[post, i]
        if len(a) == 0 or len(b) == 0:
            continue
        sd = np.std(np.concatenate([a, b])) or 1.0
        gap = abs(a.mean() - b.mean()) / sd
        worst.append((gap, nm, a.mean(), b.mean()))
    for gap, nm, ma, mb in sorted(worst, reverse=True)[:8]:
        flag = "  <<< drift" if gap > 0.5 else ""
        print(f"{nm:26s} {ma:15.3f} {mb:16.3f} {gap:15.3f}{flag}")

    print()
    print("=" * 100)
    print("TEST C - can the sampling frame be predicted without using the label?")
    print("=" * 100)
    frame = post.astype(int)          # 1 = background pool, 0 = event corpus
    sel = pre | post
    sc = loeo_scores(X[sel], frame[sel], [g for g, s in zip(groups, sel) if s])
    roc, prauc = aucs(frame[sel], sc)
    print(f"  predicting 'is this from the background pool' from features alone:")
    print(f"     ROC-AUC {roc:.3f}   PR-AUC {prauc:.3f}")
    if roc > 0.8:
        print("     -> the frames are readily separable: ANY model mixing them can cheat.")
    elif roc > 0.65:
        print("     -> the frames are partially separable: treat mixed-frame results with care.")
    else:
        print("     -> the frames are not readily separable: mixing is acceptable.")

    print()
    print("=" * 100)
    print("TEST A - pre-onset only (no sampling-frame confound)")
    print("=" * 100)
    selA = pre
    Xa, ya, ga = X[selA], y[selA], [g for g, s in zip(groups, selA) if s]
    n_pos = int(ya.sum())
    print(f"  positives (real precursors, y=1) : {n_pos}")
    print(f"  negatives (pre-onset false alarm): {len(ya) - n_pos}")
    if len(ya) - n_pos < 5 or n_pos < 5:
        print("  too few of one class to fit")
        return 0
    cws = loeo_scores(Xa, ya, ga)
    conf = Xa[:, names.index("confidence")]
    sev = Xa[:, names.index("severity")]
    k = n_pos
    print()
    print(f"{'method':28s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'P@'+str(k):>10s}")
    print("-" * 100)
    for label, s in (("baseline: confidence", conf),
                     ("baseline: severity", sev),
                     ("CWS (logistic, causal)", cws)):
        roc, pr = aucs(ya, s)
        p, _, _ = prf(ya, s, k)
        print(f"{label:28s} {roc:9.3f} {pr:9.3f} {p:10.4f}")

    roc_c, _ = aucs(ya, conf)
    roc_w, _ = aucs(ya, cws)
    p_c, _, _ = prf(ya, conf, k)
    p_w, _, _ = prf(ya, cws, k)
    print()
    print("  VERDICT on the clean comparison")
    print(f"     CWS   ROC-AUC {roc_w:.3f}  P@{k} {p_w:.4f}")
    print(f"     conf  ROC-AUC {roc_c:.3f}  P@{k} {p_c:.4f}")
    if roc_w > roc_c + 0.05:
        print("     -> CWS beats confidence even within one corpus. The gain is real.")
    else:
        print("     -> Within one corpus CWS does NOT clearly beat confidence.")
        print("        The mixed-frame gain was largely a sampling artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
