"""BSCC window robustness: beta_0 at post_days in {7, 14, 30}.

Reuses compute_bscc from validation/metrics.py with different post_days.
The post-event pool was fetched with a 30-day window, so all three windows
can be evaluated on the same articles.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results"

sys.path.insert(0, str(VAL))
from metrics import compute_bscc, _build_article_event_map  # noqa: E402

GT_PATH = VAL / "gt_events.json"
BG_ARTICLES = ROOT / "data" / "by_event" / "_background.jsonl"
SIGNALS = RES / "_background" / "signals_postevent.jsonl"  # qwen3.6-27b (primary)
SIGNALS_GEMMA4 = RES / "_background" / "signals_postevent_gemma4.jsonl"  # gemma4-31b (cross-model)

WINDOWS = [7, 14, 30, 60]


def load_gt() -> list[dict]:
    with open(GT_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["events"]


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


def main() -> int:
    gt = load_gt()
    id2event = _build_article_event_map(BG_ARTICLES, gt, post_days=60)

    print("=== BSCC window robustness (qwen3.6-27b) ===")
    print(f"{'post_days':>9} {'beta0':>6} {'n_post':>6} {'mismatch':>8} {'correct':>7} {'onset_day':>9}")
    for w in WINDOWS:
        sigs = load_signals(SIGNALS)
        r = compute_bscc(sigs, gt, post_days=w, id2event=id2event)
        if "bscc" in r and r["bscc"] is None:
            print(f"{w:>9} {'n/a':>6} {0:>6} {0:>8} {0:>7} {r.get('n_onset_day_ambiguous',0):>9}")
            continue
        print(f"{w:>9} {r['beta_0_hindsight_rate']:>6} {r['n_strictly_post_onset']:>6} "
              f"{r['n_temporal_mismatch']:>8} {r['n_confirmation_correct']:>7} "
              f"{r['n_onset_day_ambiguous']:>9}")

    # gemma4-31b if available
    if SIGNALS_GEMMA4.exists() and SIGNALS_GEMMA4.stat().st_size > 0:
        print("\n=== BSCC cross-model (gemma4-31b) ===")
        print(f"{'post_days':>9} {'beta0':>6} {'n_post':>6} {'mismatch':>8} {'correct':>7} {'onset_day':>9}")
        for w in WINDOWS:
            sigs = load_signals(SIGNALS_GEMMA4)
            r = compute_bscc(sigs, gt, post_days=w, id2event=id2event)
            if "bscc" in r and r["bscc"] is None:
                print(f"{w:>9} {'n/a':>6} {0:>6} {0:>8} {0:>7} {r.get('n_onset_day_ambiguous',0):>9}")
                continue
            print(f"{w:>9} {r['beta_0_hindsight_rate']:>6} {r['n_strictly_post_onset']:>6} "
                  f"{r['n_temporal_mismatch']:>8} {r['n_confirmation_correct']:>7} "
                  f"{r['n_onset_day_ambiguous']:>9}")
    else:
        print("\n[gemma4-31b signals not ready yet]")

    # ---- LaTeX table body ----
    print("\n% === tab:bscc_robust === (rows = model x post_days)")
    for model, path in [("qwen3.6-27b", SIGNALS), ("gemma4-31b", SIGNALS_GEMMA4)]:
        sigs = load_signals(path)
        if not sigs:
            continue
        cells = []
        for w in WINDOWS:
            r = compute_bscc(sigs, gt, post_days=w, id2event=id2event)
            if "bscc" in r and r["bscc"] is None:
                cells.append("--")
            else:
                cells.append(f"{r['beta_0_hindsight_rate']:.3f}")
        print(f"{model} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
