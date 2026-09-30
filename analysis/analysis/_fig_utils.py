"""Shared small helpers for Paper A headline figures (fig_overview/crosscorpus/foresight_map).

Kept intentionally thin: Wilson interval, plus a few confirmed constants that the
paper's headline numbers depend on. Source-of-truth values come from the existing
aggregate JSON files, not recomputed here, so the figures never drift from the text.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Confirmed headline values, cross-checked against results/v2/ and results/edgar/ ---
NEWS_AGG = ROOT / "results" / "v2" / "_aggregate_18events.json"   # 18-event main corpus
EDGAR_AGG = ROOT / "results" / "edgar" / "_aggregate.json"        # 12-firm cross-corpus


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def load_agg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Headline numbers used by fig_overview (cross-checked with paper text) ---
# beta_0 collapse on the post-event negative control (naive v1 -> clock-split v2)
BETA0_V1 = 0.500
BETA0_V2 = 0.233
BETA0_N = 84            # post-event signals in the 14-day pool (tab:mechanism)

# Protocol precision / pooled FWGS on the 18-event pre-onset corpus (v2)
NEWS = load_agg(NEWS_AGG)
NEWS_FWD = NEWS["total_forward_signals"]     # 237
NEWS_TP = NEWS["total_tp"]                   # 207
NEWS_FP = NEWS["total_fp"]                   # 30
NEWS_PREC = NEWS["aggregate_precision"]      # 0.8734
NEWS_FWGS = NEWS["pooled_fwgs"]              # 0.5329
NEWS_EVENTS = NEWS["n_events"]               # 18

# Blind dual-annotation spot-check corroboration (sec:data-models), 51 adjudicated
BLIND_K, BLIND_N = 48, 51
BLIND_PREC = BLIND_K / BLIND_N               # 0.941

# Cross-corpus EDGAR (12 Chapter 11 firms, clock-split v2)
EDGAR = load_agg(EDGAR_AGG)
EDGAR_FWD = EDGAR["total_forward_signals"]   # 39
EDGAR_TP = EDGAR["total_tp"]                 # 39
EDGAR_PREC = EDGAR["aggregate_precision"]    # 1.000
EDGAR_FWGS = EDGAR["pooled_fwgs"]            # 0.7326
EDGAR_FIRMS = EDGAR["n_events"]              # 12
