#!/usr/bin/env python3
"""Fig 7 (four-model redesign) — Events x models FWGS heatmap with qwen bootstrap CI.

Rows = 15 events (qwen FWGS non-null), ordered by qwen FWGS desc; columns = four
models. Each cell shows the point estimate; the qwen column also shows its
bootstrap 95% CI in small grey text (compensates the removed CI table). Uses the
human-readable event labels from event_meta.EVENT_SHORT.

Usage: python analysis/fig7_heatmap_4model.py --json /tmp/fwgs_4model.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation"))
from figstyle import TOL, apply_style, save  # noqa: E402
from event_meta import EVENT_SHORT  # noqa: E402
from validate import load_signals, FOREWARD_TEMPORALITIES, signal_matches_event, _parse_date  # noqa: E402
from metrics import load_jsonl  # noqa: E402

MODELS = ["qwen3.6-27b", "gemma4-31b", "muse-glimmer-30b", "deepseek-v4-flash"]
MODEL_DIRS = {
    "qwen3.6-27b": ROOT / "results" / "v2",
    "gemma4-31b": ROOT / "results" / "v2" / "gemma4",
    "muse-glimmer-30b": ROOT / "results" / "v2" / "muse",
    "deepseek-v4-flash": ROOT / "results" / "deepseek" / "gdelt",
}
GT = ROOT / "validation" / "gt_events.json"
DATA = ROOT / "data" / "by_event"
TAU, LAM = 30.0, 0.5
B = 2000
SEED = 42


def fwgs_single(signals, news, ev):
    scores = []
    onset = _parse_date(ev["gt_onset_date"])
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        lead = (onset - sd).days
        is_tp = 0 < lead <= 180 and signal_matches_event(s, ev, strict=False)
        w_f = 1 - np.exp(-lead / TAU) if lead > 0 else 0.0
        triggers = s.get("trigger_phrases", []) or []
        art = news.get(s.get("input_id", ""), {})
        text = ((art.get("title", "") or "") + " " + (art.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return sum(scores) / len(scores) if scores else None


def bootstrap_ci(signals, news, ev):
    fwd = [s for s in signals if s.get("temporality") in FOREWARD_TEMPORALITIES]
    if not fwd:
        return None
    rng = random.Random(SEED)
    ests = []
    for _ in range(B):
        sample = [fwd[rng.randrange(len(fwd))] for _ in range(len(fwd))]
        v = fwgs_single(sample, news, ev)
        ests.append(v if v is not None else 0.0)
    ests.sort()
    return ests[int(0.025 * B)], ests[int(0.975 * B)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/fwgs_4model.json")
    args = ap.parse_args()

    data = json.load(open(args.json))
    rows = data["rows"]
    apply_style()

    gt_events = json.load(open(GT, encoding="utf-8"))["events"]
    gt_by_id = {e["event_id"]: e for e in gt_events}

    valid = [r for r in rows if r["fwgs"]["qwen3.6-27b"] is not None]
    valid.sort(key=lambda r: -r["fwgs"]["qwen3.6-27b"])

    n = len(valid)
    pts = np.zeros((n, 4))
    labels = []
    qwen_ci = []
    for r in valid:
        eid = r["eid"]
        labels.append(EVENT_SHORT.get(eid, eid))
        for j, m in enumerate(MODELS):
            pts[len(labels) - 1, j] = r["fwgs"][m] if r["fwgs"][m] is not None else np.nan
        # qwen bootstrap CI
        news_path = DATA / f"{eid}.jsonl"
        news = {a["id"]: a for a in (load_jsonl(str(news_path)) if news_path.exists() else [])}
        sp = MODEL_DIRS["qwen3.6-27b"] / f"{eid}_signals.jsonl"
        sigs = [s for s in load_signals(str(sp)) if s.get("status") != "error"] if sp.exists() else []
        qwen_ci.append(bootstrap_ci(sigs, news, gt_by_id[eid]))

    fig, ax = plt.subplots(figsize=(6.2, max(4.4, n * 0.30)))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "fwgs_blue", ["#EAF2FB", "#A9CCE3", "#5499C7", TOL["blue"], "#1A4E7A"])
    im = ax.imshow(pts, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(4):
            pt = pts[i, j]
            if np.isnan(pt):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="#9AA5B1")
                continue
            txt_color = "white" if pt > 0.55 else "#1F2937"
            ax.text(j, i, f"{pt:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt_color)
            if j == 0 and qwen_ci[i] is not None:
                lo, hi = qwen_ci[i]
                ax.text(j, i + 0.30, f"[{lo:.2f},{hi:.2f}]", ha="center", va="center",
                        fontsize=5.5, color="white" if pt > 0.55 else "#6B7280")

    ax.set_xticks(range(4))
    ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FWGS", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig7_heatmap")
    plt.close(fig)
    print(f"heatmap: {n} events x 4 models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

