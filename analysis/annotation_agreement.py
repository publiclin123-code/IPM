"""Human-annotation agreement + human-validated precision + FWGS re-check.

After annotators fill `human_label` in the annotation CSV (two annotators ->
two CSVs, or one CSV with two columns), this script:

  1. computes Cohen's kappa on the two annotators (if dual-annotation),
  2. computes human-validated precision = P(true warning | LLM forward signal),
  3. recomputes FWGS with human labels as the TP set, and compares the ranking
     against the automatic (date+entity) TP set.

Usage (dual annotation, two CSVs):
  python analysis/annotation_agreement.py --ann1 a1.csv --ann2 a2.csv

Usage (single resolved label, one CSV with `human_label` filled):
  python analysis/annotation_agreement.py --resolved forward_signals_annotation.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "validation"
RES = ROOT / "results" / "v2"
DATA = ROOT / "data" / "by_event"

sys.path.insert(0, str(VAL))
from validate import signal_matches_event, _parse_date  # noqa: E402
from metrics import FOREWARD_TEMPORALITIES  # noqa: E402

EVENTS = [
    "red_sea_crisis_2023", "us_chip_export_controls_2022",
    "renesas_earthquake_2016", "toyota_steel_explosion_2019",
    "port_los_angeles_backlog_2021", "covid_supply_disruption_2020",
    "europe_energy_crisis_2022", "renesas_naka_plant_fire_2021",
]
WINDOW = 180
TAU, LAM = 30.0, 0.5


def load_gt() -> list[dict]:
    with open(VAL / "gt_events.json", encoding="utf-8") as f:
        return json.load(f)["events"]


def load_news(eid: str) -> dict:
    p = DATA / f"{eid}.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r.get("id", "")] = r
    return out


def read_labels(path: Path) -> dict[str, str]:
    """signal_id -> human_label (1 / 0 / ?)."""
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row.get("signal_id", "").strip()
            lab = row.get("human_label", "").strip()
            if sid and lab:
                labels[sid] = lab
    return labels


def kappa(a: dict[str, str], b: dict[str, str]) -> float:
    """Cohen's kappa over shared keys, treating 1/0/? as nominal categories."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    cats = ["1", "0", "?"]
    n = len(keys)
    n_agree = sum(1 for k in keys if a[k] == b[k])
    p_o = n_agree / n
    p_e = 0.0
    for c in cats:
        p_a = sum(1 for k in keys if a[k] == c) / n
        p_b = sum(1 for k in keys if b[k] == c) / n
        p_e += p_a * p_b
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def fwgs_with_tp(signals: list[dict], news: dict, gt: list[dict],
                 human_labels: dict[str, str]) -> dict:
    """Recompute FWGS using human labels as the TP flag (1=true warning)."""
    scores = []
    n_tp = n_fp = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        lead = 0
        # lead is still the automatic date-based lead (onset - signal_date)
        for ev in gt:
            try:
                onset = _parse_date(ev["gt_onset_date"])
            except (KeyError, ValueError):
                continue
            l = (onset - sd).days
            if 0 < l <= WINDOW and signal_matches_event(s, ev, strict=False):
                lead = max(lead, l)
        lab = human_labels.get(s.get("signal_id", ""), "?")
        is_tp = (lab == "1")
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        w_f = 1 - math.exp(-lead / TAU)
        triggers = s.get("trigger_phrases", []) or []
        article = news.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        g = sum(1 for t in triggers if t.lower() in text) / len(triggers) if triggers else 0.0
        conf = float(s.get("confidence", 0))
        scores.append(w_f * g * (1 if is_tp else 0) - LAM * (1 - g) * conf)
    return {"fwgs": sum(scores) / len(scores) if scores else None,
            "n": len(scores), "n_tp": n_tp, "n_fp": n_fp}


def auto_tp(signals: list[dict], gt: list[dict]) -> dict[str, bool]:
    """signal_id -> automatic TP (date + entity match)."""
    out = {}
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        is_tp = False
        for ev in gt:
            try:
                onset = _parse_date(ev["gt_onset_date"])
            except (KeyError, ValueError):
                continue
            if 0 < (onset - sd).days <= WINDOW and signal_matches_event(s, ev, strict=False):
                is_tp = True
                break
        out[s.get("signal_id", "")] = is_tp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann1", default="")
    ap.add_argument("--ann2", default="")
    ap.add_argument("--resolved", default="")
    args = ap.parse_args()

    gt = load_gt()
    # load all forward signals across events
    all_sigs: list[dict] = []
    for eid in EVENTS:
        p = RES / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") == "error":
                continue
            if s.get("temporality") in FOREWARD_TEMPORALITIES:
                all_sigs.append(s)
    news_by_event = {eid: load_news(eid) for eid in EVENTS}
    def news_for(s):
        return news_by_event.get(s.get("input_id", "").split("-")[0], {})

    # ---- dual annotation kappa ----
    if args.ann1 and args.ann2:
        a = read_labels(Path(args.ann1))
        b = read_labels(Path(args.ann2))
        k = kappa(a, b)
        print(f"=== Cohen's kappa (dual annotation) ===")
        print(f"  annotator 1: {len(a)} labels | annotator 2: {len(b)} labels")
        print(f"  shared keys : {len(set(a) & set(b))}")
        print(f"  kappa       : {k:.3f}")
        # disagreement list
        dis = [k_ for k_ in sorted(set(a) & set(b)) if a[k_] != b[k_]]
        print(f"  disagreements: {len(dis)}")
        for k_ in dis[:20]:
            print(f"    {k_} : ann1={a[k_]} ann2={b[k_]}")
        return 0

    # ---- resolved single label ----
    if args.resolved:
        labels = read_labels(Path(args.resolved))
        n1 = sum(1 for v in labels.values() if v == "1")
        n0 = sum(1 for v in labels.values() if v == "0")
        nq = sum(1 for v in labels.values() if v == "?")
        print(f"=== Resolved human labels ===")
        print(f"  1 (true warning) : {n1}")
        print(f"  0 (not warning)  : {n0}")
        print(f"  ? (uncertain)    : {nq}")
        print(f"  human precision  : {n1}/{n1+n0} = {n1/(n1+n0):.3f}  (excluding '?')")

        # per-event human precision + FWGS(human TP) vs auto
        print(f"\n{'event':<36} {'hum_tp':>6} {'hum_n':>6} {'FWGS_hum':>9} {'FWGS_auto':>10}")
        auto_map = {}
        # rebuild per-event signals
        per_event = {}
        for eid in EVENTS:
            sigs = [s for s in all_sigs if s.get("input_id", "").startswith(eid)]
            per_event[eid] = sigs
        for eid in EVENTS:
            sigs = per_event[eid]
            if not sigs:
                continue
            news = news_by_event[eid]
            r_hum = fwgs_with_tp(sigs, news, gt, labels)
            # auto FWGS (reuse existing three_model logic inline)
            # auto TP set
            auto = auto_tp(sigs, gt)
            # compute auto FWGS with auto TP
            r_auto = fwgs_with_tp(sigs, news, gt, {sid: ("1" if v else "0") for sid, v in auto.items()})
            fh = f"{r_hum['fwgs']:.3f}" if r_hum["fwgs"] is not None else "--"
            fa = f"{r_auto['fwgs']:.3f}" if r_auto["fwgs"] is not None else "--"
            print(f"{eid:<36} {r_hum['n_tp']:>6} {r_hum['n']:>6} {fh:>9} {fa:>10}")

        # pooled human FWGS
        all_scores = []
        for eid in EVENTS:
            sigs = per_event[eid]
            if not sigs:
                continue
            r = fwgs_with_tp(sigs, news_by_event[eid], gt, labels)
            if r["fwgs"] is not None:
                all_scores.append((r["fwgs"], r["n"]))
        tot_fw = sum(f * n for f, n in all_scores)
        tot_n = sum(n for _, n in all_scores)
        print(f"\nPooled FWGS (human TP): {tot_fw/tot_n:.3f}  (n={tot_n})")
        return 0

    print("Provide --ann1/--ann2 for kappa, or --resolved for human precision.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
