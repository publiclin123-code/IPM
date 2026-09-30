#!/usr/bin/env python3
"""Independent audit: recompute every headline number from raw files.

Deliberately does NOT read any _aggregate_*.json cache or
compare_title_body.json. Everything is recomputed from:
  - results/v2/*_signals.jsonl       (title extraction)
  - results/v2_body/*_signals.jsonl  (body extraction)
  - data/by_event_body/*.jsonl       (corpus rows + bodies)
  - data/by_event_body/_background.jsonl (BSCC pool)
  - validation/gt_events.json        (ground truth)
  - results/_background/beta0_body_summary.json (β0: recomputed below)
Cross-checks against the cached compare_title_body.json at the end.
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "validation"))
from validate import load_signals  # noqa: E402
from metrics import compute_fwgs, compute_bscc, _build_article_event_map  # noqa: E402

errors = []

# ---------- 1. corpus coverage ----------
rows = 0; rows_ok = 0
url_ok: dict[str, bool] = {}
eid_rows: dict[str, int] = {}
body_status = Counter()
for p in (ROOT / "data" / "by_event_body").glob("*.jsonl"):
    if "_background" in p.name:
        continue
    eid = p.stem
    eid_rows[eid] = 0
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        rows += 1
        eid_rows[eid] += 1
        bs = r.get("body_status")
        body_status[bs] += 1
        has = bs != "title_fallback" and bool(r.get("text"))
        rows_ok += has
        u = (r.get("url") or "").strip()
        if u:
            url_ok[u] = url_ok.get(u, False) or has
cov_row = rows_ok / rows
cov_url = sum(url_ok.values()) / len(url_ok)
print(f"[1] corpus: rows {rows_ok}/{rows} = {cov_row*100:.1f}% | "
      f"urls {sum(url_ok.values())}/{len(url_ok)} = {cov_url*100:.1f}%")
if abs(cov_row - 0.915) > 0.001 or abs(cov_url - 0.915) > 0.001:
    errors.append(f"coverage mismatch: {cov_row:.4f}/{cov_url:.4f} vs 0.915")
if rows != 2555:
    errors.append(f"row count {rows} != 2555")
print(f"    body_status: {dict(body_status)}")

# ---------- 2. title vs body aggregates (raw recompute) ----------
gt = json.load(open(ROOT / "validation" / "gt_events.json", encoding="utf-8"))["events"]

def recompute(res_dir: Path, data_dir: Path, label: str) -> dict:
    tot_sig = tot_tp = tot_fp = 0
    hits = 0; n_events = 0
    fwgs_wsum = 0.0
    per = []
    for ev in gt:
        eid = ev["event_id"]
        sp = res_dir / f"{eid}_signals.jsonl"
        if not sp.exists():
            # absent file (designed negative, e.g. Suez with 0 articles): count
            # as no-signal no-hit event, matching aggregate_events.py
            n_events += 1
            per.append((eid, 0, 0, 0, None, False))
            continue
        sigs = load_signals(str(sp))
        news = {a["id"]: a for a in
                (json.loads(l) for l in open(data_dir / f"{eid}.jsonl", encoding="utf-8"))
                if data_dir.joinpath(f"{eid}.jsonl").exists()}
        fw = compute_fwgs(sigs, news, [ev], window_days=180)
        tot_sig += fw["n_signals"]; tot_tp += fw["n_tp"]; tot_fp += fw["n_fp"]
        fwgs_wsum += (fw["fwgs"] or 0.0) * fw["n_signals"]
        n_events += 1
        # hit: validate
        from validate import validate
        val = validate(sigs, [ev], window_days=180, strict=False)
        per_ev = next((x for x in val["per_event"] if x["event_id"] == eid), None)
        if per_ev and per_ev.get("hit"):
            hits += 1
        per.append((eid, fw["n_signals"], fw["n_tp"], fw["n_fp"], fw["fwgs"],
                    bool(per_ev and per_ev.get("hit"))))
    pooled_fwgs = fwgs_wsum / tot_sig if tot_sig else None
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else None
    print(f"[2] {label}: n_sig={tot_sig} TP={tot_tp} FP={tot_fp} "
          f"prec={prec:.4f} FWGS={pooled_fwgs:.4f} hits={hits}/{n_events}")
    return {"n": tot_sig, "tp": tot_tp, "fp": tot_fp, "prec": prec,
            "fwgs": pooled_fwgs, "hits": hits, "n_events": n_events, "per": per}

T = recompute(ROOT / "results" / "v2", ROOT / "data" / "by_event", "title(raw)")
B = recompute(ROOT / "results" / "v2_body", ROOT / "data" / "by_event_body", "body(raw)")

# ---------- 3. cross-check against cache ----------
ref = json.load(open(ROOT / "results" / "compare_title_body.json", encoding="utf-8"))
rt, rb = ref["title_only"], ref["body"]
for got, r, label in [(T, rt, "title"), (B, rb, "body")]:
    if (got["n"], got["tp"], got["fp"]) != (r["total_forward_signals"], r["total_tp"], r["total_fp"]):
        errors.append(f"{label} mismatch vs cache: raw {got['n']}/{got['tp']}/{got['fp']} "
                      f"cache {r['total_forward_signals']}/{r['total_tp']}/{r['total_fp']}")
    else:
        print(f"    {label}: raw == cache ✓ ({got['n']}/{got['tp']}/{got['fp']})")
    if abs(got["prec"] - r["aggregate_precision"]) > 0.002:
        errors.append(f"{label} prec raw {got['prec']:.4f} vs cache {r['aggregate_precision']:.4f}")
    if got["fwgs"] is not None and r["pooled_fwgs"] is not None and abs(got["fwgs"] - r["pooled_fwgs"]) > 0.005:
        errors.append(f"{label} fwgs raw {got['fwgs']:.4f} vs cache {r['pooled_fwgs']:.4f}")

# ---------- 4. subset sensitivity (body-available only, both sides) ----------
avail: dict[str, set[str]] = {}
for p in (ROOT / "data" / "by_event_body").glob("*.jsonl"):
    if "_background" in p.name:
        continue
    s = set()
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("body_status") != "title_fallback" and r.get("text"):
            s.add(r["id"])
    avail[p.stem] = s

def subset_pooled(res_dir: Path, label: str) -> tuple:
    tot_sig = tot_tp = 0; fwgs_wsum = 0.0
    for ev in gt:
        eid = ev["event_id"]
        sp = res_dir / f"{eid}_signals.jsonl"
        if not sp.exists():
            continue
        ids = avail.get(eid, set())
        sigs = [s for s in load_signals(str(sp)) if s.get("input_id") in ids]
        news = {a["id"]: a for a in (json.loads(l) for l in open(ROOT / "data" / "by_event_body" / f"{eid}.jsonl", encoding="utf-8")) if a["id"] in ids}
        fw = compute_fwgs(sigs, news, [ev], window_days=180)
        tot_sig += fw["n_signals"]; tot_tp += fw["n_tp"]
        fwgs_wsum += (fw["fwgs"] or 0.0) * fw["n_signals"]
    prec = tot_tp / tot_sig if tot_sig else None
    fwgs = fwgs_wsum / tot_sig if tot_sig else None
    print(f"[4] subset({label}): n_sig={tot_sig} TP={tot_tp} prec={prec:.4f} FWGS={fwgs:.4f}")
    return tot_sig, tot_tp, prec, fwgs

ST = subset_pooled(ROOT / "results" / "v2", "title")
SB = subset_pooled(ROOT / "results" / "v2_body", "body")
# sanity: subset n + title_fallback n == full n
n_fb_forward = B["n"] - SB[0]
print(f"    title_fallback rows contribute {n_fb_forward} forward signals "
      f"(full {B['n']} - subset {SB[0]})")
if B["n"] - SB[0] < 0:
    errors.append("subset > full for body!")

# ---------- 5. beta0 recompute from raw negative-control signal files ----------
def beta0_from_file(path: Path, label: str):
    """β₀ via the canonical estimator.

    This previously counted every row in the file as the denominator, ignoring
    both the 14-day post-onset window and the exclusion of onset-day signals.
    Because the background probe is fetched with post_days=60, the files also
    contain signals from days 15-60, so the old version computed a different
    quantity than the paper reports and raised a false mismatch on every run.
    Reusing compute_bscc keeps the check honest: a checker that reimplements the
    metric it is checking tests the reimplementation, not the metric.
    """
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gt_events = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]
    id2event = _build_article_event_map(
        ROOT / "data" / "by_event" / "_background.jsonl", gt_events, post_days=60)
    m = compute_bscc(rows, gt_events, post_days=14, id2event=id2event)
    b = m["beta_0_hindsight_rate"]
    mm = m["n_temporal_mismatch"]
    n_strict = m["n_strictly_post_onset"]
    print(f"[5] β0({label}) from {path.name}: {b:.4f} ({mm}/{n_strict}) "
          f"[onset-day excluded: {m['n_onset_day_ambiguous']}; file rows: {len(rows)}]")
    return b, n_strict

b0n, nn = beta0_from_file(ROOT / "results" / "_background" / "signals_postevent_naive_body.jsonl", "body-naive")
b0v, nv = beta0_from_file(ROOT / "results" / "_background" / "signals_postevent_v2_body.jsonl", "body-v2")
ref_b0 = json.load(open(ROOT / "results" / "_background" / "beta0_body_summary.json", encoding="utf-8"))
print(f"    summary: naive {ref_b0['body_naive']['beta_0_hindsight_rate']:.4f} "
      f"(n={ref_b0['body_naive']['n_strictly_post_onset']}) | "
      f"v2 {ref_b0['body_v2']['beta_0_hindsight_rate']:.4f} (n={ref_b0['body_v2']['n_strictly_post_onset']})")
if abs(b0n - ref_b0['body_naive']['beta_0_hindsight_rate']) > 0.001 or nn != ref_b0['body_naive']['n_strictly_post_onset']:
    errors.append(f"β0 naive mismatch: raw {b0n:.4f}/{nn} vs summary {ref_b0['body_naive']['beta_0_hindsight_rate']:.4f}/{ref_b0['body_naive']['n_strictly_post_onset']}")
if abs(b0v - ref_b0['body_v2']['beta_0_hindsight_rate']) > 0.001 or nv != ref_b0['body_v2']['n_strictly_post_onset']:
    errors.append(f"β0 v2 mismatch: raw {b0v:.4f}/{nv} vs summary {ref_b0['body_v2']['beta_0_hindsight_rate']:.4f}/{ref_b0['body_v2']['n_strictly_post_onset']}")

# ---------- 6. worksheet backfill sanity ----------
import csv
ws = list(csv.DictReader(open(ROOT / "data" / "annotation" / "retrieval_audit_sample_body.csv", encoding="utf-8")))
n_reopen = sum(1 for r in ws if r["body_status"] not in ("ok_jina",) and not r["relevant_label_body"].strip())
print(f"[6] worksheet: {len(ws)} rows; body_status {Counter(r['body_status'] for r in ws)}")
print(f"    unlabeled relevant_label_body: {sum(1 for r in ws if not r['relevant_label_body'].strip())}")

print("\n" + ("ALL CHECKS PASSED ✓" if not errors else "ERRORS:\n- " + "\n- ".join(errors)))
