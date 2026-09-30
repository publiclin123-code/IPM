#!/usr/bin/env python3
"""Finalize task2 (§4.4 spot-check) after user adjudication: conflicts -> 1.

User ruling: all 13 task2 disagreements (D=0 / Z=1) adjudicated to 1 (warning).

Steps:
  1. Load canonical forward_spotcheck_v3.csv (utf-8-sig).
  2. Load annotator D (_d.csv, gb18030) and Z (_z.xlsx).
  3. Resolve final human_label:
       - from_round1==1 : keep existing (already adjudicated).
       - from_round1==0 : d if d==z else adjudicated (all "1" here).
  4. Write final human_label back to canonical CSV.
  5. Fill `adjudicated` in the adjudication xlsx for the 13 conflict rows.
  6. Print precision (total / round1 / new) with Wilson 95% CI.

Two exclusions are applied before every statistic: one mis-dated duplicate row
(EXCLUDE) and the signals of an event later retracted from the ground truth
(EXCLUDE_EVENTS). The analysis sample is therefore n=109.
"""
import csv
import math
from pathlib import Path
import pandas as pd

ROOT = Path("/home/e/LF_projects/Risk_analysis")
ANN = ROOT / "data" / "annotation"
CANON = ANN / "forward_spotcheck_v3.csv"
D_CSV = ANN / "task2_spotcheck_v3" / "forward_spotcheck_v3_d.csv"
Z_XLS = ANN / "task2_spotcheck_v3" / "forward_spotcheck_v3_z.xlsx"
ADJ_XLS = ANN / "task2_spotcheck_v3" / "forward_spotcheck_v3_adjudication.xlsx"
ADJUDICATE = "1"  # user ruling for all conflicts

# Round-1 carry-over dropped from the v3 sample: GDELT slug mis-dated this copy
# of the 2018-03-24 pork article to 2017-03-24. Same title/description as the
# 2018-03-24 row (information duplicated), and 2017-03-24 sits far outside the
# 180-day pre-onset window [2018-01-07, 2018-07-06), so it is not a forward
# signal under the protocol either. Excluded from all v3 statistics (n=119).
EXCLUDE = {"us_china_tariff_war_2018-2017-03-24-0000-1"}

# Events retracted from the ground truth after the sample was drawn. Judging
# whether a signal is a genuine warning presupposes the event it warns about, so
# a retracted event cannot supply spot-check evidence. `toyota_steel_explosion_2019`
# was removed from validation/gt_events.json (see pipeline/prune_gt_toyota.py and
# validation/gt_excluded.json) because no source documents it; its ten signals are
# all protocol misses, so excluding them leaves the hit stratum untouched.
EXCLUDE_EVENTS = {"toyota_steel_explosion_2019"}


def read_d():
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with open(D_CSV, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            return {r["signal_id"].strip(): (r.get("human_label") or "").strip()
                    for r in rows}
        except Exception:
            continue
    raise RuntimeError("cannot decode _d.csv")


def read_z():
    df = pd.read_excel(Z_XLS)
    out = {}
    for _, r in df.iterrows():
        v = r["human_label"]
        out[str(r["signal_id"]).strip()] = "" if pd.isna(v) else str(int(v))
    return out


def wilson(tp, n, z=1.96):
    if not n:
        return None, None
    p = tp / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - hw, 4), round(c + hw, 4)


def main():
    d = read_d()
    z = read_z()
    canon = list(csv.DictReader(open(CANON, encoding="utf-8-sig")))
    fields = list(canon[0].keys())
    n_before = len(canon)
    canon = [r for r in canon
             if r["signal_id"].strip() not in EXCLUDE
             and r["event_id"].strip() not in EXCLUDE_EVENTS]

    conflicts = 0
    for r in canon:
        sid = r["signal_id"].strip()
        if r["from_round1"].strip() == "1":
            continue  # keep existing
        dl, zl = d.get(sid, ""), z.get(sid, "")
        if dl == zl:
            r["human_label"] = dl
        else:
            conflicts += 1
            r["human_label"] = ADJUDICATE  # user: 1

    # write back canonical (utf-8-sig to preserve BOM)
    with open(CANON, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in canon:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {len(canon)} rows -> {CANON}  "
          f"(conflicts resolved: {conflicts}; excluded duplicates: {n_before - len(canon)})")

    # fill adjudication xlsx
    adj = pd.read_excel(ADJ_XLS)
    for i, row in adj.iterrows():
        if row["d_label"] != row["z_label"]:
            adj.at[i, "adjudicated"] = int(ADJUDICATE)
    adj.to_excel(ADJ_XLS, index=False)
    print(f"filled adjudicated -> {ADJ_XLS}")

    # precision
    def stats(rows, label):
        n = len(rows)
        tp = sum(1 for r in rows if (r["human_label"] or "").strip() == "1")
        lo, hi = wilson(tp, n)
        print(f"  {label:10s} n={n:3d}  tp={tp:3d}  precision={tp/n:.4f}  "
              f"Wilson95 [{lo},{hi}]")
        return n, tp

    total = canon
    r1 = [r for r in canon if r["from_round1"].strip() == "1"]
    new = [r for r in canon if r["from_round1"].strip() == "0"]
    print("\n§4.4 spot-check precision (human-validated):")
    stats(total, "total")
    stats(r1, "round1")
    stats(new, "new")


if __name__ == "__main__":
    main()
