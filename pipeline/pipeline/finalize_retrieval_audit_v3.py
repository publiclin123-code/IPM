#!/usr/bin/env python3
"""Finalize task1 (§4.3 retrieval audit) adjudication.

Adjudicator rulings for the 47 D/Z disagreements, then write the final
relevant_label_body back to the canonical worksheet and run the flip-rate /
precision analysis (same logic as compare_annotations.py).

Label rule for every row: d if d==z else ADJUDICATION[article_id].
"""
import csv
import math
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path("/home/e/LF_projects/Risk_analysis")
ANN = ROOT / "data" / "annotation"
CANON = ANN / "retrieval_audit_sample_body.csv"          # symlink -> task1 folder
D_CSV = ANN / "task1_retrieval_audit_body" / "retrieval_audit_sample_body_d.csv"
Z_XLS = ANN / "task1_retrieval_audit_body" / "retrieval_audit_sample_body_z.xlsx"
ADJ_XLS = ANN / "task1_retrieval_audit_body" / "retrieval_audit_sample_body_adjudication.xlsx"
SRC_TITLE = ANN / "retrieval_audit_sample.csv"           # title-round labels

# Adjudicator rulings for the 47 disagreements (1=relevant, 0=irrelevant).
# Commented reason kept per row for traceability.
ADJUDICATION = {
    # Beirut port explosion
    "beirut_port_explosion_2020-2020-07-17-0000": 0,     # protest/politics
    # Black Sea grain exit
    "black_sea_grain_exit_2023-2023-04-21-0000": 1,     # FLAG: Ukraine grain glut, CEE
    # COVID supply disruption
    "covid_supply_disruption_2020-2020-01-22-0000": 1,  # ADJUDICATOR: relevant (first US case)
    "covid_supply_disruption_2020-2020-01-22-0011": 0,  # health only
    # US egg shortage / H5N1  (all health / dairy / pet-food / local — not egg supply)
    "egg_shortage_birdflu_2025-2024-08-20-0000": 0,     # dairy cattle
    "egg_shortage_birdflu_2025-2024-09-07-0000": 0,     # human case
    "egg_shortage_birdflu_2025-2024-09-13-0000": 0,     # pandemic risk
    "egg_shortage_birdflu_2025-2024-11-01-0000": 0,     # human cases
    "egg_shortage_birdflu_2025-2024-12-12-0002": 0,     # raw milk / child
    "egg_shortage_birdflu_2025-2024-12-19-0001": 0,     # severe case
    "egg_shortage_birdflu_2025-2024-12-26-0000": 0,     # pet food recall
    "egg_shortage_birdflu_2025-2025-01-07-0005": 0,     # first death
    "egg_shortage_birdflu_2025-2025-01-07-0007": 0,     # death
    "egg_shortage_birdflu_2025-2025-01-11-0000": 0,     # local birds Canada
    # European energy crisis
    "europe_energy_crisis_2022-2022-03-25-0000": 1,     # FLAG: title relevant, body fetch error
    "europe_energy_crisis_2022-2022-03-30-0000": 1,     # FLAG: title relevant, body fetch error
    "europe_energy_crisis_2022-2022-06-01-0000": 0,     # FLAG: Japan Sakhalin, not Europe
    "europe_energy_crisis_2022-2022-07-22-0001": 1,     # Europe LNG -> Pakistan spillover
    "europe_energy_crisis_2022-2022-07-25-0001": 1,     # Nord Stream 1
    # Hurricane Maria / PR pharma
    "hurricane_maria_2017-2017-09-19-0006": 1,           # storm heading to PR
    "hurricane_maria_2017-2017-09-19-0008": 0,           # FLAG: Dominica, not PR
    "hurricane_maria_2017-2017-09-19-0009": 1,           # PR state of emergency
    "hurricane_maria_2017-2017-09-19-0014": 0,           # FLAG: St Thomas, not PR
    "hurricane_maria_2017-2017-09-19-0018": 1,           # storm track to PR
    "hurricane_maria_2017-2017-09-19-0022": 1,           # barrels toward PR
    # India wheat export ban
    "india_wheat_export_ban_2022-2022-02-26-0000": 0,   # aid shipment to Afghanistan
    "india_wheat_export_ban_2022-2022-04-15-0001": 1,   # FLAG: India wheat export precursor
    "india_wheat_export_ban_2022-2022-05-11-0000": 1,   # FLAG: record exports, precursor
    # LA/Long Beach port congestion
    "port_los_angeles_backlog_2021-2021-05-01-0000": 0, # generic trends
    "port_los_angeles_backlog_2021-2021-07-13-0001": 0, # professional services
    "port_los_angeles_backlog_2021-2021-08-19-0000": 1, # FLAG: retailers charter boats
    "port_los_angeles_backlog_2021-2021-08-30-0000": 0, # FLAG: global workers, not LA
    # Red Sea shipping crisis
    "red_sea_crisis_2023-2023-06-04-0001": 0,           # Suez tanker breakdown
    "red_sea_crisis_2023-2023-06-12-0000": 0,           # Yemen aid
    "red_sea_crisis_2023-2023-07-16-0000": 0,           # FSO Safer oil spill
    "red_sea_crisis_2023-2023-09-02-0000": 0,           # resource looting
    "red_sea_crisis_2023-2023-09-30-0000": 0,           # Houthi attack on Bahrain, not shipping
    # Russia-Ukraine war
    "russia_ukraine_war_2022-2022-02-07-0006": 0,       # Biden-Bennett phone call
    "russia_ukraine_war_2022-2022-02-19-0009": 0,       # Kazakhstan internet
    "russia_ukraine_war_2022-2022-02-23-0012": 0,       # Gallup poll
    # Toyota steel explosion
    "toyota_steel_explosion_2019-2018-12-28-0000": 0,   # home robots
    "toyota_steel_explosion_2019-2019-01-07-0002": 0,   # fishing program
    # US-China tariff war
    "us_china_tariff_war_2018-2018-03-20-0000": 1,      # Reuters $60B tariffs
    "us_china_tariff_war_2018-2018-03-26-0001": 0,      # Thai tourism stocks
    "us_china_tariff_war_2018-2018-05-15-0002": 0,      # Qualcomm/ZTE M&A
    "us_china_tariff_war_2018-2018-06-17-0001": 1,      # FLAG: soybeans tariff, body access-denied
    # US chip export controls
    "us_chip_export_controls_2022-2022-07-14-0002": 0,  # TSMC earnings
}


def read_d():
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252"):
        try:
            with open(D_CSV, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            return {r["article_id"].strip(): (r.get("relevant_label_body") or "").strip()
                    for r in rows}
        except Exception:
            continue
    raise RuntimeError("cannot decode _d.csv")


def read_z():
    df = pd.read_excel(Z_XLS)
    out = {}
    for _, r in df.iterrows():
        v = r["relevant_label_body"]
        out[str(r["article_id"]).strip()] = "" if pd.isna(v) else str(int(v))
    return out


def wilson(tp, n, z=1.96):
    if not n:
        return None, None
    p = tp / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - hw, 4), round(c + hw, 4)


def norm(x):
    x = (x or "").strip().lower()
    return "relevant" if x in ("1", "relevant", "yes", "true") else \
           "irrelevant" if x in ("0", "irrelevant", "no", "false") else None


def main():
    d = read_d()
    z = read_z()
    canon = list(csv.DictReader(open(CANON, encoding="utf-8")))
    fields = list(canon[0].keys())

    resolved_conflicts = 0
    missing_rulings = []
    for r in canon:
        aid = r["article_id"].strip()
        dl, zl = d.get(aid, ""), z.get(aid, "")
        if dl == zl:
            r["relevant_label_body"] = dl
        elif aid in ADJUDICATION:
            r["relevant_label_body"] = str(ADJUDICATION[aid])
            resolved_conflicts += 1
        else:
            missing_rulings.append(aid)
            r["relevant_label_body"] = ""  # leave blank

    with open(CANON, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in canon:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {len(canon)} rows -> {CANON}")
    print(f"  conflicts resolved by adjudicator: {resolved_conflicts}")
    if missing_rulings:
        print("  MISSING rulings:", missing_rulings)

    # fill adjudication xlsx
    adj = pd.read_excel(ADJ_XLS)
    for i, row in adj.iterrows():
        aid = str(row["article_id"]).strip()
        if row["d_label"] != row["z_label"] and aid in ADJUDICATION:
            adj.at[i, "adjudicated"] = int(ADJUDICATION[aid])
    adj.to_excel(ADJ_XLS, index=False)
    print(f"filled adjudicated -> {ADJ_XLS}")

    # ---- flip-rate / precision analysis (title_final_label vs relevant_label_body) ----
    n_unlabeled = flips_ri = flips_ir = same = 0
    prec = {"tp": 0, "n": 0}
    prec_all = {"tp": 0, "n": 0}
    for r in canon:
        a = norm(r.get("title_final_label"))     # title-round human label
        b = norm(r.get("relevant_label_body"))   # body-round final label
        if b is None:
            n_unlabeled += 1
            continue
        prec_all["n"] += 1
        prec_all["tp"] += 1 if b == "relevant" else 0
        has_body = bool((r.get("article_body") or "").strip())
        if a and a != b:
            if a == "relevant":
                flips_ri += 1
            else:
                flips_ir += 1
            if has_body:
                prec["n"] += 1
                prec["tp"] += 1 if b == "relevant" else 0
        elif a:
            same += 1
            if has_body:
                prec["n"] += 1
                prec["tp"] += 1 if b == "relevant" else 0

    decided = flips_ri + flips_ir + same
    print(f"\nrows={len(canon)} decided={decided} unlabeled={n_unlabeled}")
    print(f"flips: relevant->irrelevant={flips_ri}  irrelevant->relevant={flips_ir}  "
          f"unchanged={same}")
    print(f"flip rate (either direction): {(flips_ri + flips_ir) / len(canon):.3f}")
    if prec["n"]:
        p = prec["tp"] / prec["n"]
        lo, hi = wilson(prec["tp"], prec["n"])
        print(f"body-view precision (with-body subset n={prec['n']}): "
              f"{prec['tp']}/{prec['n']} = {p:.4f}  Wilson95 [{lo},{hi}]")
    if prec_all["n"]:
        p = prec_all["tp"] / prec_all["n"]
        lo, hi = wilson(prec_all["tp"], prec_all["n"])
        print(f"body-view precision (all n={prec_all['n']}): "
              f"{prec_all['tp']}/{prec_all['n']} = {p:.4f}  Wilson95 [{lo},{hi}]")


if __name__ == "__main__":
    main()
