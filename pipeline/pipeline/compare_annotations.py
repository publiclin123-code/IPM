#!/usr/bin/env python3
"""Plan B todo 9: 标注完成后计算 slug-label vs body-label 翻转率与新精度估计.

用法: python compare_annotations.py   (工作表 relevant_label_body 列填好后运行)

输出:
  - 翻转矩阵: relevant->irrelevant / irrelevant->relevant / 不变 / 未标
  - 分组: 有正文 vs 无正文 (no_fetch/title_fallback)
  - body 视角下的新 pooled precision (若有正文子集)
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "annotation" / "retrieval_audit_sample.csv"
BODY = ROOT / "data" / "annotation" / "retrieval_audit_sample_body.csv"

def norm(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ("1", "relevant", "yes", "y", "true"):
        return "relevant"
    if x in ("0", "irrelevant", "no", "n", "false"):
        return "irrelevant"
    return x or None  # '' / '?' / other

def wilson(p_hat: float, n: int, z: float = 1.96):
    if not n:
        return None, None
    den = 1 + z * z / n
    c = (p_hat + z * z / (2 * n)) / den
    hw = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / den
    return round(c - hw, 3), round(c + hw, 3)

def main():
    orig = {r["article_id"]: norm(r.get("relevant_label")) for r in
            csv.DictReader(open(SRC, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(BODY, encoding="utf-8-sig")))

    n_unlabeled = flips_ri = flips_ir = same = 0
    grp = {"body": {"flip": 0, "same": 0, "ri": 0, "ir": 0},
           "nobody": {"flip": 0, "same": 0, "ri": 0, "ir": 0}}
    # body 视角 precision 计数 (仅 relevant/irrelevant 且有正文)
    prec_body = {"tp": 0, "n": 0}

    for r in rows:
        aid = r["article_id"]
        a, b = orig.get(aid), norm(r.get("relevant_label_body"))
        if b is None:
            n_unlabeled += 1
            continue
        has_body = bool((r.get("article_body") or "").strip())
        g = grp["body" if has_body else "nobody"]
        if a and b and a != b:
            if a == "relevant":
                flips_ri += 1; g["ri"] += 1
            else:
                flips_ir += 1; g["ir"] += 1
            g["flip"] += 1
            if has_body:
                prec_body["n"] += 1
                prec_body["tp"] += 1 if b == "relevant" else 0
        elif a:
            same += 1; g["same"] += 1
            if has_body and b in ("relevant", "irrelevant"):
                prec_body["n"] += 1
                prec_body["tp"] += 1 if b == "relevant" else 0

    n_decided = flips_ri + flips_ir + same
    print(f"rows={len(rows)} decided={n_decided} unlabeled={n_unlabeled}")
    print(f"flips: relevant->irrelevant={flips_ri}  irrelevant->relevant={flips_ir}  "
          f"unchanged={same}")
    for k, g in grp.items():
        print(f"  [{k:6s}] flip={g['flip']} (ri={g['ri']}, ir={g['ir']}) same={g['same']}")
    if prec_body["n"]:
        p = prec_body["tp"] / prec_body["n"]
        lo, hi = wilson(p, prec_body["n"])
        print(f"body-view precision (with-body subset): "
              f"{prec_body['tp']}/{prec_body['n']} = {p:.3f}  Wilson95 [{lo},{hi}]")

if __name__ == "__main__":
    main()
