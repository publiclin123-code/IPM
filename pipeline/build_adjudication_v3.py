#!/usr/bin/env python3
"""Build adjudication tables from the two annotators' files (_d.csv / _z.xlsx).

For each task, produce an xlsx with the two labels side-by-side plus an empty
`adjudicated` column. Text columns are pulled from the canonical UTF-8 CSVs
(avoiding the GBK/cp1252 mojibake in the user's _d.csv). Disagreement rows are
sorted to the top for easy adjudication.

Outputs:
  data/annotation/task1_retrieval_audit_body/retrieval_audit_sample_body_adjudication.xlsx
  data/annotation/task2_spotcheck_v3/forward_spotcheck_v3_adjudication.xlsx
"""
import csv
from pathlib import Path
import pandas as pd

ROOT = Path("/home/e/LF_projects/Risk_analysis")
ANN = ROOT / "data" / "annotation"


def read_canon(path, enc="utf-8-sig"):
    return list(csv.DictReader(open(path, encoding=enc)))


def read_d_labels(path, id_col, label_col):
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252"):
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            return {r[id_col].strip(): (r.get(label_col) or "").strip() for r in rows}
        except Exception:
            continue
    return {}


def read_z_labels(path, id_col, label_col):
    df = pd.read_excel(path)
    out = {}
    for _, r in df.iterrows():
        rid = str(r[id_col]).strip()
        v = r.get(label_col)
        out[rid] = "" if pd.isna(v) else str(int(v)) if float(v) == int(v) else str(v)
    return out


def build(task1=True, task2=True):
    if task1:
        canon = read_canon(ANN / "retrieval_audit_sample_body.csv", enc="utf-8")
        by_id = {r["article_id"].strip(): r for r in canon}
        d = read_d_labels(ANN / "task1_retrieval_audit_body/retrieval_audit_sample_body_d.csv",
                          "article_id", "relevant_label_body")
        z = read_z_labels(ANN / "task1_retrieval_audit_body/retrieval_audit_sample_body_z.xlsx",
                          "article_id", "relevant_label_body")
        out = []
        for r in canon:
            aid = r["article_id"].strip()
            out.append({
                "article_id": aid,
                "event_name": r.get("event_name", ""),
                "article_date": r.get("article_date", ""),
                "article_title": r.get("article_title", ""),
                "article_body": (r.get("article_body") or "")[:800],
                "body_status": r.get("body_status", ""),
                "llm_suggest": r.get("llm_suggest", ""),
                "llm_reason": (r.get("llm_reason") or "")[:200],
                "d_label": d.get(aid, ""),
                "z_label": z.get(aid, ""),
                "adjudicated": "",
            })
        out.sort(key=lambda x: (x["d_label"] == x["z_label"], x["article_id"]))
        df = pd.DataFrame(out)
        dst = ANN / "task1_retrieval_audit_body/retrieval_audit_sample_body_adjudication.xlsx"
        df.to_excel(dst, index=False)
        dis = sum(1 for x in out if x["d_label"] != x["z_label"])
        print(f"task1 -> {dst}  rows={len(df)}  disagreements={dis}")

    if task2:
        canon = read_canon(ANN / "forward_spotcheck_v3.csv", enc="utf-8-sig")
        d = read_d_labels(ANN / "task2_spotcheck_v3/forward_spotcheck_v3_d.csv",
                          "signal_id", "human_label")
        z = read_z_labels(ANN / "task2_spotcheck_v3/forward_spotcheck_v3_z.xlsx",
                          "signal_id", "human_label")
        out = []
        for r in canon:
            sid = r["signal_id"].strip()
            out.append({
                "signal_id": sid,
                "event_id": r.get("event_id", ""),
                "signal_date": r.get("signal_date", ""),
                "from_round1": r.get("from_round1", ""),
                "description": r.get("description", ""),
                "trigger_phrases": r.get("trigger_phrases", ""),
                "article_title": r.get("article_title", ""),
                "llm_suggest": r.get("llm_suggest", ""),
                "llm_reason": (r.get("llm_reason") or "")[:200],
                "d_label": d.get(sid, ""),
                "z_label": z.get(sid, ""),
                "adjudicated": "",
            })
        out.sort(key=lambda x: (x["from_round1"] != "0" or x["d_label"] == x["z_label"], x["signal_id"]))
        df = pd.DataFrame(out)
        dst = ANN / "task2_spotcheck_v3/forward_spotcheck_v3_adjudication.xlsx"
        df.to_excel(dst, index=False)
        dis = sum(1 for x in out if x["d_label"] != x["z_label"])
        print(f"task2 -> {dst}  rows={len(df)}  disagreements={dis}")


if __name__ == "__main__":
    build()
