#!/usr/bin/env python3
"""Build the body-based re-annotation worksheet (Plan B, todo 9).

Reads  data/annotation/retrieval_audit_sample.csv   (389 rows, title-only)
       data/body_fetch/bodies.jsonl                 (fetched bodies)
Writes data/annotation/retrieval_audit_sample_body.csv
        - article_body: fetched body capped at 1500 chars (annotation-friendly),
          empty string when fetch failed
        - body_status / body_len for transparency
        - fresh empty columns relevant_label_body / note_body for annotators
"""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "annotation" / "retrieval_audit_sample.csv"
DST = ROOT / "data" / "annotation" / "retrieval_audit_sample_body.csv"
FETCH = ROOT / "data" / "body_fetch" / "bodies.jsonl"
CAP = 1500

def uh(u): return hashlib.sha1(u.encode()).hexdigest()[:16]

best = {}
for line in FETCH.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if not r["status"].startswith("ok"):
        continue
    old = best.get(r["url_hash"])
    if old is None or r["text_len"] > old["text_len"]:
        best[r["url_hash"]] = r

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
out, n_body = [], 0
for row in rows:
    rec = best.get(uh((row.get("article_url") or "").strip()))
    body = (rec["text"] or "")[:CAP] if rec else ""
    if body:
        n_body += 1
    row2 = dict(row)
    row2["article_body"] = body
    row2["body_status"] = rec["status"] if rec else "no_fetch"
    row2["body_len"] = rec["text_len"] if rec else 0
    row2["relevant_label_body"] = ""
    row2["note_body"] = ""
    out.append(row2)

cols = list(rows[0].keys()) + ["article_body", "body_status", "body_len",
                               "relevant_label_body", "note_body"]
with open(DST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(out)
print(f"worksheet: {len(out)} rows | with body: {n_body} ({n_body/len(out):.0%})")
print(f"-> {DST}")
