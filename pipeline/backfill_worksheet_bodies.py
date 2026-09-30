#!/usr/bin/env python3
"""Backfill retrieval worksheet body columns from the rebuilt body corpus.

data/annotation/retrieval_audit_sample_body.csv
- For every row, refresh article_body / body_status / body_len from
  data/by_event_body/{eid}.jsonl (clean body only; junk never written).
- Rows previously mechanically closed (ok_jina challenge pages,
  note_body starting 'challenge page') that now have a clean body are
  REOPENED: relevant_label_body / note_body blanked for re-annotation.
- Rows whose new status is title_fallback and were mechanically closed
  stay closed (labels kept), body columns updated honestly.
Backup written to *_body.csv.bak2 before overwrite.
"""
import csv, glob, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "annotation" / "retrieval_audit_sample_body.csv"
CORPUS = ROOT / "data" / "by_event_body"

corpus = {}  # (eid, article_id) -> row
for p in glob.glob(str(CORPUS / "*.jsonl")):
    if "_background" in p:
        continue
    eid = Path(p).stem
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        corpus[(eid, r["id"])] = r

shutil.copy(CSV, str(CSV) + ".bak2")
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
fields = rows[0].keys()

n_reopen = n_still_closed = n_text_upd = n_fb = 0
for r in rows:
    key = (r["event_id"], r["article_id"])
    c = corpus.get(key)
    if c is None:
        continue
    clean = c["body_status"] != "title_fallback"
    was_mech = r["body_status"] == "ok_jina" and r["note_body"].startswith("challenge page")
    if clean:
        if (c["text"] != r["article_body"]) or (c["body_status"] != r["body_status"]):
            r["article_body"] = c["text"]
            r["body_status"] = c["body_status"]
            r["body_len"] = str(c["body_len"])
            n_text_upd += 1
        if was_mech and r["relevant_label_body"].strip():
            r["relevant_label_body"] = ""
            r["note_body"] = ""
            n_reopen += 1
    else:  # new = title_fallback
        if was_mech:
            # still no clean text anywhere: keep closed, honest columns
            r["article_body"] = ""
            r["body_status"] = "title_fallback"
            r["body_len"] = "0"
            if not r["note_body"].endswith("no archive copy"):
                r["note_body"] = (r["note_body"].rstrip("; ") +
                                  "; re-cascaded 7 free archives: no archive copy")
            n_still_closed += 1
        elif r["body_status"] != "title_fallback":
            n_fb += 1  # unexpected downgrade of a real-body row; report only

with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"rows={len(rows)} text/status updated={n_text_upd} "
      f"reopened={n_reopen} still-closed={n_still_closed} downgraded={n_fb}")
