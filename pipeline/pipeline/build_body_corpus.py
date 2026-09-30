#!/usr/bin/env python3
"""Merge fetched bodies into a body corpus (Plan B, todo 6).

Reads  data/body_fetch/bodies.jsonl   (fetch results, url_hash keyed)
       data/by_event/{eid}.jsonl      (original title-only corpus, 18 GT events)
Writes data/by_event_body/{eid}.jsonl (same rows; text = body if fetched,
       else falls back to title; adds body_status + body_len fields)

Article ids and ordering are preserved 1:1 so signals join by input_id.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "by_event"
DST = ROOT / "data" / "by_event_body"
FETCH = ROOT / "data" / "body_fetch" / "bodies.jsonl"
GT = json.load(open(ROOT / "validation" / "gt_events.json", encoding="utf-8"))
EVENTS = [e["event_id"] for e in GT["events"]]

import hashlib
import re

# 挑战页/反爬签名: 命中者不得作为正文 (统一验收规则, 与 pass5/背景池一致)
JUNK_RE = re.compile(
    r"cloudflare|verify you are human|just a moment\.|enable javascript and cookies"
    r"|attention required|checking your browser|challenge-platform|are you a robot"
    r"|captcha|perimeterx|datadome|incapsula|sucuri website firewall|ddos protection by"
    r"|wayback machine (has not|doesn'?t have|does not have)", re.I)

# url_hash -> best record; 合并规则: 干净正文优先于挑战页, 同类内取更长
bodies = {}
for line in FETCH.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if not r["status"].startswith("ok"):
        continue
    t = r.get("text") or ""
    r_junk = bool(JUNK_RE.search(t))
    old = bodies.get(r["url_hash"])
    if old is None:
        bodies[r["url_hash"]] = r
        continue
    old_junk = bool(JUNK_RE.search(old.get("text") or ""))
    if r_junk and not old_junk:
        continue                       # 干净的已存在, 挑战页不让进
    if not r_junk and old_junk:
        bodies[r["url_hash"]] = r      # 干净替换挑战页
    elif r["text_len"] > old["text_len"]:
        bodies[r["url_hash"]] = r      # 同类内取更长


def uh(u: str) -> str:
    return hashlib.sha1(u.encode()).hexdigest()[:16]

DST.mkdir(parents=True, exist_ok=True)
tot = ok = fb = 0
for eid in EVENTS:
    src_p = SRC / f"{eid}.jsonl"
    if not src_p.exists():
        continue
    out_rows, n_ok, n_fb = [], 0, 0
    for line in src_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        rec = bodies.get(uh((a.get("url") or "").strip()))
        # 挑战页不算正文 (即使它是该 URL 唯一记录) -> title 回退
        if rec and rec.get("text") and not JUNK_RE.search(rec["text"]):
            a["text"] = rec["text"]
            a["body_status"] = rec["status"]
            a["body_len"] = rec["text_len"]
            n_ok += 1
        else:
            a["text"] = a.get("title", "")  # fallback: title-only
            a["body_status"] = "title_fallback"
            a["body_len"] = 0
            n_fb += 1
        out_rows.append(a)
    with open(DST / f"{eid}.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{eid:35s} n={len(out_rows):5d} body={n_ok:5d} fallback={n_fb:5d} "
          f"({n_ok/max(len(out_rows),1):.0%})")
    tot += len(out_rows); ok += n_ok; fb += n_fb

print(f"\nTOTAL n={tot} body_ok={ok} title_fallback={fb} ({ok/max(tot,1):.1%})")
