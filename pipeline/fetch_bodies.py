#!/usr/bin/env python3
"""Fetch article bodies for the GT-event pool (Plan B).

Reads   data/body_fetch/manifest.csv   (event_id, article_id, date, source, url, title)
Writes  data/body_fetch/bodies.jsonl   (append/resume safe, keyed by url_hash)

Strategy per URL:
  1. direct fetch via local proxy (browser UA, 20s timeout, 2MB cap)
  2. if fail / text too short -> Wayback Machine availability API -> snapshot (id_ raw)
Text extraction: trafilatura (favor_recall). status in {ok_direct, ok_wayback, fail}.

Usage:
  python pipeline/fetch_bodies.py --limit 20            # pilot
  python pipeline/fetch_bodies.py --workers 12          # full run (resume auto)
"""
import argparse, csv, hashlib, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import trafilatura

PROXY = "http://127.0.0.1:10809"
OUT = Path("data/body_fetch/bodies.jsonl")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_BYTES = 2_000_000
MIN_TEXT = 250          # below this a "successful" fetch is treated as no-body
MAX_TEXT_STORE = 30_000 # cap stored body length (chars)
WAYBACK_API = "https://archive.org/wayback/available?url={u}&timestamp={ts}"

client = None  # per-request clients: shared pool wedged after ~300 reqs

def _new_client():
    return httpx.Client(proxy=PROXY, headers=HEADERS, timeout=25.0,
                        follow_redirects=True, verify=False)

def url_hash(u: str) -> str:
    return hashlib.sha1(u.encode()).hexdigest()[:16]

def extract(html: str) -> str:
    if not html:
        return ""
    try:
        return trafilatura.extract(html, favor_recall=True,
                                   include_comments=False,
                                   include_tables=False) or ""
    except Exception:
        return ""

def fetch_direct(url: str):
    try:
        with _new_client() as c:
            r = c.get(url)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", "html"):
            html = r.text[:MAX_BYTES]
            return r.status_code, html
        return r.status_code, ""
    except Exception:
        return 0, ""

def wayback_lookup(url: str, ts: str):
    try:
        with _new_client() as c:
            r = c.get(WAYBACK_API.format(u=url, ts=ts.replace("-", "") + "01"))
        if r.status_code != 200:
            return None
        snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            return snap["url"].replace("http://web.archive.org", "https://web.archive.org")
    except Exception:
        pass
    return None

def fetch_one(row):
    u = row["url"]
    rec = {"event_id": row["event_id"], "article_id": row["article_id"],
           "url": u, "url_hash": url_hash(u), "status": "fail",
           "http_status": 0, "via": None, "text_len": 0, "text": None,
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # 1) direct
    code, html = fetch_direct(u)
    text = extract(html) if html else ""
    if len(text) >= MIN_TEXT:
        rec.update(status="ok_direct", http_status=code, via="direct",
                   text_len=len(text), text=text[:MAX_TEXT_STORE])
        return rec
    # 2) wayback
    snap = wayback_lookup(u, row.get("date") or "20200101")
    if snap:
        code2, html2 = fetch_direct(snap)
        text2 = extract(html2) if html2 else ""
        if len(text2) >= MIN_TEXT:
            rec.update(status="ok_wayback", http_status=code2, via=snap,
                       text_len=len(text2), text=text2[:MAX_TEXT_STORE])
            return rec
    rec["http_status"] = code or rec["http_status"]
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open("data/body_fetch/manifest.csv", encoding="utf-8")))
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: done.add(json.loads(line)["url_hash"])
                except Exception: pass
        print(f"resume: {len(done)} already fetched")
    todo = [r for r in rows if url_hash(r["url"]) not in done]
    if a.start: todo = todo[a.start:]
    if a.limit: todo = todo[:a.limit]
    print(f"to fetch: {len(todo)}  (workers={a.workers})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats = {"ok_direct": 0, "ok_wayback": 0, "fail": 0}
    t0, n = time.time(), 0
    lock_write = open(OUT, "a", encoding="utf-8")

    def run(row):
        time.sleep(random.uniform(0.1, 0.4))  # politeness jitter
        return fetch_one(row)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run, r): r for r in todo}
        for f in as_completed(futs):
            rec = f.result()
            lock_write.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lock_write.flush()
            stats[rec["status"]] += 1
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print(f"{n}/{len(todo)}  {stats}  {el:.0f}s ({n/el:.1f}/s)", flush=True)
    lock_write.close()
    print("DONE", stats, f"{time.time()-t0:.0f}s total")

if __name__ == "__main__":
    main()
