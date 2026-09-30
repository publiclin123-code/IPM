#!/usr/bin/env python3
"""Third-pass body fetch for still-failed URLs (Plan B).

  1. Wayback availability API retry (fast; pass-1 lookups sometimes raced
     timeouts while the direct connection was wedged)
  2. narrow-window CDX query (article date -90d .. +540d, exact match,
     limit 5, no collapse) when availability misses
  3. one more direct attempt with longer timeout
Appends to data/body_fetch/bodies.jsonl as before.
"""
import csv, hashlib, json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import trafilatura

ROOT = Path(__file__).resolve().parent.parent
BODIES = ROOT / "data" / "body_fetch" / "bodies.jsonl"
MANIFEST = ROOT / "data" / "body_fetch" / "manifest.csv"
PROXY = "http://127.0.0.1:10809"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MAX_BYTES = 2_000_000
MIN_TEXT = 250
MAX_TEXT_STORE = 30_000
AVAIL = "https://archive.org/wayback/available?url={u}&timestamp={ts}"
CDX = ("https://web.archive.org/cdx/search/cdx?url={u}&output=json&limit=5"
       "&filter=statuscode:200&from={f}&to={t}")

def uh(u): return hashlib.sha1(u.encode()).hexdigest()[:16]

def _client():
    return httpx.Client(proxy=PROXY, headers={"User-Agent": UA,
                   "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                   "Accept-Language": "en-US,en;q=0.9"},
                      timeout=30.0, follow_redirects=True, verify=False)

def get(url):
    try:
        with _client() as c:
            r = c.get(url)
        if r.status_code == 200 and "html" in r.headers.get("content-type", "html"):
            return r.text[:MAX_BYTES]
        return ""
    except Exception:
        return ""

def extract(html):
    if not html: return ""
    try:
        return trafilatura.extract(html, favor_recall=True,
                                   include_comments=False, include_tables=False) or ""
    except Exception:
        return ""

def _month_shift(yyyymmdd: str, months: int) -> str:
    y, m = int(yyyymmdd[:4]), int(yyyymmdd[4:6]) + months
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    return f"{y:04d}{m:02d}01"

def snapshots(url, date):
    d = (date or "20200101").replace("-", "")[:8] or "20200101"
    try:  # availability first
        with _client() as c:
            r = c.get(AVAIL.format(u=url, ts=d + "01"))
        if r.status_code == 200:
            snap = ((r.json() or {}).get("archived_snapshots") or {}).get("closest") or {}
            if snap.get("available") and snap.get("url"):
                u2 = snap["url"].replace("http://web.archive.org", "https://web.archive.org")
                return [u2]
    except Exception:
        pass
    try:  # narrow CDX
        with _client() as c:
            r = c.get(CDX.format(u=url, f=_month_shift(d, -3), t=_month_shift(d, 18)))
        if r.status_code == 200 and r.text.strip().startswith("["):
            rows = r.json()
            if len(rows) > 1:
                ts = sorted((row[2] for row in rows[1:]), key=lambda t: abs(int(t[:8]) - int(d)))
                return [f"https://web.archive.org/web/{t}id_/{url}" for t in ts[:2]]
    except Exception:
        pass
    return []

def fetch_one(row):
    u = row["url"]
    rec = {"event_id": row["event_id"], "article_id": row["article_id"], "url": u,
           "url_hash": uh(u), "status": "fail", "http_status": 0, "via": None,
           "text_len": 0, "text": None,
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t = extract(get(u))
    if len(t) >= MIN_TEXT:
        rec.update(status="ok_direct", via="direct", text_len=len(t), text=t[:MAX_TEXT_STORE])
        return rec, "direct"
    for snap in snapshots(u, row.get("date", "")):
        t = extract(get(snap))
        if len(t) >= MIN_TEXT:
            rec.update(status="ok_wayback", via=snap, text_len=len(t), text=t[:MAX_TEXT_STORE])
            return rec, "wayback"
    return rec, None

def main():
    best = {}
    for line in BODIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            old = best.get(r["url_hash"])
            if old is None or (r["status"].startswith("ok") and not old["status"].startswith("ok")) \
               or (r["status"].startswith("ok") and r["text_len"] > old["text_len"]):
                best[r["url_hash"]] = r
    todo = [r for r in best.values() if not r["status"].startswith("ok")]
    print(f"pass3 targets: {len(todo)}", flush=True)

    date_by_uh, ev_by_uh = {}, {}
    for m in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        h = uh(m["url"]); date_by_uh[h] = m["date"]
    for r in todo:
        r.setdefault("date", date_by_uh.get(r["url_hash"], ""))

    stats = {"direct": 0, "wayback": 0, "still_fail": 0}
    t0 = time.time(); n = 0
    out = open(BODIES, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for r in todo:
            time.sleep(0.15)
            futs[ex.submit(fetch_one, r)] = r
        for f in as_completed(futs):
            rec, how = f.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            stats[how or "still_fail"] += 1
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)} {stats} {time.time()-t0:.0f}s", flush=True)
    out.close()
    print("PASS3 DONE", stats, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
