#!/usr/bin/env python3
"""Second-pass body fetch for failed/short records (Plan B).

Improvements over pass 1:
  - Wayback CDX API (statuscode:200 filter, closest to article date) instead of
    the availability API, which misses many snapshots
  - retries ok_wayback records whose extraction was short (< MIN_TEXT) once more
Appends new records to data/body_fetch/bodies.jsonl (same url_hash; the corpus
builder prefers ok records and longer bodies, so appends naturally win).
"""
import json, random, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import trafilatura

ROOT = Path(__file__).resolve().parent.parent
BODIES = ROOT / "data" / "body_fetch" / "bodies.jsonl"
PROXY = "http://127.0.0.1:10809"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MAX_BYTES = 2_000_000
MIN_TEXT = 250
MAX_TEXT_STORE = 30_000
CDX = ("https://web.archive.org/cdx/search/cdx?url={u}"
       "&output=json&limit=-3&filter=statuscode:200&collapse=digest"
       "&from={f}&to={t}")

client = None  # per-request clients: avoids wedged pooled connections

def _new_client():
    return httpx.Client(proxy=PROXY, headers={"User-Agent": UA,
                   "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                   "Accept-Language": "en-US,en;q=0.9"},
                      timeout=25.0, follow_redirects=True, verify=False)

def uh(u): return hashlib.sha1(u.encode()).hexdigest()[:16]

def extract(html):
    if not html: return ""
    try:
        return trafilatura.extract(html, favor_recall=True,
                                   include_comments=False, include_tables=False) or ""
    except Exception:
        return ""

def get(url):
    try:
        with _new_client() as c:
            r = c.get(url)
        if r.status_code == 200:
            ct = r.headers.get("content-type", "html")
            if "text/html" in ct or "html" in ct:
                return r.text[:MAX_BYTES]
        return ""
    except Exception:
        return ""

def cdx_snapshots(url, date):
    d = (date or "20200101").replace("-", "")[:8]
    lo, hi = "19960101", "20261231"
    try:
        with _new_client() as c:
            r = c.get(CDX.format(u=url, f=lo, t=hi), params=None)
        if r.status_code != 200 or not r.text.strip():
            return []
        rows = r.json()
        if not rows or len(rows) < 2:
            return []
        snaps = [row[2] for row in rows[1:]]           # timestamp col
        snaps.sort(key=lambda t: abs(int(t[:8]) - int(d)))
        return [f"https://web.archive.org/web/{t}id_/{url}" for t in snaps[:3]]
    except Exception:
        return []

def fetch_one(row):
    u = row["url"]
    rec = dict(row)
    rec.update(status="fail", via=None, text_len=0, text=None,
               fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    # direct retry (transient net errors often clear)
    html = get(u)
    t = extract(html)
    if len(t) >= MIN_TEXT:
        rec.update(status="ok_direct", via="direct", text_len=len(t), text=t[:MAX_TEXT_STORE])
        return rec, "direct"
    for snap in cdx_snapshots(u, row.get("date", "")):
        t = extract(get(snap))
        if len(t) >= MIN_TEXT:
            rec.update(status="ok_wayback", via=snap, text_len=len(t), text=t[:MAX_TEXT_STORE])
            return rec, "cdx"
    return rec, None

def main():
    recs = {}
    for line in BODIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            old = recs.get(r["url_hash"])
            if old is None or (r["status"].startswith("ok") and not old["status"].startswith("ok")) \
               or (r["status"].startswith("ok") and r["text_len"] > old["text_len"]):
                recs[r["url_hash"]] = r
    todo = [r for r in recs.values() if not r["status"].startswith("ok")]
    print(f"pass2 targets: {len(todo)}")

    # need date for CDX closest-match: recover from manifest
    import csv
    date_by_uh = {uh(m["url"]): m["date"] for m in
                  csv.DictReader(open(ROOT / "data/body_fetch/manifest.csv", encoding="utf-8"))}
    for r in todo:
        r.setdefault("date", date_by_uh.get(r["url_hash"], ""))

    stats = {"direct": 0, "cdx": 0, "still_fail": 0}
    t0 = time.time(); n = 0
    out = open(BODIES, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for r in todo:
            time.sleep(0.05)
            futs[ex.submit(fetch_one, r)] = r
        for f in as_completed(futs):
            rec, how = f.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            stats[how or "still_fail"] += 1
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)} {stats} {time.time()-t0:.0f}s", flush=True)
    out.close()
    print("PASS2 DONE", stats, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
