#!/usr/bin/env python3
"""Phase 2 · Fetch SEC EDGAR filings (正文) for Chapter 11 bankruptcy events.

For each firm in validation/edgar_bankruptcy_events.json, fetch the primary
documents of 8-K / 10-Q / 10-K / NT-10-Q / NT-10-K filed in [onset-180d, onset),
strip HTML to plain text, and write data/edgar/{event_id}.jsonl.

Each output row mirrors the GDELT schema so extract_events.py can consume it:
  {id, date, source:"EDGAR", url, title, text}

Usage:
  python pipeline/fetch_edgar.py [--events a,b] [--window 180] [--sleep 0.3] [--max-docs 40]
"""
from __future__ import annotations
import argparse
import html as html_mod
import json
import re
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "edgar"
GT = ROOT / "validation" / "edgar_bankruptcy_events.json"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_HEADERS = {"User-Agent": "Research research@example.com"}

# 财务困境前兆相关 filing 类型 (忽略 Form 4 内部人交易、SC 13G 等噪声)
KEEP_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A",
              "NT 10-Q", "NT 10-K", "DEFA14A", "DEF 14A", "PRE 14A"}


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    return urllib.request.urlopen(req, context=_CTX, timeout=timeout).read()


def strip_html(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_filings(cik: str, onset: datetime, window: int, sleep_between: float,
                  max_docs: int) -> list[dict]:
    """返回 [onset-180d, onset) 内的 filing 正文列表 (按 filingDate 升序)."""
    cik_num = cik.lstrip("0")
    d = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    rec = d["filings"]["recent"]
    forms = rec["form"]
    dates = rec["filingDate"]
    accs = rec["accessionNumber"]
    docs = rec["primaryDocument"]

    start = onset - timedelta(days=window)
    rows = []
    for i in range(len(forms)):
        if forms[i] not in KEEP_FORMS:
            continue
        fd = datetime.strptime(dates[i], "%Y-%m-%d")
        if not (start <= fd < onset):
            continue
        rows.append((dates[i], forms[i], accs[i], docs[i], cik_num))

    # 按日期升序 (前兆最早在前)
    rows.sort(key=lambda r: r[0])
    if max_docs > 0:
        rows = rows[:max_docs]

    out = []
    for date_s, form, acc, doc, cik_n in rows:
        acc_no_dash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_n}/{acc_no_dash}/{doc}"
        try:
            raw = get(url).decode("utf-8", errors="ignore")
            text = strip_html(raw)
            # 截断超长正文 (10-K 可几十万字符, 抽取 prompt 撑不住)
            if len(text) > 8000:
                text = text[:8000]
            out.append({
                "date": date_s, "form": form, "accession": acc,
                "url": url, "text": text,
            })
        except Exception as e:
            print(f"    !! {date_s} {form} {doc} FAIL: {str(e)[:60]}", flush=True)
        time.sleep(sleep_between)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--max-docs", type=int, default=40)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(GT, encoding="utf-8"))
    events = gt["events"]
    only = {e.strip() for e in args.events.split(",") if e.strip()}

    for ev in events:
        eid = ev["event_id"]
        if only and eid not in only:
            continue
        onset = datetime.strptime(ev["gt_onset_date"], "%Y-%m-%d")
        print(f"\n=== {eid} ({ev['name']}, CIK {ev['cik']}) onset {ev['gt_onset_date']} ===", flush=True)
        filings = fetch_filings(ev["cik"], onset, args.window, args.sleep, args.max_docs)
        out_path = DATA / f"{eid}.jsonl"
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, fl in enumerate(filings):
                rec = {
                    "id": f"{eid}-{fl['date']}-{i:03d}",
                    "date": fl["date"],
                    "source": "EDGAR",
                    "url": fl["url"],
                    "title": f"{fl['form']} filing ({ev['name']})",
                    "text": fl["text"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"  -> {out_path}: {n} filings", flush=True)

    print("\ndone", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
