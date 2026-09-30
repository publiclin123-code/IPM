"""
按 GT 事件批量下载 GDELT 窗口语料。
对每个事件:
  1. 取窗口 [onset - T, onset)，T 默认 180 天;
  2. GDELT 2.0 自 2013-02-18 起，更早事件跳过;
  3. 用 event_keywords.json 的 url_keywords 过滤（标题/URL 含关键词才保留）;
  4. 输出到 data/by_event/{event_id}.jsonl

零第三方依赖，E:\\anaconda\\python.exe 可直接跑。

用法:
  # 全部 GT 事件
  python fetch_per_event.py

  # 指定事件 + 窗口
  python fetch_per_event.py --events red_sea_crisis_2023,suez_ever_given_2021 --window 30

  # pilot: 红海危机 30 天窗口
  python fetch_per_event.py --events red_sea_crisis_2023 --window 30
"""
import argparse
import csv
import io
import json
import ssl
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "validation" / "gt_events.json"
KEYWORDS_PATH = Path(__file__).resolve().parent / "event_keywords.json"
OUT_DIR = ROOT / "data" / "by_event"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

GDELT_START = datetime(2013, 2, 18)  # GDELT 2.0 上线日
TIMESLOTS = ["000000", "150000", "300000", "450000"]
COL_DATE = 1
COL_URL = 60
MIN_COLS = 61


def slug_title(url: str) -> str:
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.split("?")[0].split("#")[0]
    slug = slug.replace("_", " ").replace("-", " ").replace(".html", "").replace(".htm", "")
    return slug[:200]


def fetch_zip_text(url: str, timeout: int = 60) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
            data = resp.read()
    except Exception as e:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = z.namelist()[0]
            return z.read(name).decode("utf-8", errors="replace")
    except Exception:
        return None


def parse_rows(text: str):
    """逐行解析 GDELT export.CSV，yield (sqldate, url) 对。"""
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = next(csv.reader([line], delimiter="\t", quotechar="\""))
        if len(cols) < MIN_COLS:
            continue
        yield cols[COL_DATE], cols[COL_URL]


def fetch_window(start: datetime, end: datetime, keywords: list[str], sleep_between: float = 0.1) -> dict:
    """
    遍历 [start, end) 的 GDELT 文件，按 url+slug 关键词过滤，返回 {date: [rec, ...]}。
    keywords 为空则不过滤（保留全部，慎用）。
    """
    out_by_date = {}
    day = start
    n_files_ok = n_files_fail = n_articles = 0
    while day < end:
        day_str = day.strftime("%Y%m%d")
        for slot in TIMESLOTS:
            stamp = day_str + slot
            url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
            text = fetch_zip_text(url)
            if text is None:
                n_files_fail += 1
                continue
            n_files_ok += 1
            for sqldate, art_url in parse_rows(text):
                if not art_url:
                    continue
                # 关键词过滤: URL 与 slug 都搜
                haystack = (art_url + " " + slug_title(art_url)).lower()
                if keywords and not any(kw.lower() in haystack for kw in keywords):
                    continue
                if not sqldate or len(sqldate) != 8:
                    continue
                iso_date = f"{sqldate[:4]}-{sqldate[4:6]}-{sqldate[6:8]}"
                out_by_date.setdefault(iso_date, []).append(art_url)
                n_articles += 1
            time.sleep(sleep_between)
        day += timedelta(days=1)
    return {
        "by_date": out_by_date,
        "n_files_ok": n_files_ok,
        "n_files_fail": n_files_fail,
        "n_articles": n_articles,
    }


def dedup_urls(items: list) -> list:
    """同一天同 URL 去重。"""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def write_event_jsonl(event_id: str, by_date: dict) -> int:
    out_path = OUT_DIR / f"{event_id}.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for iso_date, urls in sorted(by_date.items()):
            urls = dedup_urls(urls)
            for i, u in enumerate(urls):
                title = slug_title(u)
                rec = {
                    "id": f"{event_id}-{iso_date}-{i:04d}",
                    "date": iso_date,
                    "source": "GDELT",
                    "url": u,
                    "title": title,
                    "text": title,  # GDELT 仅标题（slug）；正文另抓
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="按 GT 事件批量下载 GDELT 窗口语料")
    ap.add_argument("--events", default="", help="逗号分隔的 event_id；默认全部")
    ap.add_argument("--window", type=int, default=180, help="窗口天数（onset 前 T 天），默认 180")
    ap.add_argument("--gt", default=str(GT_PATH))
    ap.add_argument("--keywords", default=str(KEYWORDS_PATH))
    ap.add_argument("--sleep", type=float, default=0.1, help="文件间隔秒数（避免被限流）")
    args = ap.parse_args()

    with open(args.gt, encoding="utf-8") as f:
        gt_data = json.load(f)
    events = gt_data.get("events", gt_data) if isinstance(gt_data, dict) else gt_data
    with open(args.keywords, encoding="utf-8") as f:
        kw_map = json.load(f)

    selected = set(args.events.split(",")) if args.events else None
    summary = []
    for ev in events:
        eid = ev["event_id"]
        if selected and eid not in selected:
            continue
        try:
            onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            print(f"[skip] {eid}: 无效 onset", file=sys.stderr)
            continue
        start = onset - timedelta(days=args.window)
        if onset < GDELT_START:
            print(f"[skip] {eid}: onset {onset.date()} 早于 GDELT 2.0 起点 ({GDELT_START.date()})",
                  file=sys.stderr)
            summary.append({"event_id": eid, "status": "skipped_pre_gdelt"})
            continue
        # 窗口起点也不能早于 GDELT 起点
        if start < GDELT_START:
            start = GDELT_START
            print(f"[warn] {eid}: 窗口起点截断到 GDELT 起点 {GDELT_START.date()}", file=sys.stderr)
        kw_entry = kw_map.get(eid, {})
        keywords = kw_entry.get("url_keywords", [])
        print(f"\n=== {eid} ({ev['event_name']}) ===", file=sys.stderr)
        print(f"  window: {start.date()} → {onset.date()} (T={args.window}d, {len(keywords)} keywords)",
              file=sys.stderr)
        res = fetch_window(start, onset, keywords, sleep_between=args.sleep)
        n = write_event_jsonl(eid, res["by_date"])
        print(f"  -> {OUT_DIR / (eid + '.jsonl')}: {n} articles "
              f"(files ok={res['n_files_ok']}, fail={res['n_files_fail']})", file=sys.stderr)
        summary.append({
            "event_id": eid, "status": "ok",
            "n_articles": n, "n_files_ok": res["n_files_ok"],
            "n_files_fail": res["n_files_fail"],
            "window_start": start.strftime("%Y-%m-%d"),
            "window_end": onset.strftime("%Y-%m-%d"),
            "keywords": keywords,
        })

    # 写汇总
    summary_path = OUT_DIR / "_download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"window_days": args.window, "events": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n[done] summary -> {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
