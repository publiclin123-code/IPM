"""
GDELT 新闻语料抓取器 (Paper A RQ1)
从 GDELT 2.0 事件数据库批量下载指定日期范围的新闻，输出 JSONL。

GDELT 是免费开放的全球事件数据库，每天 15 分钟更新一次，含全球新闻标题/摘要。
URL 模式: https://data.gdeltproject.org/gdeltv2/{YYYYMMDD}{HHMM00}.export.CSV.zip
（{HHMM00} 用 000000/150000/300000/450000 四个时段之一；GDELT 新版命名带 .export）

零第三方依赖（标准库 urllib + zipfile + csv），E:\\anaconda\\python.exe 可直接运行。

用法:
  python fetch_gdelt.py --start 2023-11-01 --end 2023-12-31 --out news_redsea.jsonl
  python fetch_gdelt.py --start 2023-11-01 --end 2023-12-31 --out news.jsonl --query "Red Sea"

说明:
- 只抓取标题字段（SOURCEURL 里的标题）与 URL，不抓正文（正文需另行解析）。
- --query 可选：匹配标题含关键词的事件（提高信噪比）。
- 每天 4 个时段文件，约 4 次下载/天；30 天 ≈ 120 次请求。
- 输出行: {"id","date","source","url","title","text"} 与 extract_events.py 兼容。
"""
import argparse
import csv
import io
import json
import ssl
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta

# 部分网络环境证书校验不稳定（hostname mismatch），用宽松验证兜底
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

# GDELT 2.0 export 事件表列索引（实测 61 列，2026-08-12 验证）
# 0: GLOBALEVENTID, 1: SQLDATE(YYYYMMDD), 57: SOURCEURL 实际在 60
COL_DATE = 1        # SQLDATE: YYYYMMDD
COL_SOURCE = 2      # MonthYear (YYYYMM)
COL_URL = 60        # SOURCEURL（实测位置）
MIN_COLS = 61
# 注意: GDELT events 表无标题列。标题从 URL slug 提取（URL 末段），
# 或用 mentions/documents 表补充。extract_events.py 的 text 字段需要正文，
# 本脚本先给 slug 标题，正文抓取另做（见 README）。

TIMESLOTS = ["000000", "150000", "300000", "450000"]


def fetch_day(day: datetime, timeout: int = 120) -> list[dict]:
    """下载某一天的所有时段文件，返回事件行列表。"""
    rows = []
    for slot in TIMESLOTS:
        stamp = day.strftime("%Y%m%d") + slot
        url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
                data = resp.read()
        except Exception as e:
            print(f"  [skip] {stamp}: {e}", file=sys.stderr)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = z.namelist()[0]
                text = z.read(name).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [skip] {stamp}: unzip failed: {e}", file=sys.stderr)
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            cols = next(csv.reader([line], delimiter="\t", quotechar="\""))
            if len(cols) < MIN_COLS:
                continue
            rows.append(cols)
    return rows


def slug_title(url: str) -> str:
    """从 URL 末尾提取 slug 作为标题（GDELT events 表无标题字段）。"""
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.split("?")[0].split("#")[0]
    slug = slug.replace("_", " ").replace("-", " ").replace(".html", "").replace(".htm", "")
    return slug[:200]


def main() -> int:
    ap = argparse.ArgumentParser(description="GDELT 新闻抓取器")
    ap.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD（含）")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--query", default="", help="关键词过滤（匹配标题/事件）")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    n_total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        day = start
        while day <= end:
            print(f"fetching {day.date()}", file=sys.stderr)
            rows = fetch_day(day)
            for i, cols in enumerate(rows):
                date = cols[COL_DATE]
                url = cols[COL_URL]
                src = cols[COL_SOURCE]
                if args.query and args.query.lower() not in url.lower():
                    continue
                title = slug_title(url)
                rec = {
                    "id": f"{day.strftime('%Y%m%d')}-{i:05d}",
                    "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                    "source": "GDELT",
                    "url": url,
                    "title": title,
                    "text": title,  # 占位：正文需后续抓取
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
            day += timedelta(days=1)

    print(f"done: {n_total} articles -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
