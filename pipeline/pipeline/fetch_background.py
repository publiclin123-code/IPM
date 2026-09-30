"""
采样 GDELT "中性背景池" 用作 BSCC 负对照。

设计动机
--------
原 BSCC 用单一黑天鹅事件（Suez Ever Given 2021）作负对照，但该事件 onset 前
窗口内 LLM 关键词捞到 0 篇文章（黑天鹅定义上无前兆），无法估计本底捏造率 p₀。
改为：从 GDELT 采样一批新闻，让 LLM 在这些"本应无 GT 事件信号"的输入上抽取，
输出的非零信号即模型本底捏造。

三种采样模式
------------
--mode neutral: 与所有 GT 事件关键词互斥的随机新闻（如 Tunguska / 政治）。
  问题：明显无关的输入 LLM 正确拒抽 → p₀≈0 → BSCC 退化。
--mode hard-negative: 双向过滤——正向命中供应链议题通用词，负向不命中事件专属词。
  问题：议题相邻新闻里的真中断（Shell 油管、OPEC 减产）被 LLM 正确抽取 → p₀≈0.85
  → BSCC 过度校正（惩罚了真信号）。
--mode post-event (推荐): 采样每个 GT 事件 **onset 之后**的新闻（用该事件的专属词）。
  这些文章是事后报道（confirmation），LLM 应标 temporality=confirmation；
  若标为 forward_looking 即时间虚构。p₀ = 这些时间不可能信号的置信度均值。
  对能力足够的 LLM（30B+），时间虚构是真实存在的错误类型（事后信息污染前兆判断），
  而主题虚构不存在。这是 BSCC 的正确探针。

复现性：默认 random.seed(42)，跨次运行结果一致。

零第三方依赖，E:\\anaconda\\python.exe 可直接跑。

用法:
  # 默认 200 篇
  python fetch_background.py

  # 小规模测试 50 篇
  python fetch_background.py --n 50

  # hard-negative 模式
  python fetch_background.py --mode hard-negative --n 50

  # post-event 模式（推荐用于 BSCC，测时间虚构率）
  python fetch_background.py --mode post-event --n 50

  # 自定义日期范围（仅 neutral/hard-negative）
  python fetch_background.py --start 2018-01-01 --end 2023-12-31 --n 100
"""

# 议题通用词：供应链风险相关但非特定事件。hard-negative 模式的正向过滤词表。
# 来源：综合各 GT 事件的 commodities/event_type 提炼，去掉事件专属标识符。
TOPIC_KEYWORDS = [
    # 物流/航运
    "supply chain", "shipping", "cargo", "container", "freight", "logistics",
    "port", "vessel", "tanker", "bulk carrier",
    # 半导体/电子
    "chip", "semiconductor", "foundry", "wafer", "tsmc", "samsung",
    "intel", "nvidia", "qualcomm", "broadcom",
    # 制造/工厂
    "factory", "manufacturing", "plant", "production line", "industrial",
    "automotive", "steel", "chemical",
    # 商品/大宗
    "commodity", "oil", "crude", "lng", "lithium", "cobalt", "rare earth",
    "copper", "aluminum", "wheat", "soybean",
    # 中断通用词
    "disruption", "shortage", "outage", "halt", "shutdown", "delay",
    "backlog", "congestion", "bottleneck", "force majeure", "strike",
]
import argparse
import csv
import io
import json
import random
import ssl
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYWORDS_PATH = Path(__file__).resolve().parent / "event_keywords.json"
GT_PATH = ROOT / "validation" / "gt_events.json"
OUT_DIR = ROOT / "data" / "by_event"
OUT_PATH = OUT_DIR / "_background.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

GDELT_START = datetime(2013, 2, 18)
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


def load_exclude_keywords() -> list[str]:
    """汇总所有事件的 url_keywords，去重小写。空列表保留（不影响过滤）。"""
    with open(KEYWORDS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    kws = set()
    for ev, cfg in data.items():
        if ev.startswith("_"):
            continue
        if isinstance(cfg, dict):
            for kw in cfg.get("url_keywords", []) or []:
                if kw:
                    kws.add(kw.lower())
    return sorted(kws)


def load_topic_keywords() -> list[str]:
    """供应链议题通用词（hard-negative 模式正向过滤）。"""
    return [k.lower() for k in TOPIC_KEYWORDS]


def fetch_zip_text(url: str, timeout: int = 60) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
            data = resp.read()
    except Exception:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    except Exception:
        return None


def parse_rows(text: str):
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = next(csv.reader([line], delimiter="\t", quotechar="\""))
        if len(cols) < MIN_COLS:
            continue
        yield cols[COL_DATE], cols[COL_URL]


def sample_slots(start: datetime, end: datetime, n_target: int, rng: random.Random,
                 hit_rate: int = 100) -> list[tuple[str, str]]:
    """
    构造一个 (day_str, slot) 抽样队列。
    hit_rate: 每个槽位预估贡献篇数（neutral≈100，hard-negative≈15）。
    """
    total_days = max(1, (end - start).days)
    # 每 hit_rate 篇目标抽 1 个槽位，最少 5 个，最多 80 个（防止跑太久）
    n_slots = max(5, min(80, (n_target + hit_rate - 1) // hit_rate + 2))
    picked_days = [start + timedelta(days=rng.randrange(total_days)) for _ in range(n_slots)]
    slots = []
    for d in picked_days:
        slot = rng.choice(TIMESLOTS)
        slots.append((d.strftime("%Y%m%d"), slot))
    return slots


def run_post_event(args) -> int:
    """post-event 模式: 遍历每个 GT 事件的 [onset, onset+post_days] 窗口，
    用该事件的 url_keywords 抓事件相关新闻（事后报道）。"""
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    events = gt_data.get("events", gt_data) if isinstance(gt_data, dict) else gt_data
    with open(KEYWORDS_PATH, encoding="utf-8") as f:
        kw_map = json.load(f)

    # 按事件均分目标篇数
    valid_events = []
    for ev in events:
        eid = ev["event_id"]
        cfg = kw_map.get(eid, {})
        kws = cfg.get("url_keywords", []) if isinstance(cfg, dict) else []
        if not kws:
            print(f"[skip] {eid}: 无关键词", file=sys.stderr)
            continue
        try:
            onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        if onset < GDELT_START:
            print(f"[skip] {eid}: onset {onset.date()} 早于 GDELT 2.0", file=sys.stderr)
            continue
        valid_events.append((eid, onset, [k.lower() for k in kws]))

    if not valid_events:
        print("[error] 无可跑事件", file=sys.stderr)
        return 2

    per_event = max(5, args.n // len(valid_events))
    print(f"[info] mode=post-event, {len(valid_events)} 个事件, 每事件目标 {per_event} 篇 "
          f"(post_days={args.post_days})", file=sys.stderr)

    if args.dry_run:
        for eid, onset, kws in valid_events:
            print(f"  {eid}: [{onset.date()}, {onset.date()+timedelta(days=args.post_days)}] "
                  f"{len(kws)} 关键词")
        return 0

    collected = []
    seen_urls = set()
    t0 = time.time()
    n_files_ok = n_files_fail = 0

    for eid, onset, kws in valid_events:
        if len(collected) >= args.n:
            break
        end = onset + timedelta(days=args.post_days)
        day = onset
        ev_count = 0
        while day < end and ev_count < per_event and len(collected) < args.n:
            day_str = day.strftime("%Y%m%d")
            for slot in TIMESLOTS:
                if ev_count >= per_event or len(collected) >= args.n:
                    break
                stamp = day_str + slot
                url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
                text = fetch_zip_text(url)
                if text is None:
                    n_files_fail += 1
                    continue
                n_files_ok += 1
                iso = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
                for sqldate, art_url in parse_rows(text):
                    if not art_url or art_url in seen_urls:
                        continue
                    haystack = (art_url + " " + slug_title(art_url)).lower()
                    if not any(kw in haystack for kw in kws):
                        continue
                    seen_urls.add(art_url)
                    collected.append({
                        "id": f"_background-{iso}-{len(collected):04d}",
                        "date": iso,
                        "source": "GDELT",
                        "url": art_url,
                        "title": slug_title(art_url),
                        "text": slug_title(art_url),
                    })
                    ev_count += 1
                    if ev_count >= per_event or len(collected) >= args.n:
                        break
                time.sleep(args.sleep)
            day += timedelta(days=1)
        print(f"  {eid}: +{ev_count} 篇 (累计 {len(collected)})", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in collected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\n[done] {len(collected)} 篇 → {OUT_PATH.relative_to(ROOT)}", file=sys.stderr)
    print(f"       耗时 {dt:.1f}s | 文件 ok/fail: {n_files_ok}/{n_files_fail}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GDELT 中性背景池采样 (BSCC 负对照)")
    ap.add_argument("--mode", choices=["neutral", "hard-negative", "post-event"], default="post-event",
                    help="采样模式: neutral=随机无关新闻; hard-negative=议题相邻非事件; "
                         "post-event(默认)=GT事件onset后新闻，测时间虚构率")
    ap.add_argument("--n", type=int, default=200, help="目标篇数（去重后）")
    ap.add_argument("--start", default="2016-01-01", help="采样起始日期 (neutral/hard-neg)")
    ap.add_argument("--end", default="2023-12-31", help="采样结束日期 (neutral/hard-neg)")
    ap.add_argument("--post-days", type=int, default=30, help="post-event 模式: onset 后采样天数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--sleep", type=float, default=0.1, help="每个 GDELT 文件下载后 sleep 秒")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不下载")
    args = ap.parse_args()

    # post-event 模式: 走单独分支（按事件循环，用事件专属词匹配 onset 后窗口）
    if args.mode == "post-event":
        return run_post_event(args)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if start < GDELT_START:
        start = GDELT_START
        print(f"[warn] 起始日期早于 GDELT 2.0 上线，调整为 {start.date()}", file=sys.stderr)
    if end <= start:
        print(f"[error] 结束日期须晚于起始日期", file=sys.stderr)
        return 2

    exclude_kws = load_exclude_keywords()
    topic_kws = load_topic_keywords()
    print(f"[info] mode={args.mode}", file=sys.stderr)
    print(f"[info] 排除关键词 {len(exclude_kws)} 个: {exclude_kws[:8]}{'...' if len(exclude_kws) > 8 else ''}",
          file=sys.stderr)
    if args.mode == "hard-negative":
        print(f"[info] 议题通用词 {len(topic_kws)} 个 (必须命中其一): {topic_kws[:8]}...",
              file=sys.stderr)

    # hard-negative 模式命中率较低（双重过滤，每槽位约 15 篇），neutral 约 100 篇
    rng = random.Random(args.seed)
    hit_rate = 15 if args.mode == "hard-negative" else 100
    slots = sample_slots(start, end, args.n, rng, hit_rate=hit_rate)
    print(f"[info] 抽样计划: {len(slots)} 个 GDELT 槽位, 目标 {args.n} 篇 "
          f"(预估每槽 {hit_rate} 篇)", file=sys.stderr)

    if args.dry_run:
        for i, (d, s) in enumerate(slots, 1):
            print(f"  {i:2d}. {d} {s}")
        return 0

    collected = []  # [{date,url,title}]
    seen_urls = set()
    n_files_ok = n_files_fail = n_skipped_kw = 0
    t0 = time.time()

    for day_str, slot in slots:
        if len(collected) >= args.n:
            break
        stamp = day_str + slot
        url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
        text = fetch_zip_text(url)
        if text is None:
            n_files_fail += 1
            print(f"  [fail] {stamp}", file=sys.stderr)
            continue
        n_files_ok += 1
        iso_date = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        n_in_file = 0
        for sqldate, art_url in parse_rows(text):
            if not art_url or art_url in seen_urls:
                continue
            haystack = (art_url + " " + slug_title(art_url)).lower()
            if any(kw in haystack for kw in exclude_kws):
                n_skipped_kw += 1
                continue
            # hard-negative: 还须正向命中议题通用词
            if args.mode == "hard-negative" and not any(kw in haystack for kw in topic_kws):
                n_skipped_kw += 1
                continue
            seen_urls.add(art_url)
            collected.append({
                "id": f"_background-{iso_date}-{len(collected):04d}",
                "date": iso_date,
                "source": "GDELT",
                "url": art_url,
                "title": slug_title(art_url),
                "text": slug_title(art_url),
            })
            n_in_file += 1
            if len(collected) >= args.n:
                break
        print(f"  [ok]   {stamp}: +{n_in_file} 篇 (累计 {len(collected)})", file=sys.stderr)
        time.sleep(args.sleep)

    # 写出
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in collected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\n[done] {len(collected)} 篇 → {OUT_PATH.relative_to(ROOT)}", file=sys.stderr)
    print(f"       耗时 {dt:.1f}s | 文件 ok/fail: {n_files_ok}/{n_files_fail} | "
          f"关键词过滤掉 {n_skipped_kw} 篇", file=sys.stderr)
    if collected:
        dates = sorted({r["date"] for r in collected})
        print(f"       日期跨度: {dates[0]} ~ {dates[-1]} ({len(dates)} 个不同日期)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
