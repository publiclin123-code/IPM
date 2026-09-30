#!/usr/bin/env python3
"""Pass 4: 补救 bodies.jsonl 里仍为 fail 的 URL (Plan B 加强版).

Strategy A: direct retry — 完整浏览器指纹 (sec-ch-ua / sec-fetch / Referer=google)
Strategy B: r.jina.ai reader — 服务端渲染, 抓 msn.com 等 JS 站 (限速 ~20 RPM, 429 退避)

只追加 ok 行到 bodies.jsonl (build_body_corpus 自动偏好更长正文, 安全).
Usage:
  python pipeline/fetch_bodies_pass4.py --limit 8    # pilot
  python pipeline/fetch_bodies_pass4.py              # full (先直连并发, 后 jina 串行)
"""
import argparse, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROXY = "http://127.0.0.1:10809"
OUT = Path("data/body_fetch/bodies.jsonl")
LOG = Path("data/body_fetch/pass4.log")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
H_BROWSER = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "cross-site",
    "sec-fetch-user": "?1",
}
H_JINA = {"User-Agent": UA}

MIN_TEXT = 250
MAX_TEXT_STORE = 30_000
JINA_MIN_GAP = 3.5          # >=20 RPM


def url_hash(u: str) -> str:
    return hashlib.sha1(u.encode()).hexdigest()[:16]


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    """返回 (targets, meta): targets = 仍无任何 ok 行的 url_hash 列表"""
    ok_hashes, latest, meta = set(), {}, {}
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        h = r["url_hash"]
        meta.setdefault(h, {"url": r["url"], "event_id": r.get("event_id"),
                            "article_id": r.get("article_id")})
        if r["status"].startswith("ok"):
            ok_hashes.add(h)
        latest[h] = r["status"]
    targets = [h for h, st in latest.items()
               if st == "fail" and h not in ok_hashes]
    return targets, meta


def clean_jina(text: str) -> str:
    """jina 输出带 Title/URL Source 行 + markdown 记号, 剥掉."""
    lines = text.splitlines()
    out, skip_header = [], True
    for ln in lines:
        if skip_header and (ln.startswith("Title:") or ln.startswith("URL Source:")
                            or ln.startswith("Published Time:") or not ln.strip()):
            continue
        skip_header = False
        out.append(ln)
    t = "\n".join(out)
    for ch in "*#[]()":       # 去轻量 markdown 记号, 对齐 trafilatura 纯文本
        t = t.replace(ch, "")
    return t.strip()


def fetch_direct(url: str):
    try:
        with httpx.Client(proxy=PROXY, headers=H_BROWSER, timeout=30.0,
                          follow_redirects=True, verify=False) as c:
            r = c.get(url)
        if r.status_code == 200:
            ct = r.headers.get("content-type", "html")
            if "html" in ct or "text" in ct:
                return r.status_code, r.text[:2_000_000]
        return r.status_code, ""
    except Exception:
        return 0, ""


def trafil(html: str) -> str:
    try:
        import trafilatura
        return trafilatura.extract(html, favor_recall=True,
                                   include_comments=False,
                                   include_tables=False) or ""
    except Exception:
        return ""


def rec(h: str, status: str, via: str, code: int, text: str, meta: dict) -> dict:
    text = (text or "")[:MAX_TEXT_STORE]
    return {"event_id": meta.get("event_id"), "article_id": meta.get("article_id"),
            "url": meta["url"], "url_hash": h, "status": status,
            "http_status": code, "via": via, "text_len": len(text),
            "text": text if text else None,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def append(r: dict):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    targets, meta = load_state()
    if args.limit:
        targets = targets[: args.limit]
    log(f"pass4 targets: {len(targets)} fail-URLs (no ok row yet)")

    # ---- Phase A: direct retry (并发) ----
    won = []
    def direct_one(h):
        url = meta[h]["url"]
        code, html = fetch_direct(url)
        text = trafil(html) if html else ""
        if len(text) >= MIN_TEXT:
            return h, rec(h, "ok_retry", "direct-retry", code, text, meta[h])
        return h, None

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(direct_one, h): h for h in targets}
        for i, fu in enumerate(as_completed(futs), 1):
            h, r = fu.result()
            if r:
                append(r); won.append(h)
            if i % 50 == 0:
                log(f"  direct {i}/{len(targets)} recovered={len(won)}")
    log(f"Phase A done: recovered {len(won)} / {len(targets)} via direct-retry")

    rest = [h for h in targets if h not in set(won)]

    # ---- Phase B: jina reader (串行限速) ----
    jwon = 0
    last = 0.0
    for i, h in enumerate(rest, 1):
        url = meta[h]["url"]
        gap = time.time() - last
        if gap < JINA_MIN_GAP:
            time.sleep(JINA_MIN_GAP - gap)
        try:
            with httpx.Client(proxy=PROXY, headers=H_JINA, timeout=60.0,
                              follow_redirects=True, verify=False) as c:
                r = c.get("https://r.jina.ai/" + url)
            if r.status_code == 429:
                log("  jina 429 -> sleep 60s")
                time.sleep(60)
                r = httpx.Client(proxy=PROXY, headers=H_JINA, timeout=60.0,
                                 follow_redirects=True,
                                 verify=False).get("https://r.jina.ai/" + url)
            last = time.time()
            # jina 免费档常返回 403 但仍带完整正文; 以清洗后长度为准
            text = clean_jina(r.text)
            if len(text) >= MIN_TEXT:
                append(rec(h, "ok_jina", "jina", r.status_code, text, meta[h]))
                jwon += 1
        except Exception as e:
            last = time.time()
        if i % 20 == 0:
            log(f"  jina {i}/{len(rest)} recovered={jwon}")
    log(f"Phase B done: recovered {jwon} / {len(rest)} via jina")
    log(f"PASS4 TOTAL: +{len(won)+jwon} bodies "
        f"(direct {len(won)}, jina {jwon}) of {len(targets)} targets")


if __name__ == "__main__":
    main()
