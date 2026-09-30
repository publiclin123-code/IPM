#!/usr/bin/env python3
"""背景池 (post-event, 340 条) 正文抓取 — 与主池同一套验收规则.

规则 (全文统一): direct -> wayback -> jina; 每一级正文都必须
  (a) 长度 >= 250; (b) 不含反爬/挑战页签名. 挑战页一律拒收.

输出: data/body_fetch/background_bodies.jsonl         (抓取记录, url_hash 键)
      data/by_event_body/_background.jsonl            (行保序, text=正文或标题回退)
Usage:
  python pipeline/fetch_background_bodies.py --limit 8   # pilot
  python pipeline/fetch_background_bodies.py             # full (可续跑)
"""
import argparse
import datetime
import hashlib
import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "by_event" / "_background.jsonl"
OUT = ROOT / "data" / "body_fetch" / "background_bodies.jsonl"
CORPUS = ROOT / "data" / "by_event_body" / "_background.jsonl"
LOG = ROOT / "data" / "body_fetch" / "background.log"

PROXY = "http://127.0.0.1:10809"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
H_BROWSER = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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

MIN_TEXT = 250
JUNK_RE = re.compile(
    r"cloudflare|verify you are human|just a moment\.|enable javascript and cookies"
    r"|attention required|checking your browser|challenge-platform|are you a robot"
    r"|captcha|perimeterx|datadome|incapsula|sucuri website firewall|ddos protection by"
    r"|wayback machine (has not|doesn'?t have|does not have)", re.I)


def log(msg: str) -> None:
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def trafil(html: str) -> str:
    try:
        import trafilatura
        return (trafilatura.extract(html) or "").strip()
    except Exception:
        return ""


def clean_jina(text: str) -> str:
    """r.jina.ai markdown -> 纯文本 (复用 pass4 逻辑)."""
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # 图片
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)           # 链接
    t = re.sub(r"^#+\s*", "", t, flags=re.M)                 # 标题符
    t = re.sub(r"[-*]{3,}", " ", t)                          # 分隔线
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def acceptable(text: str) -> bool:
    return len(text) >= MIN_TEXT and not JUNK_RE.search(text)


def fetch_direct(url: str) -> tuple[str, str]:
    with httpx.Client(proxy=PROXY, headers=H_BROWSER, timeout=45.0,
                      follow_redirects=True, trust_env=False) as c:
        r = c.get(url)
        if r.status_code != 200:
            return "", f"http{r.status_code}"
        return trafil(r.text), "direct"


def fetch_wayback(url: str) -> tuple[str, str]:
    with httpx.Client(proxy=PROXY, headers={"User-Agent": UA},
                      timeout=60.0, follow_redirects=True, trust_env=False) as c:
        api = "https://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe="")
        r = c.get(api, timeout=20.0)
        snap = ((r.json().get("archived_snapshots") or {}).get("closest") or {})
        if not snap.get("available") or snap.get("status") not in ("200", None):
            return "", "no_snapshot"
        ts = snap["timestamp"]
        r2 = c.get(f"https://web.archive.org/web/{ts}id_/{url}")
        if r2.status_code != 200:
            return "", f"wb{r2.status_code}"
        return trafil(r2.text), "wayback"


def fetch_jina(url: str) -> tuple[str, str]:
    with httpx.Client(proxy=PROXY, headers={"User-Agent": UA},
                      timeout=60.0, follow_redirects=True, trust_env=False) as c:
        r = c.get("https://r.jina.ai/" + url)
        if r.status_code != 200:
            return "", f"jina{r.status_code}"
        return clean_jina(r.text), "jina"


def fetch_one(url: str) -> dict:
    uh = hashlib.sha1(url.encode()).hexdigest()[:16]
    base = {"url": url, "url_hash": uh,
            "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}
    for fn in (fetch_direct, fetch_wayback):
        try:
            text, via = fn(url)
        except Exception as e:
            log(f"  {via if 'via' in dir() else fn.__name__} ERR {url[:60]} {type(e).__name__}")
            continue
        if acceptable(text):
            return {**base, "via": via, "status": f"ok-{via}",
                    "http_status": 200, "text_len": len(text), "text": text}
    try:
        time.sleep(3.2)  # jina 免费档限速
        text, via = fetch_jina(url)
        if acceptable(text):
            return {**base, "via": "jina", "status": "ok-jina",
                    "http_status": 200, "text_len": len(text), "text": text}
        return {**base, "via": "all", "status": "fail", "http_status": 0,
                "text_len": 0, "text": ""}
    except Exception as e:
        return {**base, "via": "all", "status": "fail", "http_status": 0,
                "text_len": 0, "text": "", "err": str(e)[:120]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: dict[str, dict] = {}
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["status"].startswith("ok"):
                    done[r["url"]] = r
    todo = [(r.get("url") or "").strip() for r in rows
            if (r.get("url") or "").strip() and (r.get("url") or "").strip() not in done]
    if a.limit:
        todo = todo[:a.limit]
    log(f"background fetch: pool={len(rows)} done={len(done)} todo={len(todo)}")

    results = dict(done)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, u): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec["status"].startswith("ok"):
                results[rec["url"]] = rec
            if i % 20 == 0:
                log(f"  progress {i}/{len(todo)}")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 写 body 版背景语料 (行保序) ----
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fb = 0
    with open(CORPUS, "w", encoding="utf-8") as f:
        for arow in rows:
            a = dict(arow)
            u = (a.get("url") or "").strip()
            rec = results.get(u)
            if rec and rec.get("text"):
                a["text"] = rec["text"]
                a["body_status"] = rec["status"]
                a["body_len"] = rec["text_len"]
                n_ok += 1
            else:
                a["text"] = a.get("title", "")
                a["body_status"] = "title_fallback"
                a["body_len"] = 0
                n_fb += 1
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    log(f"background corpus written: body_ok={n_ok} title_fallback={n_fb} "
        f"({n_ok / max(len(rows), 1):.0%})")


if __name__ == "__main__":
    main()
