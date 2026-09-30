#!/usr/bin/env python3
"""免费正文补充源库 (全部免费, 无 API key, 无付费服务).

源与顺序 (cascade):
  1. wayback6     CDX API 多快照 (修 availability 只给 closest 单快照的缺陷)
  2. archivetoday archive.ph/is/today 最新快照
  3. memento      timetravel.mementoweb.org 聚合器 (跳过已试过的 web.archive.org)
  4. allorigins   api.allorigins.win/raw 换 IP 直抓 (绕 IP 封锁)
  5. codetabs     api.codetabs.com/v1/proxy 换 IP 直抓
  6. commoncrawl  Common Crawl WARC 索引 (老文章利器)
  7. jina6        r.jina.ai 免费档兜底 (调用方控制 sleep)

每一级产出都必须过 acceptable(): >= MIN_TEXT 且无 JUNK_RE 签名.
"""
import gzip
import hashlib
import json
import re
import time
import urllib.parse

import httpx

PROXY = "http://127.0.0.1:10809"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
H = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

MIN_TEXT = 250

JUNK_RE = re.compile(
    r"cloudflare|verify you are human|just a moment\.|enable javascript and cookies"
    r"|attention required|checking your browser|challenge-platform|are you a robot"
    r"|captcha|perimeterx|datadome|incapsula|sucuri website firewall|ddos protection by"
    r"|wayback machine (has not|doesn'?t have|does not have)", re.I)

CC_INDEXES = ["CC-MAIN-2024-10", "CC-MAIN-2021-43", "CC-MAIN-2018-13"]
AT_DOMAINS = ["archive.ph", "archive.today", "archive.is"]


def client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(proxy=PROXY, headers=H, timeout=httpx.Timeout(timeout),
                        follow_redirects=True, trust_env=False)


def trafil(html: str) -> str:
    try:
        import trafilatura
        return (trafilatura.extract(html) or "").strip()
    except Exception:
        return ""


def clean_jina(text: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.M)
    t = re.sub(r"[-*]{3,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def acceptable(text: str) -> bool:
    return len(text) >= MIN_TEXT and not JUNK_RE.search(text)


# ---------------- 各源 ----------------

def wayback6(url: str) -> tuple[str, str]:
    """CDX 列出全部 200 快照, 依次试前 3 个 (collapse=digest 去重)."""
    with client() as c:
        for attempt in range(3):
            try:
                r = c.get("https://web.archive.org/cdx/search/cdx", params=[
                    ("url", url), ("output", "json"), ("filter", "statuscode:200"),
                    ("collapse", "digest"), ("limit", "10")], timeout=30.0)
                break
            except Exception:
                time.sleep(4 * (attempt + 1))   # 与 pass5 同宿主竞争时的连接重试
        else:
            return "", "cdx_conn"
        if r.status_code != 200:
            return "", f"cdx{r.status_code}"
        rows = r.json()
        if not rows or len(rows) < 2:
            return "", "no_snapshot"
        for row in rows[1:4]:                # ts, original, ...
            ts, orig = row[1], row[2]
            time.sleep(0.5)                  # 对 web.archive.org 礼貌
            try:
                r2 = c.get(f"https://web.archive.org/web/{ts}id_/{orig}", timeout=60.0)
            except Exception:
                continue
            if r2.status_code != 200:
                continue
            t = trafil(r2.text)
            if acceptable(t):
                return t, "wayback6"
        return "", "cdx_all_bad"


def archivetoday(url: str) -> tuple[str, str]:
    for d in AT_DOMAINS:
        try:
            with client() as c:
                r = c.get(f"https://{d}/newest/{url}", timeout=35.0)
                if r.status_code != 200:
                    continue
                html = r.text
                m = re.search(r'href="/(\d{12,14})/https?://', html)
                if not m:
                    continue
                r2 = c.get(f"https://{d}/{m.group(1)}/{url}", timeout=35.0)
                if r2.status_code != 200:
                    continue
                t = trafil(r2.text)
                if acceptable(t):
                    return t, "archivetoday"
        except Exception:
            continue
    return "", "at_miss"


def memento(url: str) -> tuple[str, str]:
    try:
        with client() as c:
            r = c.get("http://timetravel.mementoweb.org/api/json/20200101000000/" + url,
                      timeout=25.0)
            if r.status_code != 200:
                return "", f"mem{r.status_code}"
            mems = ((r.json().get("mementos") or {}).get("all")
                    or (r.json().get("mementos") or {}).get("list") or [])
            for m in mems[:6]:
                uri = m.get("uri", "")
                if "web.archive.org" in uri:      # CDX 已试过
                    continue
                try:
                    r2 = c.get(uri, timeout=40.0)
                    if r2.status_code == 200:
                        t = trafil(r2.text)
                        if acceptable(t):
                            return t, "memento"
                except Exception:
                    continue
    except Exception:
        pass
    return "", "memento_miss"


def allorigins(url: str) -> tuple[str, str]:
    with client() as c:
        r = c.get("https://api.allorigins.win/raw?url="
                  + urllib.parse.quote(url, safe=""), timeout=45.0)
        if r.status_code != 200:
            return "", f"ao{r.status_code}"
        return trafil(r.text), "allorigins"


def codetabs(url: str) -> tuple[str, str]:
    with client() as c:
        r = c.get("https://api.codetabs.com/v1/proxy/?quest="
                  + urllib.parse.quote(url, safe=""), timeout=45.0)
        if r.status_code != 200:
            return "", f"ct{r.status_code}"
        return trafil(r.text), "codetabs"


def commoncrawl(url: str) -> tuple[str, str]:
    for idx in CC_INDEXES:
        try:
            with client() as c:
                r = c.get(f"https://index.commoncrawl.org/{idx}-index",
                          params={"url": url, "output": "json"}, timeout=30.0)
                if r.status_code != 200:
                    continue
                for line in r.text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    fn, off, ln = rec.get("filename"), rec.get("offset"), rec.get("length")
                    if not (fn and off and ln):
                        continue
                    r2 = c.get(f"https://data.commoncrawl.org/{fn}",
                               headers={"Range": f"bytes={off}-{off + ln - 1}"},
                               timeout=60.0)
                    if r2.status_code not in (200, 206):
                        continue
                    raw = r2.content
                    try:
                        raw = gzip.decompress(raw)      # 每条记录独立 gzip 成员
                    except Exception:
                        pass
                    # 剥 WARC 信封 + HTTP 头
                    if raw.startswith(b"WARC/"):
                        _, _, raw = raw.partition(b"\r\n\r\n")
                    if raw.startswith(b"HTTP/"):
                        _, _, raw = raw.partition(b"\r\n\r\n")
                    try:
                        html = raw.decode("utf-8", "ignore")
                    except Exception:
                        continue
                    t = trafil(html)
                    if acceptable(t):
                        return t, "commoncrawl"
        except Exception:
            continue
    return "", "cc_miss"


def jina6(url: str, sleep_s: float = 3.2) -> tuple[str, str]:
    time.sleep(sleep_s)
    with client() as c:
        r = c.get("https://r.jina.ai/" + url, timeout=60.0)
        if r.status_code != 200:
            return "", f"jina{r.status_code}"
        return clean_jina(r.text), "jina6"


# ---------------- 级联 ----------------

CASCADE = [wayback6, archivetoday, memento, allorigins, codetabs, commoncrawl]


def cascade(url: str, jina_sleep: float = 3.2,
            errlog=None) -> dict | None:
    """按顺序尝试所有免费源, 返回 ok 记录或 None."""
    base = {"url": url,
            "url_hash": hashlib.sha1(url.encode()).hexdigest()[:16]}
    for fn in CASCADE:
        try:
            text, via = fn(url)
        except Exception as e:
            if errlog:
                errlog(f"  {fn.__name__} ERR {url[:60]} {type(e).__name__}")
            continue
        if acceptable(text):
            import datetime
            return {**base, "via": via, "status": f"ok-{via}",
                    "http_status": 200, "text_len": len(text), "text": text,
                    "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        text, via = jina6(url, sleep_s=jina_sleep)
        if acceptable(text):
            import datetime
            return {**base, "via": via, "status": f"ok-{via}",
                    "http_status": 200, "text_len": len(text), "text": text,
                    "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}
    except Exception:
        pass
    return None


def url2ids(pool_dir) -> dict[str, tuple[str, str]]:
    """url -> (event_id, article_id), 取首个出现的事件 (与语料合并口径一致)."""
    m: dict[str, tuple[str, str]] = {}
    for p in sorted(pool_dir.glob("*.jsonl")):
        if p.stem.startswith("_"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = (r.get("url") or "").strip()
            if u and u not in m:
                m[u] = (p.stem, r.get("id"))
    return m
