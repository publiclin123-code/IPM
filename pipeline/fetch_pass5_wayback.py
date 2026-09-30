#!/usr/bin/env python3
"""Pass 5: 用 Wayback Machine 补回 bodies.jsonl 里正文为反爬挑战页的 URL.

目标: best-text 为挑战页签名的 URL (822 个).
流程: archive.org/wayback/available 查快照 -> 抓快照页 -> trafilatura 抽正文
      -> 反爬签名复查 -> 干净且 >=400 字符才接受 (status=ok-wayback5).
合并规则由 build_body_corpus.py 保证: 干净正文优先于挑战页 (不比长度).

Usage:
  python pipeline/fetch_pass5_wayback.py --limit 8   # pilot
  python pipeline/fetch_pass5_wayback.py             # full (可重复跑, 自动续)
"""
import argparse
import datetime
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "body_fetch" / "bodies.jsonl"
LOG = ROOT / "data" / "body_fetch" / "pass5.log"

PROXY = "http://127.0.0.1:10809"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
H = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

MIN_TEXT = 400
WORKERS = 4

# 挑战页 / 反爬 / 存档缺页 签名 (对正文文本匹配, 大小写不敏感)
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


def load_targets() -> tuple[dict, set]:
    """返回 (url->best_text, 已恢复的 url 集)."""
    best: dict[str, str] = {}
    done: set[str] = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["status"].startswith("ok"):
            continue
        u, t = r["url"], r.get("text") or ""
        if r.get("via") == "wayback5" and not JUNK_RE.search(t) and len(t) >= MIN_TEXT:
            done.add(u)
        if u not in best or len(t) > len(best[u]):
            best[u] = t
    targets = {u: t for u, t in best.items() if JUNK_RE.search(t)}
    return targets, done


def trafil(html: str) -> str:
    try:
        import trafilatura
        return (trafilatura.extract(html) or "").strip()
    except Exception:
        return ""


def client() -> httpx.Client:
    return httpx.Client(proxy=PROXY, headers=H, timeout=httpx.Timeout(60.0),
                        follow_redirects=True, trust_env=False)


def fetch_one(url: str) -> dict | None:
    """返回 ok 记录或 None."""
    try:
        with client() as c:
            api = "https://archive.org/wayback/available?url=" + httpx.QueryParams({"u": url}).get("u", url)
            # 手工 urlencode 避免查询串歧义
            import urllib.parse
            api = "https://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe="")
            for attempt in range(3):
                r = c.get(api, timeout=20.0)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1)); continue
                break
            snap = ((r.json().get("archived_snapshots") or {}).get("closest") or {})
            if not snap.get("available"):
                return None
            ts = snap["timestamp"]
            if snap.get("status") not in ("200", "200 ", None):
                return None
            # id_ 后缀取原始资源, 避免.wayback 工具栏注入
            snap_url = f"https://web.archive.org/web/{ts}id_/{url}"
            for attempt in range(3):
                r2 = c.get(snap_url)
                if r2.status_code == 429:
                    time.sleep(8 * (attempt + 1)); continue
                break
            if r2.status_code != 200:
                return None
            text = trafil(r2.text)
            if len(text) < MIN_TEXT or JUNK_RE.search(text):
                return None
            return {"url": url,
                    "url_hash": __import__("hashlib").sha1(url.encode()).hexdigest()[:16],
                    "via": "wayback5", "status": "ok-wayback5",
                    "http_status": r2.status_code, "text_len": len(text), "text": text,
                    "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}
    except Exception as e:
        log(f"  ERR {url[:70]} -> {type(e).__name__}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()

    targets, done = load_targets()
    todo = sorted(set(targets) - done)
    if a.limit:
        todo = todo[:a.limit]
    log(f"pass5 wayback: targets={len(targets)} done={len(done)} todo={len(todo)}")

    lock = __import__("threading").Lock()
    n_ok = n_fail = 0
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, u): u for u in todo}

        def _flush():
            f.flush()

        for fut in as_completed(futs):
            u = futs[fut]
            rec = fut.result()
            with lock:
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); _flush()
                    n_ok += 1
                else:
                    n_fail += 1
                if (n_ok + n_fail) % 25 == 0:
                    log(f"  progress {n_ok + n_fail}/{len(todo)} ok={n_ok} fail={n_fail}")
    log(f"pass5 done: ok={n_ok} fail={n_fail} (fail 含无快照/非200/仍反爬/太短)")


if __name__ == "__main__":
    main()
