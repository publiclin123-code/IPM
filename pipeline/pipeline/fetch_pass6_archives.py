#!/usr/bin/env python3
"""Pass 6: pass5 之后的深度回收, 用免费多源级联 (free_archives.cascade).

目标: bodies.jsonl 里 best-text 仍为挑战页, 且尚无任何干净 ok 记录的 URL.
写入: ok-<via> 记录追加到 bodies.jsonl (自带 event_id/article_id).
可重复跑: done 集排除已恢复 URL.

Usage:
  python pipeline/fetch_pass6_archives.py --limit 8   # pilot
  python pipeline/fetch_pass6_archives.py             # full
"""
import argparse
import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from free_archives import JUNK_RE, cascade, url2ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "body_fetch" / "bodies.jsonl"
LOG = ROOT / "data" / "body_fetch" / "pass6.log"


def log(msg: str) -> None:
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_targets() -> tuple[list[str], set[str]]:
    best: dict[str, str] = {}
    clean_ok: set[str] = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["status"].startswith("ok"):
            continue
        u, t = r["url"], r.get("text") or ""
        if t and not JUNK_RE.search(t):
            clean_ok.add(u)
        if u not in best or len(t) > len(best[u]):
            best[u] = t
    targets = sorted(u for u, t in best.items()
                     if JUNK_RE.search(t) and u not in clean_ok)
    return targets, clean_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    targets, clean_ok = load_targets()
    todo = targets
    if a.limit:
        todo = todo[:a.limit]
    log(f"pass6 archives: junk_targets={len(targets)} clean_ok={len(clean_ok)} "
        f"todo={len(todo)}")

    ids = url2ids(ROOT / "data" / "by_event")
    lock = threading.Lock()
    n_ok = n_fail = 0

    def work(u: str) -> dict | None:
        rec = cascade(u, errlog=lambda m: log(m))
        if rec:
            hit = ids.get(u)
            if hit:
                rec["event_id"], rec["article_id"] = hit
        return rec

    with open(OUT, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, u): u for u in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                else:
                    n_fail += 1
                if (n_ok + n_fail) % 25 == 0:
                    log(f"  progress {n_ok + n_fail}/{len(todo)} ok={n_ok} fail={n_fail}")
    log(f"pass6 done: ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
