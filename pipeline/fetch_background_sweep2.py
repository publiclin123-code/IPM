#!/usr/bin/env python3
"""背景池二扫: 对 fetch_background_bodies 的 149 个 title_fallback 用免费多源级联.

做完三件事:
  1. cascade 抓取, 新 ok 记录并入 background_bodies.jsonl
  2. 重写 data/by_event_body/_background.jsonl (行保序, title 回退)
  3. 把正文发生变化的 article id 从两个 bg 信号文件里删掉 (resume 重抽)
"""
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from free_archives import cascade  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "by_event" / "_background.jsonl"
OUT = ROOT / "data" / "body_fetch" / "background_bodies.jsonl"
CORPUS = ROOT / "data" / "by_event_body" / "_background.jsonl"
LOG = ROOT / "data" / "body_fetch" / "background_sweep2.log"
SIGS = [ROOT / "results" / "_background" / "signals_postevent_naive_body.jsonl",
        ROOT / "results" / "_background" / "signals_postevent_v2_body.jsonl"]


def log(msg: str) -> None:
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    results: dict[str, dict] = {}
    for l in OUT.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r["status"].startswith("ok"):
                results[r["url"]] = r
    todo = sorted({(r.get("url") or "").strip() for r in rows
                   if (r.get("url") or "").strip()} - set(results))
    log(f"sweep2: pool={len(rows)} ok={len(results)} todo={len(todo)}")

    # jina sleep 放大到 6.5s: 可能与 pass6 并行, 两进程合计仍 <20 RPM
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(cascade, u, 6.5, None): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec:
                results[rec["url"]] = rec
            if i % 20 == 0:
                log(f"  progress {i}/{len(todo)}")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 重写 corpus, 记录变化 id
    changed_ids: set[str] = set()
    n_ok = n_fb = 0
    with open(CORPUS, "w", encoding="utf-8") as f:
        for a in rows:
            d = dict(a)
            rec = results.get((d.get("url") or "").strip())
            if rec and rec.get("text"):
                if d.get("body_status") != rec["status"]:
                    changed_ids.add(d["id"])
                d["text"] = rec["text"]
                d["body_status"] = rec["status"]
                d["body_len"] = rec["text_len"]
                n_ok += 1
            else:
                if d.get("body_status") not in (None, "title_fallback"):
                    changed_ids.add(d["id"])
                d["text"] = d.get("title", "")
                d["body_status"] = "title_fallback"
                d["body_len"] = 0
                n_fb += 1
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log(f"corpus rewritten: body_ok={n_ok} title_fallback={n_fb} changed_ids={len(changed_ids)}")

    # 失效: 从信号文件删变化 id 的行
    for sig in SIGS:
        if not sig.exists():
            continue
        lines = [l for l in sig.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep, dropped = [], 0
        for l in lines:
            r = json.loads(l)
            key = r.get("input_id") or r.get("id") or ""
            if key in changed_ids:
                dropped += 1
            else:
                keep.append(l)
        sig.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        log(f"invalidate {sig.name}: dropped={dropped} kept={len(keep)}")


if __name__ == "__main__":
    main()
