#!/usr/bin/env python3
"""pass5 wayback5 行回填 event_id/article_id (fetch 时没带).

从 data/by_event/*.jsonl 建 url -> (event_id, article_id) 映射, 重写
bodies.jsonl: 给缺失 event_id 的行补上. 同一 url 出现在多事件时写列表首个
(与 build_body_corpus 按 url_hash 合并的行为一致, 信号重抽按行进行).
必须在 pass5 进程退出后运行 (整文件重写).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BODIES = ROOT / "data" / "body_fetch" / "bodies.jsonl"
POOL = ROOT / "data" / "by_event"

url2ids = {}
for p in sorted(POOL.glob("*.jsonl")):
    eid = p.stem
    if eid.startswith("_"):
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        u = (r.get("url") or "").strip()
        if u and u not in url2ids:
            url2ids[u] = (eid, r.get("id"))

rows, n_filled, n_unmapped = [], 0, 0
for line in BODIES.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if not r.get("event_id"):
        hit = url2ids.get(r.get("url", ""))
        if hit:
            r["event_id"], r["article_id"] = hit
            n_filled += 1
        else:
            n_unmapped += 1
    rows.append(r)

BODIES.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                  encoding="utf-8")
print(f"bodies={len(rows)} filled={n_filled} unmapped={n_unmapped}")
