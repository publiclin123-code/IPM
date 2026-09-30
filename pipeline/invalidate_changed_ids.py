#!/usr/bin/env python3
"""pass4/wayback5/pass6 捞回正文后, 使这些文章的旧 done 行失效, 让 resume 重新提取.

新正文判定: bodies.jsonl 里 via 属于回收类 (pass4 重试 + pass5 wayback +
pass6 多源: cdx/archivetoday/memento/allorigins/codetabs/commoncrawl/jina6).
对每个 (event_id, article_id): 删除 results/v2_body/{eid}_signals.jsonl 中
该 input_id 的所有行 → resume 视为未处理, 用新正文重抽.

Usage: python pipeline/invalidate_changed_ids.py [--dry-run]
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BODIES = ROOT / "data" / "body_fetch" / "bodies.jsonl"
RES = ROOT / "results" / "v2_body"
PASS4_VIA = {"direct-retry", "jina", "wayback5", "wayback6", "archivetoday",
             "memento", "allorigins", "codetabs", "commoncrawl", "jina6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = {}          # eid -> set(input_id)
    for line in BODIES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("via") in PASS4_VIA and r.get("status", "").startswith("ok"):
            changed.setdefault(r["event_id"], set()).add(r["article_id"])

    total_ids = sum(len(v) for v in changed.values())
    print(f"pass4-recovered articles: {total_ids} across {len(changed)} events")
    if not total_ids:
        print("nothing to invalidate")
        return

    removed = 0
    for eid, ids in changed.items():
        sig = RES / f"{eid}_signals.jsonl"
        if not sig.exists():
            continue
        rows = [l for l in sig.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep, drop = [], 0
        for l in rows:
            try:
                r = json.loads(l)
            except Exception:
                keep.append(l); continue
            if r.get("input_id") in ids:
                drop += 1
            else:
                keep.append(l)
        if drop:
            removed += drop
            print(f"  {eid}: drop {drop} stale rows (of {len(rows)})")
            if not args.dry_run:
                sig.write_text("\n".join(keep) + ("\n" if keep else ""),
                               encoding="utf-8")
    print(f"{'would remove' if args.dry_run else 'removed'} {removed} rows "
          f"-> resume will re-extract with new bodies")


if __name__ == "__main__":
    main()
