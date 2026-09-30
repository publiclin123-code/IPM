#!/usr/bin/env python3
"""为 389 工作表的有正文行预填 LLM 建议标签 (人工只做确认/改判).

读 data/annotation/retrieval_audit_sample_body.csv,
对 article_body 非空的行: qwen3.6-27b 读 (event_name, title, body)
  -> {relevant: true/false, reason: "..."}
写两列: llm_suggest (relevant/irrelevant), llm_reason.
幂等: 已有 llm_suggest 的行跳过. 断点续跑安全.

Usage: python pipeline/prefill_annotation_body.py [--limit 5]
"""
import argparse, csv, json, os, time
import urllib.request

BASE = "http://localhost:8080/v1/chat/completions"
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "qwen3.6-27b"
SRC = "data/annotation/retrieval_audit_sample_body.csv"

PROMPT = """You are auditing news-retrieval precision. An article was retrieved by keyword filter for this disruption event:

EVENT: {event} ({eid})

ARTICLE TITLE: {title}

ARTICLE BODY (truncated): {body}

Question: Is this article substantively about this event or its supply-chain disruption topic (causes, evolution, impacts, related policy/companies)? A mere keyword coincidence (same word, unrelated story) means irrelevant.

Answer in strict JSON only: {{"relevant": true/false, "reason": "<=15 words"}}"""


def ask(event: str, eid: str, title: str, body: str) -> tuple[str, str]:
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [{"role": "user",
                      "content": PROMPT.format(event=event, eid=eid,
                                               title=title, body=body)}],
    }).encode()
    req = urllib.request.Request(BASE, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    txt = out["choices"][0]["message"]["content"].strip()
    # 容错: 抓 JSON 子串
    if "{" in txt:
        txt = txt[txt.index("{"): txt.rindex("}") + 1]
    d = json.loads(txt)
    return ("relevant" if d.get("relevant") else "irrelevant"), \
           str(d.get("reason", ""))[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not KEY:
        raise SystemExit("set LLM_API_KEY env")

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    fields = list(rows[0].keys())
    for c in ("llm_suggest", "llm_reason"):
        if c not in fields:
            fields.append(c)

    todo = [r for r in rows
            if (r.get("article_body") or "").strip()
            and not (r.get("llm_suggest") or "").strip()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"prefill targets: {len(todo)} rows")

    done = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            lab, why = ask(r["event_name"], r["event_id"],
                           r["article_title"], r["article_body"][:4000])
            r["llm_suggest"], r["llm_reason"] = lab, why
            done += 1
        except Exception as e:
            fail += 1
            r["llm_suggest"], r["llm_reason"] = "", f"error: {str(e)[:60]}"
        if i % 25 == 0:
            # 中途落盘 (幂等续跑)
            with open(SRC, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
            print(f"  {i}/{len(todo)} done={done} fail={fail}")
    with open(SRC, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"PREFILL DONE: {done} ok, {fail} fail")


if __name__ == "__main__":
    main()
