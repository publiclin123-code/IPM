#!/usr/bin/env python3
"""LLM pre-label the 68 NEW rows of forward_spotcheck_v3.csv.

Judgment: is this forward signal a GENUINE warning for its event (a real,
plausible precursor with substance), or a false alarm (keyword bleed,
backward-looking report mislabeled forward, generic risk boilerplate)?

Writes llm_suggest (warning/not_warning) + llm_reason for rows where
human_label is empty. Round-1 rows already have human labels: skipped.
Idempotent; safe to resume.
"""
import argparse, csv, json, os, time
import urllib.request

BASE = "http://localhost:8080/v1/chat/completions"
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "qwen3.6-27b"
SRC = "data/annotation/forward_spotcheck_v3.csv"

PROMPT = """You are auditing whether an extracted risk signal is a GENUINE early warning.

EVENT: {event} (onset {onset})
SIGNAL DATE: {sdate}   (must be BEFORE onset to count as a warning)
SIGNAL: {desc}
TRIGGER PHRASES: {triggers}
SOURCE ARTICLE TITLE: {title}

Judge: is this signal a real, substantive precursor to the event's disruption
(not keyword bleed, not a backward-looking report of an already-realized
trigger mislabeled as forward, not generic risk boilerplate)?

Answer in strict JSON only: {{"warning": true/false, "reason": "<=15 words"}}"""


def ask(event: str, onset: str, sdate: str, desc: str, triggers: str, title: str) -> tuple[str, str]:
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [{"role": "user",
                      "content": PROMPT.format(event=event, onset=onset,
                                               sdate=sdate, desc=desc,
                                               triggers=triggers, title=title)}],
    }).encode()
    req = urllib.request.Request(BASE, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    txt = out["choices"][0]["message"]["content"].strip()
    if "{" in txt:
        txt = txt[txt.index("{"): txt.rindex("}") + 1]
    d = json.loads(txt)
    return ("warning" if d.get("warning") else "not_warning"), \
           str(d.get("reason", ""))[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not KEY:
        print("LLM_API_KEY not set; export it first")
        return 1

    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    targets = [r for r in rows
               if r["from_round1"] == "0" and not (r.get("llm_suggest") or "").strip()]
    if args.limit:
        targets = targets[:args.limit]
    print(f"待预标注: {len(targets)} 行")

    ev_onset = {}
    gt = json.load(open("validation/gt_events.json", encoding="utf-8"))["events"]
    for ev in gt:
        ev_onset[ev["event_id"]] = ev["gt_onset_date"]

    done = 0
    for r in targets:
        try:
            sug, why = ask(
                event=r["event_id"], onset=ev_onset.get(r["event_id"], ""),
                sdate=r.get("signal_date", ""), desc=r.get("description", ""),
                triggers=r.get("trigger_phrases", ""), title=r.get("article_title", ""))
        except Exception as e:
            print(f"  ERR {r['signal_id']}: {e}; stop to resume later")
            break
        r["llm_suggest"] = sug
        r["llm_reason"] = why
        done += 1
        print(f"  [{done}] {r['signal_id'][:40]:40s} {sug:12s} {why}")

    fields = list(rows[0].keys()) + ["llm_suggest", "llm_reason"] \
        if "llm_suggest" not in rows[0] else list(rows[0].keys())
    with open(SRC, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"完成 {done}/{len(targets)} -> {SRC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
