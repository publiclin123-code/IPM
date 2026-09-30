"""I3: does the 2019 Toyota supplier explosion exist? (resolves DATA_ISSUES DI-3)

Why this is blocking
--------------------
`toyota_steel_explosion_2019` is one of only three sudden-class events with any
hits, and the manuscript uses it to argue that type-dependent foresight is a
central tendency with overlap rather than a clean split. Its 27 true positives
turned out to be US steel-tariff articles (`audit_match_evidence.py`), and the
identity matcher cut them to 1.

Before restating any numbers, the event itself has to be established:

  - the ground truth calls it "Toyota engine plant steel explosion (Japan)",
    onset 2019-06-04, onset_status draft-needs-verification
  - a Wikipedia search for the exact date found nothing, and the nearest hit was
    the *1997 Aisin fire* -- Aisin is also a Toyota supplier
  - if no source documents a 2019 event, the event leaves the ground truth and
    every count changes again

This script searches for corroborating sources on both language editions and
prints everything it finds, including negative evidence.

Usage
-----
  python3 pipeline/verify_toyota_event.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

PROXY = "socks5h://127.0.0.1:10808"
UA = "event-verify/1.0 (research; contact: contact@example.invalid)"
ROOT = Path(__file__).resolve().parent.parent

SEARCHES = [
    ("en", "Toyota plant explosion 2019"),
    ("en", "Toyota supplier explosion 2019 production halt"),
    ("en", "Toyota factory explosion Japan 2019"),
    ("en", "Aisin fire 1997 Toyota"),
    ("en", "Toyota engine plant explosion"),
    ("en", "2019 Toyota production halt supplier"),
    ("ja", "トヨタ 工場 爆発 2019"),
    ("ja", "トヨタ サプライヤー 爆発 2019 生産停止"),
    ("ja", "2019年 トヨタ 工場火災"),
    ("ja", "アイシン 火災 1997"),
]

# Ground-truth claims, for the report.
CLAIM = {
    "event_id": "toyota_steel_explosion_2019",
    "event_name": "Toyota engine plant steel explosion (Japan)",
    "gt_onset_date": "2019-06-04",
    "onset_status": "draft-needs-verification",
    "event_type": "quality",
    "commodities": ["automotive engines", "steel"],
    "companies": ["Toyota"],
    "geographies": ["Japan"],
}


def curl(url: str, timeout: int = 45) -> str:
    cmd = ["curl", "-sS", "--max-time", str(timeout), "-x", PROXY, "-L",
           "-H", f"User-Agent: {UA}", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return ""
    return r.stdout if r.returncode == 0 else ""


def api(host: str, params: dict) -> dict | None:
    u = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    try:
        return json.loads(curl(u))
    except Exception:
        return None


def search(host: str, q: str, limit: int = 6) -> list[tuple[str, str]]:
    d = api(host, {"action": "query", "list": "search", "srsearch": q,
                   "srlimit": str(limit), "format": "json"})
    if not d:
        return []
    out = []
    for x in d.get("query", {}).get("search", []):
        snip = re.sub(r"<[^>]+>", "", x.get("snippet", ""))
        out.append((x["title"], snip))
    return out


def article_text(host: str, title: str) -> str:
    d = api(host, {"action": "query", "prop": "revisions", "rvprop": "content",
                   "rvslots": "main", "titles": title, "format": "json",
                   "formatversion": "2"})
    try:
        return d["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return ""


def clean(t: str) -> str:
    t = re.sub(r"<ref[^>/]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", " ", t)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


def main() -> int:
    print("=" * 100)
    print("STEP 1 - the claim under test")
    print("=" * 100)
    for k, v in CLAIM.items():
        print(f"  {k:16s} {v}")

    print()
    print("=" * 100)
    print("STEP 2 - what the corpus actually contains")
    print("=" * 100)
    p = ROOT / "data" / "by_event" / "toyota_steel_explosion_2019.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  corpus rows: {len(rows)}")
    slugs = [r.get("title", "") for r in rows]
    for kw in ("toyota", "explos", "steel", "fire", "aichi", "japan"):
        n = sum(1 for s in slugs if kw in s.lower())
        print(f"    slug contains {kw!r:10s}: {n:4d}  ({n / len(rows) * 100:.1f}%)")
    print()
    print("  titles containing BOTH toyota and (explosion OR fire):")
    both = [s for s in slugs if "toyota" in s.lower() and ("explos" in s.lower() or "fire" in s.lower())]
    if both:
        for s in both[:10]:
            print(f"    - {s[:100]}")
    else:
        print("    (none)")
    print()
    print("  URLs whose date falls in the week after the claimed onset 2019-06-04:")
    near = [r for r in rows if "2019-06-0" in str(r.get("date", "")) or "2019-06-1" in str(r.get("date", ""))]
    print(f"    {len(near)} rows")
    for r in near[:8]:
        print(f"    [{r.get('date')}] {str(r.get('title'))[:88]}")

    print()
    print("=" * 100)
    print("STEP 3 - encyclopedia search (en + ja)")
    print("=" * 100)
    hits: list[tuple] = []
    for lang, q in SEARCHES:
        host = f"{lang}.wikipedia.org"
        res = search(host, q)
        print(f"\n  [{lang}] Q: {q}   -> {len(res)} hits")
        for t, snip in res:
            print(f"     - {t}")
            if snip:
                print(f"       {snip[:150]}")
            hits.append((lang, q, t, snip))

    print()
    print("=" * 100)
    print("STEP 4 - look for a dated 2019 explosion sentence in any hit")
    print("=" * 100)
    DATE = re.compile(r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
                      r"September|October|November|December)\s+\d{4}|"
                      r"(?:January|February|March|April|May|June|July|August|September|"
                      r"October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|"
                      r"\d{4}年\d{1,2}月\d{1,2}日)\b")
    found = False
    seen = set()
    for lang, q, title, _ in hits:
        key = (lang, title)
        if key in seen:
            continue
        seen.add(key)
        text = clean(article_text(f"{lang}.wikipedia.org", title))
        if not text:
            continue
        for sent in re.split(r"(?<=[。！？.!?])\s+", text):
            if not (40 < len(sent) < 600):
                continue
            low = sent.lower()
            if "toyota" not in low and "トヨタ" not in sent:
                continue
            if not any(w in low or w in sent for w in ("explos", "爆発", "fire", "火災")):
                continue
            ds = DATE.findall(sent)
            if not ds:
                continue
            yr2019 = "2019" in sent
            mark = "  <<< 2019 !!!" if yr2019 else ""
            print(f"  [{title[:38]:38s}] {', '.join(dict.fromkeys(ds))[:24]:24s} "
                  f"{sent[:130]}{mark}")
            if yr2019:
                found = True

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  a date-bearing 2019 Toyota explosion/fire sentence on Wikipedia: "
          f"{'FOUND' if found else 'NOT FOUND'}")
    print()
    print("  Interpretation rules:")
    print("   - FOUND -> the event may be real; verify the date against the source")
    print("     and keep it in the ground truth.")
    print("   - NOT FOUND -> Wikipedia is not proof of absence. It does mean the")
    print("     event cannot be cited from an encyclopaedia, so corroborate from")
    print("     industry or news sources, or remove the event. Removal is the")
    print("     safer default: the event currently contributes 1 hit under the")
    print("     identity rule, so nothing in the analysis rests on it, while")
    print("     keeping an unverifiable event invites a reviewer challenge to the")
    print("     entire ground-truth table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
