"""Second-pass probe: find date evidence for the three events that returned empty.

Reuses the fetch helpers from verify_onsets.py but greps the article body
directly for the entity name, so that a mention without a matching keyword in
the same sentence is still surfaced.
"""
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_onsets import api, plain_text, clean, DATE_PATTERNS  # noqa: E402

PROXY = "socks5h://127.0.0.1:10808"

SEARCHES = {
    "renesas_naka_fire": [
        "Renesas Naka plant fire", "Naka fire Renesas 2021",
        "automotive semiconductor shortage fire Japan",
    ],
    "toyota_explosion": [
        "Toyota plant explosion", "Toyota supplier explosion 2019",
        "Toyota production halt 2019", "Chuo Spring explosion",
    ],
    "us_chip_controls": [
        "October 2022 semiconductor export controls China",
        "Bureau of Industry and Security export controls 2022",
        "United States restrictions on semiconductor exports to China",
    ],
}

GREP_TERMS = {
    "renesas_naka_fire": ["naka", "fire"],
    "toyota_explosion": ["explos", "halt", "suspend"],
    "us_chip_controls": ["october 2022", "export control", "advanced computing"],
}


def show(title: str, terms: list[str], yr: int | None = None) -> int:
    wt = plain_text(title, PROXY)
    if not wt:
        print(f"  [no content] {title}")
        return 0
    text = clean(wt)
    n = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not (40 < len(sent) < 700):
            continue
        low = sent.lower()
        if not any(t in low for t in terms):
            continue
        ds = [m.group(0) for p in DATE_PATTERNS for m in p.finditer(sent)]
        if not ds:
            continue
        if yr and not re.search(rf"\b{yr}\b", sent):
            continue
        print(f"     [{', '.join(dict.fromkeys(ds))[:30]:30s}] {sent[:200]}")
        n += 1
        if n >= 12:
            break
    if n == 0:
        print(f"  [no dated sentences] {title}")
    return n


for label, queries in SEARCHES.items():
    print("=" * 100)
    print(f"### {label}")
    seen: set[str] = set()
    for q in queries:
        d = api({"action": "query", "list": "search", "srsearch": q,
                 "srlimit": "5", "format": "json"}, PROXY)
        titles = [x["title"] for x in (d or {}).get("query", {}).get("search", [])]
        print(f"\n  Q: {q}")
        for t in titles:
            print(f"     - {t}")
            if t not in seen:
                seen.add(t)
                show(t, GREP_TERMS[label])
