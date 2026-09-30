"""Third-pass probe: Japanese Wikipedia, for the two Japanese industrial events.

The English edition does not cover the Renesas Naka fire or a 2019 Toyota
supplier explosion. The search for the latter surfaced the 1997 Aisin fire, and
Aisin is a Toyota supplier, which raises the possibility that the stored event
is a conflation. Japanese sources are the natural place to settle it.
"""
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROXY = "socks5h://127.0.0.1:10808"
UA = "onset-verify/1.0 (research; contact: contact@example.invalid)"


def curl(url, timeout=50):
    r = subprocess.run(["curl", "-sS", "--max-time", str(timeout), "-x", PROXY,
                        "-L", "-H", f"User-Agent: {UA}", url],
                       capture_output=True, text=True, timeout=timeout + 15)
    return r.stdout if r.returncode == 0 else ""


def api(host, params):
    u = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    out = curl(u)
    try:
        return json.loads(out)
    except Exception:
        return None


def article(host, title):
    d = api(host, {"action": "query", "prop": "revisions", "rvprop": "content",
                   "rvslots": "main", "titles": title, "format": "json",
                   "formatversion": "2"})
    try:
        return d["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return ""


def clean(t):
    t = re.sub(r"<ref[^>/]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", " ", t)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


DATE = re.compile(
    r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}\s+(?:January|February|March|April|May|"
    r"June|July|August|September|October|November|December)\s+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})")

JOBS = [
    ("ja.wikipedia.org", "ルネサスエレクトロニクス", ["那珂", "火災", "工場"], "Naka fire"),
    ("ja.wikipedia.org", "ルネサスエレクトロニクス", ["火災"], "Naka fire (loose)"),
    ("ja.wikipedia.org", "熊本地震 (2016年)", ["地震", "発生"], "Kumamoto 2016"),
    ("ja.wikipedia.org", "2020年から2023年の半導体不足", ["火災", "那珂"], "chip shortage ja"),
    ("en.wikipedia.org", "Aisin", ["fire", "1997", "explos"], "1997 Aisin fire"),
    ("ja.wikipedia.org", "アイシン", ["火災"], "Aisin ja"),
    ("en.wikipedia.org", "Toyota", ["explos", "halted", "production"], "Toyota explosions"),
]

for host, title, terms, label in JOBS:
    raw = article(host, title)
    if not raw:
        print(f"\n### [{label}] {title} @ {host}: NO CONTENT")
        continue
    text = clean(raw)
    print(f"\n{'='*100}\n### [{label}] {title} @ {host}")
    n = 0
    for sent in re.split(r"(?<=[。！？.!?])\s+", text):
        if not (30 < len(sent) < 700):
            continue
        low = sent.lower()
        if not any(t.lower() in low for t in terms):
            continue
        ds = DATE.findall(sent)
        if not ds:
            continue
        print(f"   [{', '.join(dict.fromkeys(ds))[:26]:26s}] {sent[:210]}")
        n += 1
        if n >= 10:
            break
    if n == 0:
        print("   (no dated sentences matched)")
