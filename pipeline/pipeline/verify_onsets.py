"""Onset verification via Wikipedia (resolves DATA_ISSUES DI-3).

Seven of the eighteen news events in `validation/gt_events.json` carry
`onset_status = "draft-needs-verification"`, and those seven carry 39.7% of the
forward signals and 35.7% of the true positives in the manuscript. An
unverified onset date is not publishable, because every lead-time figure is
measured against it.

This script gathers citable evidence for each draft onset from Wikipedia and
reports, per event, the sentences that mention a date near the event, so that a
human can confirm or correct the stored date and record the source in
`onset_evidence`.

It does not rewrite the ground truth. Verification is a judgement call and stays
with the author; the script only assembles the evidence.

Network note: `en.wikipedia.org` is unreachable directly from this host (the
TLS handshake is reset), so requests go through the local sing-box SOCKS
listener. Override with --proxy if the port differs.

Usage
-----
  python3 pipeline/verify_onsets.py                    # all draft onsets
  python3 pipeline/verify_onsets.py --event red_sea_crisis_2023
  python3 pipeline/verify_onsets.py --out reports/onsets.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "validation" / "gt_events.json"
DEFAULT_PROXY = "socks5h://127.0.0.1:10808"
UA = "onset-verify/1.0 (research; contact: contact@example.invalid)"

# Per-event article candidates and the keywords that mark a relevant sentence.
# Titles verified to exist via the Wikipedia search API on 2026-09-17.
ARTICLE_HINTS: dict[str, tuple[list[str], list[str]]] = {
    "red_sea_crisis_2023": (
        ["Hijacking of the Galaxy Leader", "Red Sea crisis", "Galaxy Leader"],
        ["hijack", "seiz", "boarded", "Houthi", "attacked"],
    ),
    "us_chip_export_controls_2022": (
        ["China\u2013United States chip war", "Semiconductor industry in China",
         "United States export controls"],
        ["export control", "October 2022", "BIS", "announc", "restrict"],
    ),
    "covid_supply_disruption_2020": (
        ["COVID-19 lockdowns", "COVID-19 pandemic in Hubei",
         "2020\u20132023 global chip shortage"],
        ["Wuhan", "lockdown", "quarantin", "January 2020", "travel"],
    ),
    "renesas_earthquake_2016": (
        ["2016 Kumamoto earthquakes", "Renesas Electronics",
         "2020\u20132023 global chip shortage"],
        ["Kumamoto", "Renesas", "earthquake", "April 2016", "halt", "suspend"],
    ),
    "toyota_steel_explosion_2019": (
        ["Toyota", "2020\u20132023 global chip shortage", "Aisin"],
        ["explos", "plant", "halt", "suspend", "production"],
    ),
    "europe_energy_crisis_2022": (
        ["Nord Stream 1", "Nord Stream pipelines sabotage",
         "2021\u20132023 global energy crisis"],
        ["September 2022", "halt", "shut", "indefinite", "suspend", "flow"],
    ),
    "renesas_naka_plant_fire_2021": (
        ["Renesas Electronics", "Naka, Ibaraki",
         "2020\u20132023 global chip shortage"],
        ["Naka", "Renesas", "fire", "March 2021", "halt", "suspend"],
    ),
}

# Terms whose neighbourhood we scan for dates.
CONTEXT_PATTERNS = [
    r"hang", r"lockdown", r"announce", r"effective", r"impos", r"cut",
    r"halt", r"shut", r"explos", r"fire", r"earthquake", r"seiz", r"hijack",
    r"struck", r"grounded", r"suspend", r"resume", r"begin", r"start",
]

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")

DATE_PATTERNS = [
    re.compile(rf"\b(?P<d>\d{{1,2}})\s+(?:{MONTHS})\s+(?P<y>\d{{4}})\b"),
    re.compile(rf"\b(?:{MONTHS})\s+(?P<d>\d{{1,2}}),?\s+(?P<y>\d{{4}})\b"),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
]


def curl(url: str, proxy: str, timeout: int = 45) -> str | None:
    cmd = ["curl", "-sS", "-L", "--max-time", str(timeout),
           "-H", f"User-Agent: {UA}", url]
    if proxy:
        cmd = cmd[:1] + ["-x", proxy] + cmd[1:]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def api(params: dict, proxy: str) -> dict | None:
    q = urllib.parse.urlencode(params)
    url = f"https://en.wikipedia.org/w/api.php?{q}"
    out = curl(url, proxy)
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def search_articles(query: str, proxy: str, limit: int = 5) -> list[str]:
    d = api({"action": "query", "list": "search", "srsearch": query,
             "srlimit": str(limit), "format": "json"}, proxy)
    if not d:
        return []
    return [x["title"] for x in d.get("query", {}).get("search", [])]


def plain_text(title: str, proxy: str) -> str | None:
    """Fetch article wikitext (has full history detail; summaries do not)."""
    d = api({"action": "query", "prop": "revisions", "rvprop": "content",
             "rvslots": "main", "titles": title, "format": "json",
             "formatversion": "2"}, proxy)
    if not d:
        return None
    pages = d.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None
    return pages[0]["revisions"][0]["slots"]["main"]["content"]


def clean(wikitext: str) -> str:
    t = wikitext
    t = re.sub(r"<ref[^>/]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", " ", t)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def sentences_with_dates(text: str, target_year: int,
                         keywords: list[str] | None = None) -> list[tuple[str, str]]:
    """Return (date_string, sentence) pairs for sentences relevant to the event.

    A sentence qualifies when it carries an explicit date, mentions the target
    year, and contains one of the event keywords. Requiring the keyword keeps
    the output short enough to read, which is the whole point: a human has to
    check each line against the stored date.
    """
    out: list[tuple[str, str]] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(sent) < 30 or len(sent) > 700:
            continue
        if not re.search(rf"\b{target_year}\b", sent):
            continue
        if keywords:
            if not any(k.lower() in sent.lower() for k in keywords):
                continue
        elif not any(re.search(p, sent, re.I) for p in CONTEXT_PATTERNS):
            continue
        dates = []
        for pat in DATE_PATTERNS:
            for m in pat.finditer(sent):
                dates.append(m.group(0))
        if dates:
            out.append((" | ".join(dict.fromkeys(dates)), sent.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify draft onset dates via Wikipedia")
    ap.add_argument("--event", default="", help="single event id")
    ap.add_argument("--proxy", default=DEFAULT_PROXY, help="socks proxy ('' to go direct)")
    ap.add_argument("--out", default="", help="write a markdown report here")
    ap.add_argument("--all", action="store_true",
                    help="re-verify every event, not only those still marked "
                         "draft-needs-verification (useful for regenerating the "
                         "evidence trail once all onsets are verified)")
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))["events"]
    if args.event:
        gt = [e for e in gt if e["event_id"] == args.event]
        if not gt:
            print(f"unknown event: {args.event}", file=sys.stderr)
            return 2
    elif not args.all:
        gt = [e for e in gt if e.get("onset_status") != "verified"]

    print(f"verifying {len(gt)} event(s) via {args.proxy or 'direct connection'}")
    print("=" * 100)

    report_lines = ["# Onset verification report", "",
                    "Generated by `pipeline/verify_onsets.py`. Evidence only; the "
                    "stored date is not modified automatically.", ""]

    for ev in gt:
        eid = ev["event_id"]
        stored = ev["gt_onset_date"]
        year = int(stored[:4])
        print(f"\n### {eid}")
        print(f"  stored onset: {stored}  (status: {ev.get('onset_status')})")
        print(f"  stored evidence: {(ev.get('onset_evidence') or '')[:150]}")
        report_lines += [f"## {eid}", "",
                         f"- stored onset: **{stored}**",
                         f"- stored status: `{ev.get('onset_status')}`",
                         f"- stored evidence: {(ev.get('onset_evidence') or '').strip()}",
                         ""]

        candidates, kws = ARTICLE_HINTS.get(eid, ([], []))
        if not candidates:
            candidates = search_articles(ev.get("event_name", eid), args.proxy)
        seen_titles: set[str] = set()
        found_any = False

        for title in candidates:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            wt = plain_text(title, args.proxy)
            if not wt:
                print(f"  [miss] {title}: no content")
                continue
            text = clean(wt)
            hits = sentences_with_dates(text, year, kws)
            if not hits:
                continue
            found_any = True
            print(f"  [{title}] {len(hits)} date-bearing sentences in {year}")
            report_lines += [f"### {title}", ""]
            for ds, sent in hits[:14]:
                flag = "  <<< MATCHES STORED" if stored in sent else ""
                print(f"      {ds:38s} {sent[:110]}{flag}")
                report_lines.append(f"- `{ds}` — {sent}{flag}")
            report_lines.append("")

        if not found_any:
            print("  [no dated context sentences found]")
            report_lines += ["_No dated context sentences found._", ""]

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\nreport -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
