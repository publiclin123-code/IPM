"""Sharper probe: does IPM publish papers whose contribution is an evaluation finding?

The previous probe (recon_ipm_prompt.py) counted titles containing topic words
such as "temporal reasoning" and "reliability". Reading the returned titles
showed that most of them are *method* papers that merely contain those words --
"Cross-modal event extraction via Visual Event Grounding", "A graph propagation
model with rich event structures". Counting them as evidence that a venue
publishes evaluation work would repeat the volume-artifact error in a subtler
form: topical word matches are not genre matches.

This script probes *genre* instead, using markers that appear in titles when the
contribution is an evaluative finding rather than a new method:

    "empirical study/evaluation", "systematic evaluation", "comparative
    evaluation", "reassessment", "pitfalls", "limitations", "reproducibility",
    "validity", "how well", "quality of", "reliability of", "evidence from"

It prints the titles, because the whole point is to judge them by reading.

Usage
-----
  python3 pipeline/recon_genre.py --journal IPM --rows 8
  python3 pipeline/recon_genre.py --journal ESWA --rows 8
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAILTO = "contact@example.invalid"
FROM = "2022-01-01"

JOURNALS = {
    "IPM": "0306-4573",
    "JASIST": "2330-1635",
    "JIS": "0165-5515",
    "ESWA": "0957-4174",
    "TOIS": "1046-8188",
}

GENRE_PROBES = [
    ("empirical_study", "empirical study"),
    ("empirical_eval", "empirical evaluation"),
    ("systematic_eval", "systematic evaluation"),
    ("comparative_eval", "comparative evaluation"),
    ("reassessment", "reassessment"),
    ("pitfalls", "pitfalls"),
    ("limitations_of", "limitations of"),
    ("reproducibility", "reproducibility"),
    ("validity_of", "validity of"),
    ("how_well", "how well"),
    ("quality_of", "quality of"),
    ("reliability_of", "reliability of"),
    ("evidence_from", "evidence from"),
    ("do_llms", "do large language models"),
    ("are_llms", "are large language models"),
    ("can_llms", "can large language models"),
]


def cr(params: dict, tries: int = 3) -> dict | None:
    p = dict(params)
    p["mailto"] = MAILTO
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(p)
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"recon/1.0 (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())["message"]
        except Exception:
            time.sleep(1.5 + a * 2)
    return None


def total(issn: str) -> int:
    m = cr({"filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
            "rows": "0", "select": "DOI"})
    return int(m.get("total-results", 0)) if m else 0


def count(issn: str, phrase: str) -> int:
    m = cr({"query.title": phrase,
            "filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
            "rows": "0", "select": "DOI"})
    return int(m.get("total-results", 0)) if m else 0


def titles(issn: str, phrase: str, rows: int) -> list[tuple]:
    m = cr({"query.title": phrase,
            "filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
            "rows": str(rows), "select": "title,issued,DOI"})
    out = []
    for it in (m or {}).get("items", []):
        t = " ".join((it.get("title") or ["?"])[0].split())
        y = (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
        out.append((y, t[:120], it.get("DOI")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="IPM")
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()
    issn = JOURNALS[args.journal]
    n = total(issn)
    print(f"{args.journal} (issn {issn})   2022+ articles: {n}")
    print("=" * 108)
    print(f"{'genre probe':22s} {'count':>6s} {'per-mille':>10s}")
    print("-" * 108)
    for label, phrase in GENRE_PROBES:
        c = count(issn, phrase)
        time.sleep(0.4)
        print(f"{label:22s} {c:6d} {c / n * 1000:10.1f}")
    print()
    print("=" * 108)
    print("TITLES (read these: is the contribution an evaluation finding?)")
    print("=" * 108)
    for label, phrase in GENRE_PROBES:
        ts = titles(issn, phrase, args.rows)
        print(f"\n  [{label}]")
        for y, t, doi in ts:
            print(f"    [{y}] {t}")
            print(f"           doi:{doi}")
        time.sleep(0.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
