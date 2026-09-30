"""Targeted recon: does IPM publish this paper's artifact type?

The user's concern is specific and correct: the manuscript's core is a schema
change, i.e. a prompt change. If a venue reads it as "we edited a prompt", it is
rejected regardless of how the density tables look.

So this script does not count topic words. It looks for the *artifact types*
that would make the work legible to an Information Science audience, and it
prints real titles so the judgement is made on evidence rather than on numbers.

Probes, in descending order of how directly they bear on the manuscript:

  A. Is prompt/schema design treated as a research object? If a venue publishes
     "does schema X change extraction quality", the paper is legible there.
  B. Is LLM output validity / reliability / error taxonomy a recognised topic?
  C. Is event extraction evaluation published (not just event extraction)?
  D. Is there work on temporal/event relations in LLM extraction?
  E. Is hallucination/faithfulness of extraction studied?

Output is real titles with years and DOIs, for IPM and for a small set of
comparators, so the comparison is like-for-like.

Usage
-----
  python3 pipeline/recon_ipm_prompt.py
  python3 pipeline/recon_ipm_prompt.py --rows 8
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
}

# (label, title phrase). Chosen to test artifact type, not domain.
PROBES = [
    ("prompt_sensitivity", "prompt sensitivity"),
    ("prompt_design", "prompt design"),
    ("schema_design", "schema"),
    ("llm_annotation", "large language model annotation"),
    ("llm_reliability", "reliability large language model"),
    ("llm_evaluation", "evaluation large language model"),
    ("event_eval", "event extraction evaluation"),
    ("temporal_reasoning", "temporal reasoning"),
    ("temporal_relation", "temporal relation"),
    ("faithfulness", "faithfulness"),
    ("hallucination", "hallucination"),
    ("zero_shot_extract", "zero-shot extraction"),
    ("information_extraction_llm", "information extraction large language model"),
    ("agreement", "inter-annotator agreement"),
]


def cr(params: dict, tries: int = 3) -> dict | None:
    p = dict(params)
    p["mailto"] = MAILTO
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(p)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"recon/1.0 (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())["message"]
        except Exception:
            time.sleep(1.5 + attempt * 2)
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
            "rows": str(rows), "select": "title,issued,DOI,container-title"})
    out = []
    for it in (m or {}).get("items", []):
        t = " ".join((it.get("title") or ["?"])[0].split())
        y = (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
        out.append((y, t[:118], it.get("DOI")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--only", default="", help="restrict to one journal key")
    args = ap.parse_args()

    js = {k: v for k, v in JOURNALS.items() if not args.only or k == args.only}

    totals = {}
    print(f"Crossref title search from {FROM}\n")
    for k, issn in js.items():
        totals[k] = total(issn)
        time.sleep(0.4)
    print(f"{'probe':26s}" + "".join(f"{k:>9s}" for k in js))
    print(f"{'n_articles':26s}" + "".join(f"{totals[k]:9d}" for k in js))
    print("-" * (26 + 9 * len(js)))
    for label, phrase in PROBES:
        line = f"{label:26s}"
        for k, issn in js.items():
            c = count(issn, phrase)
            time.sleep(0.4)
            line += f"{c:9d}"
        print(line, flush=True)

    print()
    print("=" * 110)
    print("REAL TITLES - is each artifact type actually published?")
    print("=" * 110)
    key_probes = ["prompt_sensitivity", "schema_design", "llm_reliability",
                  "llm_evaluation", "event_eval", "temporal_reasoning", "faithfulness"]
    for k, issn in js.items():
        print(f"\n{'#' * 110}\n### {k}\n{'#' * 110}")
        for label, phrase in PROBES:
            if label not in key_probes:
                continue
            ts = titles(issn, phrase, args.rows)
            print(f"\n  [{label}]  ({len(ts)} shown)")
            for y, t, doi in ts:
                print(f"    [{y}] {t}")
                print(f"           doi:{doi}")
            time.sleep(0.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
