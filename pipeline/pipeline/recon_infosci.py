"""Crossref recon for information-science / information-management venues.

Purpose: the manuscript's product is the measurement validity of LLM extraction
output, plus one measured configuration fix. That is not a system contribution,
so the venue question is which journals publish measurement-validity work in this
area.

Two things this script does deliberately, both learned from an earlier mistake:

1. Normalises by journal output. Raw Crossref title counts track journal size,
   not topical fit. An earlier session published a "density" table that was
   pure volume artifact and had to be retracted. Every count below is divided by
   the journal's 2022+ output and reported in per-mille.

2. Splits the "信息类" category in two. Information *science* venues publish
   retrieval, text mining and evaluation methods; information *systems* venues
   publish behavioural, organisational and design-science work. The manuscript
   fits the first group only. Mixing them would repeat the earlier error of
   treating scope similarity as substitutable across sub-fields.

Every number is reproducible with `python3 pipeline/recon_infosci.py`.

Usage
-----
  python3 pipeline/recon_infosci.py
  python3 pipeline/recon_infosci.py --rows 5
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

# (label, ISSN, group). group: sci = information science / IR,
# sys = information systems / management, ctrl = non-IS comparators.
JOURNALS = [
    ("IPM  Information Processing & Management", "0306-4573", "sci"),
    ("JASIST  J. Assoc. Inf. Sci. Technol.", "2330-1635", "sci"),
    ("JIS  Journal of Information Science", "0165-5515", "sci"),
    ("OIR  Online Information Review", "1468-4527", "sci"),
    ("Aslib J. Information Management", "2050-3806", "sci"),
    ("LISR  Library & Inf. Science Research", "0740-8188", "sci"),
    ("Scientometrics", "0138-9130", "sci"),
    ("Information & Management", "0378-7206", "sys"),
    ("IJIM  Int. J. Information Management", "0268-4012", "sys"),
    ("Inf. Systems Frontiers", "1387-3326", "sys"),
    ("Decision Support Systems", "0167-9236", "sys"),
    ("Inf. Technology & People", "0959-3845", "sys"),
    ("Telematics and Informatics", "0736-5853", "sys"),
    ("ESWA  Expert Systems w/ Applications", "0957-4174", "ctrl"),
    ("ACM TOIS  Trans. Inf. Systems", "1046-8188", "sci"),
]

# Topic probes. "validity" is the one that matters most: it asks whether these
# venues publish measurement-validation work, which is what this paper is.
QUERIES = {
    "supply_chain": "supply chain",
    "LLM": "large language model",
    "early_warning": "early warning",
    "event_extraction": "event extraction",
    "validity_eval": "validity evaluation",
    "benchmark": "benchmark",
    "nlp_text": "text mining",
}


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
            time.sleep(2 + attempt * 2)
    return None


def total_for(issn: str) -> int:
    m = cr({"filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
            "rows": "0", "select": "DOI"})
    return int(m.get("total-results", 0)) if m else 0


def count_title(issn: str, phrase: str) -> int:
    m = cr({"query.title": phrase,
            "filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
            "rows": "0", "select": "DOI"})
    return int(m.get("total-results", 0)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Information-science venue recon")
    ap.add_argument("--rows", type=int, default=0, help="also print sample titles")
    args = ap.parse_args()

    rows = []
    print(f"Crossref, title search, from {FROM}. Counts normalised per 1000 "
          f"journal articles.\n")
    hdr = f"{'journal':42s} {'group':6s} {'n_arts':>7s}"
    for k in QUERIES:
        hdr += f" {k[:11]:>11s}"
    print(hdr)
    print("-" * len(hdr), flush=True)

    for label, issn, group in JOURNALS:
        tot = total_for(issn)
        time.sleep(0.4)
        if not tot:
            print(f"{label:42s} {group:6s}  (no data)")
            continue
        counts = {}
        for k, phrase in QUERIES.items():
            counts[k] = count_title(issn, phrase)
            time.sleep(0.4)
        rows.append((label, group, tot, counts))
        line = f"{label:42s} {group:6s} {tot:7d}"
        for k in QUERIES:
            pm = counts[k] / tot * 1000
            line += f" {pm:11.1f}"
        print(line, flush=True)

    print()
    print("Legend: values are per-mille of that journal's 2022+ output.")
    print("  group sci = information science / IR   (publishes evaluation methods)")
    print("  group sys = information systems        (behavioural / design science)")
    print("  group ctrl = non-IS comparator")
    print()
    print("=" * 100)
    print("FOCUS: does the venue publish measurement-validity work AND LLM text work?")
    print("=" * 100)
    ranked = []
    for label, group, tot, c in rows:
        score = c["LLM"] / tot + c["validity_eval"] / tot + c["event_extraction"] / tot + c["nlp_text"] / tot
        ranked.append((score * 1000, label, group, tot, c))
    print(f"{'journal':42s} {'grp':4s} {'LLM‰':>7s} {'valid‰':>8s} "
          f"{'evtext‰':>8s} {'textmin‰':>9s}")
    print("-" * 100)
    for score, label, group, tot, c in sorted(ranked, reverse=True):
        print(f"{label:42s} {group:4s} {c['LLM'] / tot * 1000:7.1f} "
              f"{c['validity_eval'] / tot * 1000:8.1f} "
              f"{c['event_extraction'] / tot * 1000:8.1f} "
              f"{c['nlp_text'] / tot * 1000:9.1f}")

    if args.rows:
        print()
        print("=" * 100)
        print("Sample titles per probe (for the top-ranked sci venue)")
        print("=" * 100)
        for label, issn, group in JOURNALS:
            if group != "sci":
                continue
            print(f"\n--- {label} ---")
            for k in ("LLM", "validity_eval", "event_extraction"):
                m = cr({"query.title": QUERIES[k],
                        "filter": f"issn:{issn},from-pub-date:{FROM},type:journal-article",
                        "rows": str(args.rows),
                        "select": "title,issued,DOI"})
                items = (m or {}).get("items", [])
                print(f"  [{k}]")
                for it in items:
                    t = " ".join((it.get("title") or ["?"])[0].split())[:88]
                    y = (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
                    print(f"     [{y}] {t}")
                time.sleep(0.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
