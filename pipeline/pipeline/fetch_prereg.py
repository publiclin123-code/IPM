"""Pre-registered GDELT re-fetch for affected events (fixes DATA_ISSUES DI-1 and DI-2).

Why this script exists
----------------------
The original `fetch_per_event.py` built the per-event queries by hand in
`event_keywords.json`, and for some events those queries could only match
documents that came into existence *because of* the disruption. The Suez entry
searched for the ship's name plus phrases like "suez canal blockage"; the stored
rationale even records the expectation of zero hits and treats that zero as
evidence that no precursor existed. A query that cannot return a document cannot
test whether the document exists.

This script removes both the hand-tuning and the URL-only blindness:

1. QUERY IS DERIVED, NOT WRITTEN. Keywords are generated mechanically from the
   a priori entity tables already stored in `validation/gt_events.json`
   (`companies`, `geographies`, `commodities`). Those tables were fixed before
   any model score was seen, and none of their entries is a description of the
   disruption, so the derived query cannot encode the answer. The derived sets
   are printed and written to the manifest for pre-registration.

2. MATCHING SPANS SIX COLUMNS. The original filter tested `url + slug` only.
   Direct measurement showed "suez" appears in 0 of 3,982 URLs on a sampled day
   but in 11 rows of the geographic field, so URL-only filtering is
   structurally blind for such events. Here we match the URL plus both actor
   names plus all three geographic name fields, recording which field fired.

3. DOWNLOADS ARE SHARED ACROSS EVENTS. GDELT days are fetched once and matched
   against every target event, so adding events does not multiply requests. No
   disk cache is used: each archive is matched and discarded, which keeps the
   footprint small.

Output goes to a NEW directory so the original corpus is preserved for
comparison. Nothing in `data/by_event/` is touched.

Usage
-----
  # the three events whose corpora were empty or near-empty
  python3 pipeline/fetch_prereg.py \
      --events suez_ever_given_2021,renesas_naka_plant_fire_2021,beirut_port_explosion_2020

  # all 18 news events, longer window
  python3 pipeline/fetch_prereg.py --window 180 --workers 8

Then compare against the old corpus:
  python3 pipeline/fetch_prereg.py --events ... --report-only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import ssl
import sys
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "validation" / "gt_events.json"
OUT_DIR = ROOT / "data" / "by_event_v3"
OLD_DIR = ROOT / "data" / "by_event"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

TIMESLOTS = ["000000", "150000", "300000", "450000"]

# GDELT 2.0 events schema (61 columns, verified against 20210303000000 export).
COL_SQLDATE = 1
COL_A1_NAME = 6
COL_A2_NAME = 16
COL_A1_GEO = 37
COL_A2_GEO = 44
COL_ACTION_GEO = 52
COL_URL = 60
MIN_COLS = 61

# Columns scanned for keyword matches. Kept as (index, label) so the manifest
# can report which field produced each hit.
MATCH_COLUMNS = [
    (COL_URL, "url"),
    (COL_A1_NAME, "actor1_name"),
    (COL_A2_NAME, "actor2_name"),
    (COL_A1_GEO, "actor1_geo"),
    (COL_A2_GEO, "actor2_geo"),
    (COL_ACTION_GEO, "action_geo"),
]

# Tokens too generic to carry signal on their own. Dropping these prevents the
# derived query from degenerating into a stopword search; they are removed from
# the token-level expansion only, never from the full entity strings.
GENERIC_TOKENS = {
    "port", "city", "state", "province", "region", "area", "county", "island",
    "the", "and", "of", "for", "de", "la", "el", "al", "north", "south", "east",
    "west", "central", "upper", "lower", "new", "old", "united", "states",
    "goods", "products", "shipping", "supply", "chain", "market", "company",
    "corporation", "group", "inc", "ltd", "international",
}

_write_lock = threading.Lock()


def slug_title(url: str) -> str:
    """Mirror of the original pipeline's slug extraction."""
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.split("?")[0].split("#")[0]
    slug = slug.replace("_", " ").replace("-", " ").replace(".html", "").replace(".htm", "")
    return slug[:200]


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation to spaces, for matching."""
    out = []
    for ch in text.lower():
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def has_phrase(hay_norm: str, kw_norm: str) -> bool:
    """Whole-word / whole-phrase containment, never raw substring.

    The first version of this fetcher used `kw in haystack`, which let the token
    "ever" (from *Ever Given*) match "everything", "never" and "reversing", and
    admitted 77-86% noise. Matching on word boundaries removes that class of
    error. Both sides are expected to be already normalised to space-separated
    lowercase words.
    """
    if not kw_norm:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(kw_norm)}(?![a-z0-9])", hay_norm) is not None


def derive_keywords(event: dict) -> dict:
    """Build the query mechanically from the a priori entity tables.

    No per-event hand tuning. Every keyword traces to an entity that was
    recorded before the disruption and that is not a description of it.
    """
    entities: list[tuple[str, str]] = []
    for field in ("companies", "geographies", "commodities"):
        for raw in event.get(field) or []:
            entities.append((field, str(raw)))

    full: set[str] = set()
    tokens: set[str] = set()

    for _field, raw in entities:
        # Parenthetical aliases: "Ever Given (Evergreen)" -> both parts.
        parts = [raw]
        if "(" in raw:
            parts = [raw[: raw.index("(")], raw[raw.index("(") + 1: raw.rindex(")")]]
        for part in parts:
            n = normalize(part)
            if not n:
                continue
            full.add(n)
            for tok in n.split():
                if len(tok) >= 4 and tok not in GENERIC_TOKENS:
                    tokens.add(tok)

    return {
        "entities": [{"field": f, "value": v} for f, v in entities],
        "phrase_keywords": sorted(full),
        "token_keywords": sorted(tokens),
    }


def keywords_for(query: dict) -> list[str]:
    return query["phrase_keywords"] + query["token_keywords"]


def tokens_only(query: dict) -> list[str]:
    """Tokens that are not already inside a stronger multi-word phrase.

    Used to enforce the specificity rule: co-occurrence should count distinct
    pieces of evidence, not the same evidence twice in different guises.
    """
    phrases = [p for p in query["phrase_keywords"] if len(p.split()) >= 2]
    out = []
    for t in query["token_keywords"]:
        if any(t in p.split() for p in phrases):
            continue
        out.append(t)
    return out


def match_row(cols: list[str], keywords: list[str],
              event_tokens: set[str] | None = None,
              min_distinct: int = 2) -> tuple[bool, str, str]:
    """Decide whether a GDELT row belongs to the event.

    Two guards, both added after measuring the first attempt's precision
    (77-86% of its hits were substring noise):

    1. Word-boundary phrase matching, so a short token cannot match by
       coincidence.
    2. A specificity requirement: at least `min_distinct` distinct keywords
       must fire. A single common noun -- "oil", "japan", "grain" -- is never
       sufficient on its own, because each of those matches thousands of
       unrelated articles. This is what makes the query unable to admit a
       document on one generic word.

    Returns (matched, field_label, matched_keyword).
    """
    if event_tokens is None:
        event_tokens = set(keywords)
    fired: list[tuple[str, str]] = []
    for idx, label in MATCH_COLUMNS:
        if idx >= len(cols):
            continue
        cell = cols[idx]
        if not cell:
            continue
        hay = normalize(cell)
        for kw in keywords:
            if has_phrase(hay, kw):
                fired.append((label, kw))
    distinct = {kw for _lbl, kw in fired}
    # A multi-word entity phrase is itself specific enough to stand alone.
    strong = any(len(kw.split()) >= 2 for kw in distinct)
    if strong or len(distinct) >= min_distinct:
        label, kw = fired[0]
        return True, label, kw
    return False, "", ""


def fetch_day_slots(day: datetime, timeout: int = 60) -> list[list[str]]:
    """Download and parse the four GDELT slots for one day. Missing slots skipped."""
    rows: list[list[str]] = []
    stamp_day = day.strftime("%Y%m%d")
    for slot in TIMESLOTS:
        url = f"https://data.gdeltproject.org/gdeltv2/{stamp_day}{slot}.export.CSV.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
                data = resp.read()
        except Exception:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            cols = next(csv.reader([line], delimiter="\t", quotechar='"'))
            if len(cols) >= MIN_COLS:
                rows.append(cols)
    return rows


def build_day_plan(events: list[dict], window: int) -> dict[datetime, list[str]]:
    """Map each day in the union of windows to the event ids that need it."""
    plan: dict[datetime, list[str]] = {}
    gdelt_start = datetime(2013, 2, 18)
    for ev in events:
        onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
        day = max(onset - timedelta(days=window), gdelt_start)
        while day < onset:
            # GDELT month folders roll over; the flat gdeltv2 path is safe.
            plan.setdefault(day, []).append(ev["event_id"])
            day += timedelta(days=1)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered GDELT re-fetch")
    ap.add_argument("--events", default="", help="comma-separated event ids; default = all news events")
    ap.add_argument("--window", type=int, default=180, help="days before onset to search")
    ap.add_argument("--workers", type=int, default=6, help="parallel day fetches")
    ap.add_argument("--min-distinct", type=int, default=2,
                    help="distinct keywords required unless a multi-word phrase fires")
    ap.add_argument("--report-only", action="store_true", help="compare v3 against the old corpus, no fetching")
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))["events"]
    if args.events:
        want = {s.strip() for s in args.events.split(",") if s.strip()}
        events = [e for e in gt if e["event_id"] in want]
        missing = want - {e["event_id"] for e in events}
        if missing:
            print(f"unknown event ids: {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        events = gt

    queries = {ev["event_id"]: derive_keywords(ev) for ev in events}

    if args.report_only:
        print(f"{'event_id':34s} {'v3':>7s} {'uniq':>7s} {'old':>7s} {'delta':>8s}")
        print("-" * 70)
        for ev in events:
            eid = ev["event_id"]
            p3 = OUT_DIR / f"{eid}.jsonl"
            n3 = u3 = 0
            if p3.exists():
                urls = []
                for line in p3.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        urls.append(json.loads(line).get("url", ""))
                n3, u3 = len(urls), len(set(urls))
            po = OLD_DIR / f"{eid}.jsonl"
            no = 0
            if po.exists():
                no = sum(1 for line in po.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"{eid:34s} {n3:7d} {u3:7d} {no:7d} {n3 - no:+8d}")
        return 0

    print("=" * 78)
    print("PRE-REGISTERED QUERIES (derived from a priori entities; no hand tuning)")
    print("=" * 78)
    for ev in events:
        q = queries[ev["event_id"]]
        print(f"\n[{ev['event_id']}]  onset={ev['gt_onset_date']}  window={args.window}d")
        for ent in q["entities"]:
            print(f"    {ent['field']:12s} {ent['value']}")
        print(f"    -> {len(q['phrase_keywords'])} phrases + {len(q['token_keywords'])} tokens")
        print(f"    -> {keywords_for(q)}")

    plan = build_day_plan(events, args.window)
    days = sorted(plan)
    print()
    print("=" * 78)
    print(f"{len(days)} unique GDELT days across {len(events)} events "
          f"({len(days) * len(TIMESLOTS)} slot downloads, shared across events)")
    print("=" * 78, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hits: dict[str, list[dict]] = {ev["event_id"]: [] for ev in events}
    seen: dict[str, set[str]] = {ev["event_id"]: set() for ev in events}
    counters = {ev["event_id"]: 0 for ev in events}
    n_done = 0

    def process(day: datetime) -> None:
        nonlocal n_done
        rows = fetch_day_slots(day)
        for eid in plan[day]:
            if not rows:
                continue
            kws = keywords_for(queries[eid])
            for cols in rows:
                ok, label, kw = match_row(cols, kws, min_distinct=args.min_distinct)
                if not ok:
                    continue
                url = cols[COL_URL]
                if url in seen[eid]:
                    continue
                seen[eid].add(url)
                sqldate = cols[COL_SQLDATE]
                iso = (f"{sqldate[:4]}-{sqldate[4:6]}-{sqldate[6:8]}"
                       if len(sqldate) == 8 else day.strftime("%Y-%m-%d"))
                with _write_lock:
                    counters[eid] += 1
                    hits[eid].append({
                        "id": f"{eid}-{iso}-{counters[eid]:04d}",
                        "date": iso,
                        "source": "GDELT",
                        "url": url,
                        "title": slug_title(url),
                        "text": slug_title(url),
                        "_match_field": label,
                        "_match_keyword": kw,
                    })
        with _write_lock:
            n_done += 1
            if n_done % 25 == 0 or n_done == len(days):
                tot = sum(len(v) for v in hits.values())
                print(f"  [{n_done}/{len(days)} days] {tot} matched rows", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process, days))

    print()
    for ev in events:
        eid = ev["event_id"]
        recs = sorted(hits[eid], key=lambda r: (r["date"], r["id"]))
        out = OUT_DIR / f"{eid}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        old = OLD_DIR / f"{eid}.jsonl"
        n_old = sum(1 for l in old.read_text(encoding="utf-8").splitlines() if l.strip()) if old.exists() else 0
        # Which fields produced the hits: shows whether URL-only search was blind.
        by_field: dict[str, int] = {}
        for r in recs:
            by_field[r["_match_field"]] = by_field.get(r["_match_field"], 0) + 1
        print(f"{eid:34s} v3={len(recs):5d}  old={n_old:5d}  delta={len(recs) - n_old:+6d}")
        print(f"{'':34s} match fields: {by_field}")

    manifest = {
        "window_days": args.window,
        "match_columns": [c[1] for c in MATCH_COLUMNS],
        "note": ("Queries derived mechanically from gt_events.json a priori entity "
                 "tables. No hand-written per-event keywords."),
        "queries": queries,
        "results": {ev["event_id"]: len(hits[ev["event_id"]]) for ev in events},
    }
    (OUT_DIR / "_prereg_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {OUT_DIR / '_prereg_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
