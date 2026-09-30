"""Identity-aware entity matching (DATA_ISSUES.md DI-8).

The problem
-----------
`validation/validate.py` credits a hit when a company matches, or, in loose mode,
when a geography **or** a commodity matches, using bi-directional substring
comparison. Two failure modes follow:

  - "christopher steele" matches the commodity "steel" (substring)
  - a US steel-tariff article matches the event "Toyota engine plant steel
    explosion" because both list the commodity "steel"

Measured: 69 of 207 true positives (33.3%) have no company-level evidence, and
the Toyota event has 26 of 27. See `pipeline/audit_match_evidence.py`.

The idea
--------
An event's identity lives in a particular entity dimension, and which dimension
depends on the kind of event. "US Section 301 tariffs on China" is identified by
*places* -- a US-China trade-war signal is on-topic even if it concerns pork
rather than steel. "Renesas Naka plant fire" is identified by a *firm* -- a
generic steel-tariff signal is off-topic even though it contains the word steel.

So the match rule should be selected by the identity axis, and the axis is read
off the event name, which was fixed before any model output was seen.

    identity=company    require a company match
    identity=geo_multi  require >=2 distinct geographies, or 1 geography + 1 commodity
    identity=geo_one    require 1 geography + 1 commodity

All matching is word-boundary, not substring.

Two honesty requirements
------------------------
1. The axis assignment must be justified by the event name alone, and the
   justification is printed so it can be checked.
2. Because the axis changes the headline, every rule is reported side by side.
   A measurement paper cannot present one matcher as "the" matcher; the
   sensitivity to the matching rule is itself a result.

Usage
-----
  python3 pipeline/identity_matcher.py
  python3 pipeline/identity_matcher.py --variant body
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORWARD = {"forward_looking", "latent"}
WINDOW = 180

# Identity axis, assigned from the event NAME (a priori) and justified in the
# comment. "Where does this event's name get its meaning from?"
IDENTITY: dict[str, tuple[str, str]] = {
    # --- firm-identified: the name centres on a company or institution ---
    "toyota_steel_explosion_2019": ("company", "name centres on Toyota"),
    "renesas_naka_plant_fire_2021": ("company", "name centres on Renesas"),
    "uaw_auto_strike_2023": ("company", "name centres on the UAW and the Big Three"),
    # --- place-identified, single place: name centres on one location ---
    "covid_supply_disruption_2020": ("geo_one", "name centres on Wuhan/Hubei"),
    "india_wheat_export_ban_2022": ("geo_one", "name centres on India"),
    "taiwan_strait_crisis_2022": ("geo_one", "name centres on the Taiwan Strait"),
    "hurricane_maria_2017": ("geo_one", "name centres on Puerto Rico"),
    "beirut_port_explosion_2020": ("geo_one", "name centres on Beirut"),
    "suez_ever_given_2021": ("geo_one", "name centres on the Suez Canal"),
    "port_los_angeles_backlog_2021": ("geo_one", "name centres on LA/Long Beach"),
    "egg_shortage_birdflu_2025": ("geo_one", "name centres on the US and a commodity"),
    # --- place-identified, two or more places: the pairing IS the event ---
    "us_china_tariff_war_2018": ("geo_multi", "name pairs United States and China"),
    "russia_ukraine_war_2022": ("geo_multi", "name pairs Russia and Ukraine"),
    "red_sea_crisis_2023": ("geo_multi", "name pairs the Red Sea and Yemen"),
    "us_chip_export_controls_2022": ("geo_multi", "name pairs the US and China"),
    "europe_energy_crisis_2022": ("geo_multi", "name pairs Europe and Russia"),
    "black_sea_grain_exit_2023": ("geo_multi", "name pairs the Black Sea and Ukraine"),
    # Renesas earthquake: name is the earthquake, but the supply-chain entity of
    # interest is the firm. Tested both ways below; declared as company.
    "renesas_earthquake_2016": ("company", "name is a quake; the supply-chain entity is Renesas"),
}


def norm(s: str) -> str:
    return (s or "").strip().lower()


def words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", norm(s)) if w]


def _acronym_matches(short: str, long_phrase: str) -> bool:
    """True when `short` is the initialism of `long_phrase`.

    Needed because word-boundary matching alone breaks abbreviations: the
    ground truth lists the event's actor as "UAW" while the extractor correctly
    emits "United Auto Workers", and requiring a consecutive-word match rejects
    that as a mismatch. The same trap applies to General Motors/GM and
    European Union/EU.

    Guards: the short side must be alphabetic and 2-6 characters, the long side
    must have at least as many words as the short side has letters, and the
    initials must correspond exactly. This keeps false positives down -- an
    arbitrary short word will not match a long phrase by accident.
    """
    s = norm(short)
    if not (2 <= len(s) <= 6 and s.isalpha()):
        return False
    lw = words(long_phrase)
    if len(lw) < len(s):
        return False
    return "".join(w[0] for w in lw) == s


def match_word(sig_vals, gt_vals) -> int:
    """Number of gt values matched by some sig value, word-boundary.

    Whole-phrase or consecutive-word containment, never raw substring, so
    "steele" cannot match "steel" and "ever" cannot match "everything".
    Acronym/expansion pairs are matched separately by `_acronym_matches`.
    """
    sv = [norm(v) for v in (sig_vals or []) if v]
    gv = [norm(v) for v in (gt_vals or []) if v]
    n = 0
    for b in gv:
        bw = words(b)
        if not bw:
            continue
        for a in sv:
            aw = words(a)
            if not aw:
                continue
            if aw == bw:
                n += 1
                break
            if _acronym_matches(a, b) or _acronym_matches(b, a):
                n += 1
                break
            hit = False
            for (x, y) in ((aw, bw), (bw, aw)):
                if len(x) >= len(y):
                    for i in range(len(x) - len(y) + 1):
                        if x[i:i + len(y)] == y:
                            hit = True
                            break
                if hit:
                    break
            if hit:
                n += 1
                break
    return n


def match_sub(sig_vals, gt_vals) -> int:
    """The current production rule: bi-directional substring, plus acronyms.

    Acronyms are included so that R0 and R4 differ only in the dimensions they
    accept, not in whether they understand abbreviations. Without this, R0's
    known substring weakness would be confounded with a capability gap.
    """
    sv = [norm(v) for v in (sig_vals or []) if v]
    gv = [norm(v) for v in (gt_vals or []) if v]
    n = 0
    for b in gv:
        for a in sv:
            if a == b or a in b or b in a or _acronym_matches(a, b) or _acronym_matches(b, a):
                n += 1
                break
    return n


def comps(sig: dict) -> list[str]:
    out = []
    for c in sig.get("companies", []) or []:
        out.append(c.get("name", "") if isinstance(c, dict) else str(c))
    return [x for x in out if x]


def rule_match(sig: dict, ev: dict, rule: str) -> bool:
    axis = IDENTITY.get(ev["event_id"], ("geo_multi", ""))[0]
    gc, gg, gm = ev.get("companies", []), ev.get("geographies", []), ev.get("commodities", [])

    if rule == "R0_loose_substring":
        return bool(match_sub(comps(sig), gc) or match_sub(sig.get("geographies", []), gg)
                    or match_sub(sig.get("commodities", []), gm))
    if rule == "R1_loose_word":
        return bool(match_word(comps(sig), gc) or match_word(sig.get("geographies", []), gg)
                    or match_word(sig.get("commodities", []), gm))
    if rule == "R2_two_dimension":
        return bool(match_word(comps(sig), gc)
                    or (match_word(sig.get("geographies", []), gg)
                        and match_word(sig.get("commodities", []), gm)))
    if rule == "R3_company_only":
        return bool(match_word(comps(sig), gc))
    if rule == "R4_identity":            # word-boundary, axis-selected
        nc = match_word(comps(sig), gc)
        ng = match_word(sig.get("geographies", []), gg)
        nm = match_word(sig.get("commodities", []), gm)
        if nc:
            return True
        if axis == "company":
            return False
        if axis == "geo_multi":
            return ng >= 2 or (ng >= 1 and nm >= 1)
        return ng >= 1 and nm >= 1       # geo_one
    raise ValueError(rule)


RULES = ["R0_loose_substring", "R1_loose_word", "R2_two_dimension",
         "R3_company_only", "R4_identity"]


def parse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def load(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        if s.get("status") in ("empty", "error") or not s.get("temporality"):
            continue
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="title", choices=["title", "body"])
    ap.add_argument("--signals-dir", default="",
                    help="override the extraction directory to score (e.g. results/q38_slug_v1)")
    args = ap.parse_args()
    if args.signals_dir:
        sdir = Path(args.signals_dir)
        if not sdir.is_absolute():
            sdir = ROOT / sdir
    else:
        sdir = ROOT / ("results/v2" if args.variant == "title" else "results/v2_body")
    gt = json.loads((ROOT / "validation" / "gt_events.json").read_text(encoding="utf-8"))["events"]

    print("=" * 112)
    print("IDENTITY AXIS (assigned from the event name, before any model output)")
    print("=" * 112)
    print(f"{'event':34s} {'axis':10s} justification")
    print("-" * 112)
    for ev in gt:
        ax, why = IDENTITY.get(ev["event_id"], ("geo_multi", "unassigned"))
        print(f"{ev['event_id']:34s} {ax:10s} {why}")

    agg = {r: Counter() for r in RULES}
    per_event: dict[str, dict[str, int]] = defaultdict(dict)
    fwd_total = 0

    for ev in gt:
        eid = ev["event_id"]
        p = sdir / f"{eid}_signals.jsonl"
        if not p.exists():
            continue
        onset = parse(ev["gt_onset_date"])
        fwd = []
        for s in load(p):
            if s.get("temporality") not in FORWARD or not s.get("signal_date"):
                continue
            try:
                lead = (onset - parse(s["signal_date"])).days
            except Exception:
                continue
            if 0 < lead <= WINDOW:
                fwd.append(s)
        fwd_total += len(fwd)
        for r in RULES:
            n = sum(1 for s in fwd if rule_match(s, ev, r))
            agg[r]["tp"] += n
            agg[r]["hits"] += 1 if n else 0
            per_event[eid][r] = n

    print()
    print("=" * 112)
    print(f"RESULT   variant={args.variant}   forward signals (pre-onset) = {fwd_total}")
    print("=" * 112)
    n_events = len(gt)
    print(f"{'rule':22s} {'TP':>6s} {'precision':>10s} {'events hit':>11s}")
    print("-" * 112)
    for r in RULES:
        a = agg[r]
        prec = a["tp"] / fwd_total if fwd_total else 0
        print(f"{r:22s} {a['tp']:6d} {prec:10.4f} {a['hits']:8d}/{n_events}")

    print()
    print("per-event TP under each rule")
    print("-" * 112)
    print(f"{'event':34s} " + "".join(f"{r.split('_')[0]:>8s}" for r in RULES))
    for ev in gt:
        eid = ev["event_id"]
        if eid not in per_event:
            continue
        print(f"{eid:34s} " + "".join(f"{per_event[eid][r]:8d}" for r in RULES))

    print()
    print("=" * 112)
    print("DELTA: what the current rule credits that the identity rule does not")
    print("=" * 112)
    for ev in gt:
        eid = ev["event_id"]
        if eid not in per_event:
            continue
        d = per_event[eid]["R0_loose_substring"] - per_event[eid]["R4_identity"]
        if d:
            ax = IDENTITY.get(eid, ("?", ""))[0]
            print(f"   {eid:34s} [{ax:9s}]  -{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
