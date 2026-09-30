#!/usr/bin/env python3
"""v2 pre-onset extraction. Writes to results/v2/; never overwrites v1 signals.

Usage:
  python pipeline/run_v2_preonset.py
  python pipeline/run_v2_preonset.py --events red_sea_crisis_2023
  python pipeline/run_v2_preonset.py --limit 1          # smoke one article of first event
  python pipeline/run_v2_preonset.py --no-thinking
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
VAL = ROOT / "validation"
DATA = ROOT / "data" / "by_event"
OUT = ROOT / "results" / "v2"
PROMPT = ROOT / "prompts" / "event_extraction_prompt_v2_temporal.md"
GT = VAL / "gt_events.json"
PY = sys.executable

# Smallest-first so QA happens before the two large events.
EVENTS = [
    "red_sea_crisis_2023",
    "port_los_angeles_backlog_2021",
    "renesas_earthquake_2016",
    "toyota_steel_explosion_2019",
    "us_chip_export_controls_2022",
]


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(OUT / "_run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> int:
    log(f">>> {label}")
    log("    " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        log(f"    !! {label} FAILED exit={r.returncode}")
    return r.returncode


def count_status(path: Path) -> dict:
    n_sig = n_err = n_empty = 0
    if not path.exists():
        return {"signals": 0, "errors": 0, "empty": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            st = rec.get("status")
            if st == "error":
                n_err += 1
            elif st == "empty":
                n_empty += 1
            else:
                n_sig += 1
    return {"signals": n_sig, "errors": n_err, "empty": n_empty}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--out-dir", default=str(OUT), help="output dir (default results/v2)")
    ap.add_argument("--events", default="", help="comma-sep event_ids (default: planned five)")
    ap.add_argument("--data-dir", default=str(DATA), help="corpus dir (default data/by_event)")
    ap.add_argument("--prompt", default=str(PROMPT),
                    help="prompt template; use event_extraction_prompt.md for the "
                         "single-field ontology and event_extraction_prompt_v2_temporal.md "
                         "for the clock split")
    ap.add_argument("--limit", type=int, default=0, help="per-event article cap (0=all)")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    prompt_path = Path(args.prompt)

    if not prompt_path.exists():
        print(f"missing prompt: {prompt_path}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    wanted = [e.strip() for e in args.events.split(",") if e.strip()] or list(EVENTS)
    data_dir = Path(args.data_dir)
    summary = []

    for eid in wanted:
        data_path = data_dir / f"{eid}.jsonl"
        sig_path = OUT / f"{eid}_signals.jsonl"
        if not data_path.exists():
            log(f"SKIP {eid}: no {data_path.name}")
            summary.append({"event_id": eid, "status": "no_data"})
            continue
        n_art = sum(1 for _ in open(data_path, encoding="utf-8"))
        if n_art == 0:
            log(f"SKIP {eid}: 0 articles (designed negative)")
            summary.append({"event_id": eid, "status": "zero_articles"})
            continue

        log(f"=== EXTRACT {eid} n_articles={n_art} prompt={prompt_path.name} ===")
        cmd = [
            PY, str(PIPE / "extract_events.py"),
            "--input", str(data_path),
            "--output", str(sig_path),
            "--model", args.model,
            "--prompt", str(prompt_path),
            "--resume",
        ]
        if args.base_url:
            cmd += ["--base-url", args.base_url]
        if args.no_thinking:
            cmd.append("--no-thinking")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        rc_ex = run(cmd, f"{eid}: extract")
        counts = count_status(sig_path)
        log(f"    counts {eid}: {counts}")
        if counts["signals"] + counts["errors"] + counts["empty"] > 0:
            err_rate = counts["errors"] / max(counts["signals"] + counts["errors"] + counts["empty"], 1)
            if err_rate > 0.20:
                log(f"STOP: {eid} error rate {err_rate:.0%} > 20%")
                summary.append({"event_id": eid, "status": "high_error", "counts": counts, "rc": rc_ex})
                (OUT / "_batch_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                return 3

        rc_val = rc_met = None
        if not args.skip_validate and sig_path.exists():
            rep_loose = OUT / f"{eid}_validation_loose.json"
            met_path = OUT / f"{eid}_metrics.json"
            rc_val = run(
                [PY, str(VAL / "validate.py"),
                 "--signals", str(sig_path), "--gt", str(GT),
                 "--report", str(rep_loose), "--window", "180", "--loose"],
                f"{eid}: validate loose",
            )
            rc_met = run(
                [PY, str(VAL / "metrics.py"),
                 "--signals", str(sig_path), "--gt", str(GT), "--news", str(data_path),
                 "--out", str(met_path), "--window", "180"],
                f"{eid}: metrics",
            )

        summary.append({
            "event_id": eid, "status": "ran", "n_articles": n_art,
            "counts": counts, "rc_extract": rc_ex, "rc_validate": rc_val, "rc_metrics": rc_met,
        })

    (OUT / "_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"summary -> {OUT / '_batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
