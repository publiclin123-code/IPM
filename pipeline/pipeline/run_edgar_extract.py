#!/usr/bin/env python3
"""Phase 2 · EDGAR financial-distress extraction orchestrator.

Extract going-concern / liquidity / credit-event / restructuring signals from
data/edgar/{event_id}.jsonl using the financial-distress prompt, then run the
SAME date test + entity match validation as the GDELT main results.

Signals -> results/edgar/{event_id}_signals.jsonl
Validate -> results/edgar/{event_id}_validation_loose.json (strict=False loose)
Metrics  -> results/edgar/{event_id}_metrics.json

Usage:
  python pipeline/run_edgar_extract.py [--events a,b] [--limit N] [--no-validate]
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
VAL = ROOT / "validation"
DATA = ROOT / "data" / "edgar"
OUT = ROOT / "results" / "edgar"
GT = VAL / "edgar_bankruptcy_events.json"
PROMPT = ROOT / "prompts" / "event_extraction_prompt_financial_distress.md"

PY = sys.executable
MODEL = "qwen3.6-27b"
BASE_URL = "http://localhost:8080"
API_KEY = "DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--model", default=MODEL,
                    help="model alias served by the local endpoint")
    ap.add_argument("--prompt", default=str(PROMPT))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--no-thinking", action="store_true", default=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(GT, encoding="utf-8"))["events"]
    only = {e.strip() for e in args.events.split(",") if e.strip()}
    summary = []

    for ev in gt:
        eid = ev["event_id"]
        if only and eid not in only:
            continue
        data_path = DATA / f"{eid}.jsonl"
        sig_path = out_dir / f"{eid}_signals.jsonl"
        if not data_path.exists():
            print(f"SKIP {eid}: no data", flush=True)
            continue
        n_art = sum(1 for _ in open(data_path, encoding="utf-8"))
        print(f"\n=== EXTRACT {eid} n_filings={n_art} onset={ev['gt_onset_date']} ===", flush=True)

        cmd = [PY, str(PIPE / "extract_events.py"),
               "--input", str(data_path), "--output", str(sig_path),
               "--model", args.model, "--prompt", str(args.prompt), "--resume",
               "--base-url", args.base_url, "--api-key", API_KEY, "--no-thinking"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        r = subprocess.run(cmd, cwd=str(ROOT))

        # 统计
        sig = err = emp = 0
        if sig_path.exists():
            for line in open(sig_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if s.get("status") == "error":
                    err += 1
                elif s.get("status") == "empty":
                    emp += 1
                else:
                    sig += 1
        print(f"    counts: signals={sig} empty={emp} errors={err}", flush=True)

        rc_val = rc_met = None
        if not args.no_validate and sig_path.exists():
            rep = out_dir / f"{eid}_validation_loose.json"
            met = out_dir / f"{eid}_metrics.json"
            rc_val = subprocess.run(
                [PY, str(VAL / "validate.py"),
                 "--signals", str(sig_path), "--gt", str(GT),
                 "--report", str(rep), "--window", "180", "--loose"],
                cwd=str(ROOT)).returncode
            rc_met = subprocess.run(
                [PY, str(VAL / "metrics.py"),
                 "--signals", str(sig_path), "--gt", str(GT), "--news", str(data_path),
                 "--out", str(met), "--window", "180"],
                cwd=str(ROOT)).returncode

        summary.append({"event_id": eid, "n_filings": n_art,
                        "signals": sig, "empty": emp, "errors": err,
                        "rc_extract": r.returncode, "rc_validate": rc_val, "rc_metrics": rc_met})

    (out_dir / "_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {out_dir / '_batch_summary.json'}", flush=True)
    for s in summary:
        print(f"  {s['event_id']:32s} filings={s['n_filings']:3d} signals={s['signals']:3d} empty={s['empty']:3d} err={s['errors']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
