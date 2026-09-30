#!/usr/bin/env python3
"""Phase 2b · DeepSeek V4 Flash robustness extraction (commercial-API contrast).

Runs the identical extraction protocol with DeepSeek V4 Flash over both corpora:
  - GDELT news (18 events, v2 prompt)          -> results/deepseek/gdelt/{eid}_signals.jsonl
  - SEC EDGAR filings (12 firms, distress prompt) -> results/deepseek/edgar/{eid}_signals.jsonl

This is the commercial-API robustness check referenced in the paper. The API key
is read from ~/.deepseek_key (never hard-coded, never printed).

Usage:
  python pipeline/run_deepseek.py --corpus gdelt
  python pipeline/run_deepseek.py --corpus edgar
  python pipeline/run_deepseek.py --corpus both [--events a,b] [--limit N] [--no-validate]
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
DATA_GDELT = ROOT / "data" / "by_event"
DATA_EDGAR = ROOT / "data" / "edgar"
OUT = ROOT / "results" / "deepseek"

PY = sys.executable
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
PROMPT_V2 = ROOT / "prompts" / "event_extraction_prompt_v2_temporal.md"
PROMPT_FIN = ROOT / "prompts" / "event_extraction_prompt_financial_distress.md"


def api_key() -> str:
    p = Path.home() / ".deepseek_key"
    return p.read_text().strip() if p.exists() else ""


def run_extract(data_path, sig_path, prompt_path, key):
    cmd = [PY, str(PIPE / "extract_events.py"),
           "--input", str(data_path), "--output", str(sig_path),
           "--model", MODEL, "--prompt", str(prompt_path), "--resume",
           "--base-url", BASE_URL, "--api-key", key, "--no-thinking"]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def count(sig_path):
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
    return sig, emp, err


def validate(sig_path, data_path, gt_path):
    rep = sig_path.with_name(sig_path.stem.replace("_signals", "_validation_loose") + ".json")
    met = sig_path.with_name(sig_path.stem.replace("_signals", "_metrics") + ".json")
    r1 = subprocess.run([PY, str(VAL / "validate.py"),
                         "--signals", str(sig_path), "--gt", str(gt_path),
                         "--report", str(rep), "--window", "180", "--loose"],
                        cwd=str(ROOT)).returncode
    r2 = subprocess.run([PY, str(VAL / "metrics.py"),
                         "--signals", str(sig_path), "--gt", str(gt_path),
                         "--news", str(data_path), "--out", str(met), "--window", "180"],
                        cwd=str(ROOT)).returncode
    return r1, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["gdelt", "edgar", "both"], default="both")
    ap.add_argument("--events", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    key = api_key()
    if not key:
        print("ERROR: ~/.deepseek_key missing or empty", file=sys.stderr)
        return 2
    only = {e.strip() for e in args.events.split(",") if e.strip()}

    jobs = []
    if args.corpus in ("gdelt", "both"):
        gt = json.load(open(VAL / "gt_events.json", encoding="utf-8"))["events"]
        for ev in gt:
            if only and ev["event_id"] not in only:
                continue
            jobs.append(("gdelt", ev["event_id"], DATA_GDELT / f"{ev['event_id']}.jsonl",
                         OUT / "gdelt" / f"{ev['event_id']}_signals.jsonl", PROMPT_V2,
                         VAL / "gt_events.json"))
    if args.corpus in ("edgar", "both"):
        gt = json.load(open(VAL / "edgar_bankruptcy_events.json", encoding="utf-8"))["events"]
        for ev in gt:
            if only and ev["event_id"] not in only:
                continue
            jobs.append(("edgar", ev["event_id"], DATA_EDGAR / f"{ev['event_id']}.jsonl",
                         OUT / "edgar" / f"{ev['event_id']}_signals.jsonl", PROMPT_FIN,
                         VAL / "edgar_bankruptcy_events.json"))

    (OUT / "gdelt").mkdir(parents=True, exist_ok=True)
    (OUT / "edgar").mkdir(parents=True, exist_ok=True)

    summary = []
    for corpus, eid, data_path, sig_path, prompt_path, gt_path in jobs:
        if not data_path.exists():
            print(f"SKIP {corpus}/{eid}: no data", flush=True)
            continue
        n = sum(1 for _ in open(data_path, encoding="utf-8"))
        print(f"\n=== {corpus}/{eid} n={n} ===", flush=True)
        rc_ex = run_extract(data_path, sig_path, prompt_path, key)
        sig, emp, err = count(sig_path)
        print(f"    signals={sig} empty={emp} errors={err}", flush=True)
        rc_v = rc_m = None
        if not args.no_validate and sig_path.exists():
            rc_v, rc_m = validate(sig_path, data_path, gt_path)
        summary.append({"corpus": corpus, "event_id": eid, "n": n,
                        "signals": sig, "empty": emp, "errors": err,
                        "rc_extract": rc_ex, "rc_validate": rc_v, "rc_metrics": rc_m})

    (OUT / "_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {OUT / '_batch_summary.json'}", flush=True)
    for s in summary:
        print(f"  {s['corpus']:5s} {s['event_id']:32s} n={s['n']:3d} signals={s['signals']:3d} err={s['errors']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
