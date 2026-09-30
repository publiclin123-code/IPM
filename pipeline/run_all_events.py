#!/usr/bin/env python3
"""
Batch orchestrator: run fetch -> extract -> validate -> metrics for all GT events.

Usage:
    python run_all_events.py --window 180 --model qwen3.6-27b [--skip-done] [--events a,b,c]

Skips events whose onset is before GDELT 2.0 start (2013-02-18) automatically.
With --skip-done, also skips events that already have a *_signals.jsonl in results/.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # paper_a_early_warning/
PIPE = ROOT / "pipeline"
VAL = ROOT / "validation"
DATA = ROOT / "data" / "by_event"
RES = ROOT / "results"
GT = VAL / "gt_events.json"
GDELT_START = datetime(2013, 2, 18)

PY = sys.executable  # was hardcoded Windows path E:\anaconda\python.exe


def load_events():
    with open(GT, encoding="utf-8") as f:
        d = json.load(f)
    return d["events"]


def run(cmd: list[str], label: str) -> int:
    print(f"\n>>> {label}", flush=True)
    print("    cmd:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"    !! {label} FAILED (exit {r.returncode})", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--base-url", default="", help="llama-server 地址；留空按模型名自动定位")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--skip-done", action="store_true", help="skip events with existing signals file")
    ap.add_argument("--events", default="", help="comma-sep event_ids to run (default: all)")
    ap.add_argument("--only", choices=["fetch", "extract", "validate", "all"], default="all")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    only_filter = set(args.events.split(",")) if args.events else None
    events = load_events()
    summary = []

    for ev in events:
        eid = ev["event_id"]
        if only_filter and eid not in only_filter:
            continue
        onset = datetime.fromisoformat(ev["gt_onset_date"])
        if onset < GDELT_START:
            print(f"\n=== SKIP {eid} (onset {ev['gt_onset_date']} before GDELT 2.0 start) ===", flush=True)
            summary.append({"event_id": eid, "status": "skipped_pre_gdelt"})
            continue

        sig_path = RES / f"{eid}_signals.jsonl"
        if args.skip_done and sig_path.exists():
            print(f"\n=== SKIP {eid} (signals already exist: {sig_path.name}) ===", flush=True)
            summary.append({"event_id": eid, "status": "skipped_done"})
            continue

        print(f"\n{'='*70}", flush=True)
        print(f"=== EVENT: {eid} ({ev['event_name']})", flush=True)
        print(f"    onset: {ev['gt_onset_date']} | type: {ev['event_type']}", flush=True)
        print(f"{'='*70}", flush=True)

        data_path = DATA / f"{eid}.jsonl"
        rep_strict = RES / f"{eid}_validation.json"
        rep_loose = RES / f"{eid}_validation_loose.json"
        met_path = RES / f"{eid}_metrics.json"
        rc = {"fetch": None, "extract": None, "validate": None, "metrics": None}

        # 1. fetch
        if args.only in ("all", "fetch"):
            rc["fetch"] = run(
                [PY, str(PIPE / "fetch_per_event.py"),
                 "--events", eid, "--window", str(args.window), "--sleep", str(args.sleep)],
                f"{eid}: fetch",
            )

        # 2. extract
        if args.only in ("all", "extract") and data_path.exists():
            cmd = [PY, str(PIPE / "extract_events.py"),
                   "--input", str(data_path), "--output", str(sig_path), "--model", args.model]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
            rc["extract"] = run(cmd, f"{eid}: extract")
        elif args.only in ("all", "extract") and not data_path.exists():
            print(f"    !! {eid}: no data file {data_path.name}, skip extract", flush=True)

        # 3. validate (strict + loose)
        if args.only in ("all", "validate") and sig_path.exists():
            rc["validate"] = run(
                [PY, str(VAL / "validate.py"),
                 "--signals", str(sig_path), "--gt", str(GT),
                 "--report", str(rep_strict), "--window", str(args.window), "--strict"],
                f"{eid}: validate strict",
            )
            run(
                [PY, str(VAL / "validate.py"),
                 "--signals", str(sig_path), "--gt", str(GT),
                 "--report", str(rep_loose), "--window", str(args.window), "--loose"],
                f"{eid}: validate loose",
            )
            # 4. metrics
            rc["metrics"] = run(
                [PY, str(VAL / "metrics.py"),
                 "--signals", str(sig_path), "--gt", str(GT), "--news", str(data_path),
                 "--out", str(met_path), "--window", str(args.window)],
                f"{eid}: metrics",
            )
        elif args.only in ("all", "validate") and not sig_path.exists():
            print(f"    !! {eid}: no signals file, skip validate+metrics", flush=True)

        # count articles + signals
        n_art = sum(1 for _ in open(data_path, encoding="utf-8")) if data_path.exists() else 0
        n_sig = 0
        if sig_path.exists():
            with open(sig_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip() and json.loads(line).get("status") != "error":
                        n_sig += 1
        summary.append({
            "event_id": eid, "status": "ran", "rc": rc,
            "n_articles": n_art, "n_signals": n_sig,
        })

    # final summary
    print(f"\n{'='*70}\n=== BATCH SUMMARY ===\n{'='*70}", flush=True)
    for s in summary:
        if s["status"] != "ran":
            print(f"  {s['event_id']}: {s['status']}", flush=True)
        else:
            print(f"  {s['event_id']}: {s['n_articles']} articles -> {s['n_signals']} signals "
                  f"(rc: fetch={s['rc']['fetch']} ex={s['rc']['extract']} "
                  f"val={s['rc']['validate']} met={s['rc']['metrics']})", flush=True)

    (RES / "_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {RES / '_batch_summary.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
