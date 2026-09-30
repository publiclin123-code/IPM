"""
Paper A · RQ1+RQ4 端到端编排脚本
按 GT 事件批量执行: 下载 GDELT 窗口 → LLM 抽取信号 → 前视验证 → 可信度指标。

用法:
  # 单事件 pilot
  python run_pipeline.py --events red_sea_crisis_2023 --window 14

  # 全部事件 (180 天窗口)
  python run_pipeline.py

  # 跳过下载（已有 data/by_event/{event_id}.jsonl）
  python run_pipeline.py --skip-download --events red_sea_crisis_2023

输出目录布局:
  data/by_event/{event_id}.jsonl          # 原始新闻
  results/{event_id}_signals.jsonl        # LLM 抽取信号
  results/{event_id}_validation.json      # 命中/lead time/precision
  results/{event_id}_metrics.json         # ECE/faithfulness
  results/_pipeline_summary.json          # 总汇
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(__file__).resolve().parent
PYTHON = sys.executable

GT_PATH = PAPER / "validation" / "gt_events.json"
DATA_DIR = PAPER / "data" / "by_event"
RESULTS_DIR = PAPER / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], label: str):
    print(f"\n>>> {label}: {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {label}: rc={r.returncode}", file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
    elif r.stderr:
        # 提取 stderr 摘要（脚本通常把进度打到 stderr）
        tail = r.stderr.strip().splitlines()[-5:]
        for line in tail:
            print(f"  {line}", file=sys.stderr)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper A 端到端编排")
    ap.add_argument("--events", default="", help="逗号分隔 event_id；默认全部")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--base-url", default="", help="llama-server 地址；留空按模型名自动定位")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true",
                    help="复用已有 signals.jsonl，只重跑验证+指标")
    args = ap.parse_args()

    selected = set(args.events.split(",")) if args.events else None
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    events = gt_data.get("events", gt_data) if isinstance(gt_data, dict) else gt_data

    pipeline = []
    for ev in events:
        eid = ev["event_id"]
        if selected and eid not in selected:
            continue
        data_path = DATA_DIR / f"{eid}.jsonl"
        sig_path = RESULTS_DIR / f"{eid}_signals.jsonl"
        val_path = RESULTS_DIR / f"{eid}_validation.json"
        met_path = RESULTS_DIR / f"{eid}_metrics.json"

        if not args.skip_download:
            run([PYTHON, str(PAPER / "pipeline" / "fetch_per_event.py"),
                 "--events", eid, "--window", str(args.window)],
                label=f"download {eid}")
        if not args.skip_extract and data_path.exists():
            cmd = [PYTHON, str(PAPER / "pipeline" / "extract_events.py"),
                   "--input", str(data_path), "--output", str(sig_path),
                   "--model", args.model]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
            run(cmd, label=f"extract {eid}")
        if sig_path.exists():
            run([PYTHON, str(PAPER / "validation" / "validate.py"),
                 "--signals", str(sig_path), "--gt", str(GT_PATH),
                 "--report", str(val_path), "--window", str(args.window), "--strict"],
                label=f"validate(strict) {eid}")
            run([PYTHON, str(PAPER / "validation" / "metrics.py"),
                 "--signals", str(sig_path), "--gt", str(GT_PATH),
                 "--news", str(data_path), "--out", str(met_path),
                 "--window", str(args.window)],
                label=f"metrics {eid}")
        else:
            print(f"[skip] {eid}: no signals.jsonl", file=sys.stderr)
        pipeline.append({"event_id": eid,
                         "data_path": str(data_path.relative_to(ROOT)) if data_path.exists() else None,
                         "signals_path": str(sig_path.relative_to(ROOT)) if sig_path.exists() else None,
                         "validation_path": str(val_path.relative_to(ROOT)) if val_path.exists() else None,
                         "metrics_path": str(met_path.relative_to(ROOT)) if met_path.exists() else None})

    # 汇总
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "window_days": args.window,
        "events": pipeline,
    }
    summary_path = RESULTS_DIR / "_pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[done] pipeline summary -> {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
