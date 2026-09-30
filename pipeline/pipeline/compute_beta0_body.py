#!/usr/bin/env python3
"""β₀(body): 背景池正文版提取的 brier skill 变化, 对齐 title 版验证配置.

对齐项: gt_events.json + _background.jsonl 构建 id2event (post_days=60),
compute_bscc 窗口 w=14. 输出 title(基线, 来自 _new 文件) vs body 对比.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))
from metrics import compute_bscc, _build_article_event_map  # noqa: E402

gt = json.loads((ROOT / "validation" / "gt_events.json").read_text())["events"]
id2event = _build_article_event_map(ROOT / "data" / "by_event" / "_background.jsonl",
                                    gt, post_days=60)

OUTD = ROOT / "results" / "_background"
runs = {
    "title_naive": OUTD / "signals_postevent_naive_new.jsonl",
    "title_v2": OUTD / "signals_postevent_v2_new.jsonl",
    "body_naive": OUTD / "signals_postevent_naive_body.jsonl",
    "body_v2": OUTD / "signals_postevent_v2_body.jsonl",
}

summary = {}
for name, p in runs.items():
    if not p.exists():
        print(f"!! missing {p}")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    metrics = compute_bscc(rows, gt, post_days=14, id2event=id2event)
    metrics["n_signals"] = len(rows)
    summary[name] = metrics
    print(f"{name:12s} n={len(rows):4d} " +
          " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in metrics.items()))

(OUTD / "beta0_body_summary.json").write_text(json.dumps(summary, indent=2))
print(f"written {OUTD/'beta0_body_summary.json'}")
