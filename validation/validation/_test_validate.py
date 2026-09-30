"""
validate.py 的快速回归测试。构造合成信号集，验证命中判定与指标计算正确。

跑: python _test_validate.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "_test_tmp"
TMP.mkdir(exist_ok=True)

# 构造合成信号 (Red Sea, onset=2023-11-19)
synthetic_signals = [
    # 命中: forward_looking, 在窗口内, geo+commodity 匹配
    {
        "signal_id": "S1", "input_id": "X1", "signal_date": "2023-11-05",
        "temporality": "forward_looking", "event_type": "geopolitical",
        "description": "Houthi threat to Red Sea shipping",
        "commodities": ["oil"], "geographies": ["Red Sea"],
        "companies": [{"name": "Maersk", "role": "logistics"}],
        "severity": 4, "confidence": 0.8,
        "trigger_phrases": ["houthi threat", "red sea"],
    },
    # 命中 (latency): latent, 早 30 天; Suez Canal 与 GT geographies 子串匹配
    {
        "signal_id": "S2", "input_id": "X2", "signal_date": "2023-10-20",
        "temporality": "latent", "event_type": "geopolitical",
        "description": "Suez Canal traffic at risk from regional tensions",
        "commodities": ["containerized goods"], "geographies": ["Suez Canal"],
        "companies": [], "severity": 3, "confidence": 0.6,
        "trigger_phrases": ["suez canal"],
    },
    # 命中 (company-only match): 仅有公司匹配 (Maersk)，无 geo/commodity 匹配
    {
        "signal_id": "S2b", "input_id": "X2b", "signal_date": "2023-10-15",
        "temporality": "forward_looking", "event_type": "logistics",
        "description": "Maersk warns of route disruption",
        "commodities": ["coffee"], "geographies": ["Brazil"],
        "companies": [{"name": "Maersk", "role": "logistics"}],
        "severity": 3, "confidence": 0.65,
        "trigger_phrases": ["maersk warns"],
    },
    # 不命中: confirmation temporality
    {
        "signal_id": "S3", "input_id": "X3", "signal_date": "2023-11-10",
        "temporality": "confirmation", "event_type": "geopolitical",
        "description": "Ship attacked", "commodities": ["oil"], "geographies": ["Red Sea"],
        "companies": [], "severity": 4, "confidence": 0.9, "trigger_phrases": ["attack"],
    },
    # 不命中: signal_date >= onset
    {
        "signal_id": "S4", "input_id": "X4", "signal_date": "2023-11-25",
        "temporality": "forward_looking", "event_type": "geopolitical",
        "description": "More disruption expected", "commodities": ["oil"], "geographies": ["Red Sea"],
        "companies": [], "severity": 4, "confidence": 0.7, "trigger_phrases": ["expected"],
    },
    # 不命中: 不匹配 (无关地区/商品)
    {
        "signal_id": "S5", "input_id": "X5", "signal_date": "2023-11-01",
        "temporality": "forward_looking", "event_type": "logistics",
        "description": "Port strike in Brazil",
        "commodities": ["coffee"], "geographies": ["Brazil"],
        "companies": [], "severity": 2, "confidence": 0.5, "trigger_phrases": ["strike"],
    },
    # 不命中: 在窗口外 (>180 天前)
    {
        "signal_id": "S6", "input_id": "X6", "signal_date": "2023-04-01",
        "temporality": "forward_looking", "event_type": "geopolitical",
        "description": "Early warning",
        "commodities": ["oil"], "geographies": ["Red Sea"],
        "companies": [], "severity": 3, "confidence": 0.4, "trigger_phrases": ["early"],
    },
]

sig_path = TMP / "_synthetic_signals.jsonl"
with open(sig_path, "w", encoding="utf-8") as f:
    for s in synthetic_signals:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# 用真实 GT 跑（Red Sea onset=2023-11-19，window=180d）
gt_path = str(ROOT / "gt_events.json")
report_path = str(TMP / "_synthetic_report.json")

print("Running validate.py on synthetic signals...")
r = subprocess.run([
    sys.executable, str(ROOT / "validate.py"),
    "--signals", str(sig_path), "--gt", gt_path,
    "--report", report_path, "--window", "180", "--strict",
], capture_output=True, text=True)
print(r.stderr)
if r.returncode != 0:
    print("FAIL: validate.py crashed")
    print(r.stdout)
    sys.exit(1)

with open(report_path, encoding="utf-8") as f:
    report = json.load(f)

# 断言
redsea = next(e for e in report["per_event"] if e["event_id"] == "red_sea_crisis_2023")
assert redsea["hit"] is True, f"Red Sea should hit, got {redsea}"
# STRICT 模式 (commodity+geo, or company):
#   S1 ✓ (comms oil + geo Red Sea)
#   S2 ✗ (geo Suez Canal ✓, comms containerized goods ✗ → strict 需同时满足)
#   S2b ✓ (company Maersk → company path)
# → 2 命中
assert redsea["n_signals_in_window"] == 2, (
    f"expected 2 hits (S1+S2b) in strict, got {redsea['n_signals_in_window']}: {redsea['matched_signal_ids']}")
# earliest 应该是 S2b (2023-10-15) → lead_time = 2023-11-19 - 2023-10-15 = 35 天
assert redsea["earliest_signal_date"] == "2023-10-15", (
    f"earliest should be S2b (10-15), got {redsea['earliest_signal_date']}")
assert redsea["lead_time_days"] == 35, f"lead time should be 35, got {redsea['lead_time_days']}"
assert set(redsea["matched_signal_ids"]) == {"S1", "S2b"}, redsea["matched_signal_ids"]

# S3 (confirmation) excluded ✓; S4 (post-onset) excluded ✓;
# S5 (no match) excluded ✓; S6 (out of window) excluded ✓; S2 (strict needs com+geo) excluded ✓

# Loose mode: geo OR commodity 任一即可。S2 (geo Suez Canal) 应当命中
r2 = subprocess.run([
    sys.executable, str(ROOT / "validate.py"),
    "--signals", str(sig_path), "--gt", gt_path,
    "--report", str(TMP / "_loose_report.json"), "--loose",
], capture_output=True, text=True)
assert r2.returncode == 0
with open(TMP / "_loose_report.json", encoding="utf-8") as f:
    loose_report = json.load(f)
loose_redsea = next(e for e in loose_report["per_event"] if e["event_id"] == "red_sea_crisis_2023")
assert loose_redsea["n_signals_in_window"] == 3, (
    f"loose: expected 3 hits (S1+S2+S2b), got {loose_redsea['n_signals_in_window']}: {loose_redsea['matched_signal_ids']}")
assert set(loose_redsea["matched_signal_ids"]) == {"S1", "S2", "S2b"}, loose_redsea["matched_signal_ids"]
print(f"  [loose mode] Red Sea: {loose_redsea['n_signals_in_window']} hits (S1+S2+S2b) ✓")

print("\n[PASS] All assertions OK:")
print(f"  Red Sea: hit={redsea['hit']}, lead_time={redsea['lead_time_days']}d, "
      f"n_signals={redsea['n_signals_in_window']}")
print(f"  Summary: {report['summary']}")

# 清理
import shutil
shutil.rmtree(TMP)
print("  (temp files cleaned)")
