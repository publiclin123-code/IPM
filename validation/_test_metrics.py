"""
metrics.py 的快速回归测试。重点验证 ECE 计算与 trigger 忠实性逻辑。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import compute_ece, faithfulness_audit


# ---------- ECE ----------
# 构造完美校准: 每个 bin 内 acc == avg_conf
perfect = ([{"confidence": 0.05, "correct": False}] * 10
         + [{"confidence": 0.15, "correct": True}] * 2 + [{"confidence": 0.15, "correct": False}] * 8
         + [{"confidence": 0.95, "correct": True}] * 9 + [{"confidence": 0.95, "correct": False}] * 1)
res = compute_ece(perfect, n_bins=10)
# bin 1 (0-0.1): 10 items, all False → acc=0, avg_conf=0.05, gap=0.05
# bin 2 (0.1-0.2): 10 items, 2 True → acc=0.2, avg_conf=0.15, gap=0.05
# bin 10 (0.9-1.0): 10 items, 9 True → acc=0.9, avg_conf=0.95, gap=0.05
# 其他 bin 空 → ECE = (10/30)*0.05 + (10/30)*0.05 + (10/30)*0.05 = 0.05
assert abs(res["ece"] - 0.05) < 1e-6, f"perfect ECE wrong: {res['ece']}"
print(f"[PASS] ECE perfect calibration = {res['ece']} (expected ~0.05)")

# 完全错校准: 所有 confidence=0.9，但 acc=0.1
miscal = [{"confidence": 0.9, "correct": False}] * 9 + [{"confidence": 0.9, "correct": True}] * 1
res2 = compute_ece(miscal, n_bins=10)
# bin 10: 10 items, 1 True → acc=0.1, avg_conf=0.9, gap=0.8; ECE = 0.8
assert abs(res2["ece"] - 0.8) < 1e-6, f"miscal ECE wrong: {res2['ece']}"
print(f"[PASS] ECE miscalibration = {res2['ece']} (expected 0.8)")


# ---------- Faithfulness ----------
# 构造: 部分信号 trigger 在原文中，部分不在
signals = [
    {"signal_id": "OK1", "input_id": "A1",
     "trigger_phrases": ["red sea", "houthi attack"]},
    {"signal_id": "OK2", "input_id": "A2",
     "trigger_phrases": ["chip export"]},
    {"signal_id": "BAD1", "input_id": "A1",
     "trigger_phrases": ["fabricated phrase that does not exist"]},
]
news = {
    "A1": {"id": "A1", "title": "Red Sea tensions", "text": "Houthi attack reported."},
    "A2": {"id": "A2", "title": "Chip export controls", "text": "..."},
}
result = faithfulness_audit(signals, news)
assert result["n_total"] == 3
assert result["n_faithful"] == 2  # OK1, OK2
assert result["n_violations"] == 1
assert result["faithfulness_rate"] == round(2 / 3, 4)
assert result["violations"][0]["signal_id"] == "BAD1"
assert result["violations"][0]["missing_triggers"] == ["fabricated phrase that does not exist"]
print(f"[PASS] Faithfulness = {result['faithfulness_rate']} (2/3), 1 violation ✓")
