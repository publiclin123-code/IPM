#!/bin/bash
# 总回收链 (免费多源版): pass5 -> pass6 -> 背景池二扫+重抽 -> 主池重建全链
set -u
cd /home/e/LF_projects/Risk_analysis
PY=/home/e/miniconda3/bin/python
KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT

# 1) 等 pass5 退出
while pgrep -f fetch_pass5_wayback > /dev/null; do sleep 30; done
echo "[full] pass5 exited $(date)"

# 2) pass6 多源深度回收 (免费), 跑两遍收瞬时网络失败 (可续跑)
$PY pipeline/fetch_pass6_archives.py --workers 3
$PY pipeline/fetch_pass6_archives.py --workers 3

# 3) 等背景池首轮提取链完成 (llama slot 空出来)
while ! grep -q "ALL DONE" data/body_fetch/chain_bg_extract.log 2>/dev/null; do sleep 30; done
echo "[full] bg extract round1 done $(date)"

# 4) 背景池二扫 + 失效 + 重抽变化 id + β₀(body) 重算
$PY pipeline/fetch_background_sweep2.py
$PY pipeline/extract_events.py \
  --input data/by_event_body/_background.jsonl \
  --output results/_background/signals_postevent_naive_body.jsonl \
  --prompt prompts/event_extraction_prompt.md \
  --model qwen3.6-27b --base-url http://localhost:8080 \
  --api-key "$KEY" --no-thinking --resume
$PY pipeline/extract_events.py \
  --input data/by_event_body/_background.jsonl \
  --output results/_background/signals_postevent_v2_body.jsonl \
  --prompt prompts/event_extraction_prompt_v2_temporal.md \
  --model qwen3.6-27b --base-url http://localhost:8080 \
  --api-key "$KEY" --no-thinking --resume
$PY pipeline/compute_beta0_body.py

# 5) 主池: 回填 id -> 重建语料 -> 失效 -> 4 分片重提取 -> 校验 -> 对比
$PY pipeline/backfill_wayback5_ids.py
$PY pipeline/build_body_corpus.py 2>&1 | tail -22
$PY pipeline/invalidate_changed_ids.py
bash pipeline/launch_body_extraction.sh
bash pipeline/validate_body_extraction.sh 2>&1 | tail -25
$PY pipeline/compare_title_body.py
echo "[full] ALL DONE $(date)"
