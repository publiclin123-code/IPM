#!/bin/bash
# 背景池 (post-event, 340 行) 正文版提取 + β₀(body) 计算
# 等待 fetch_background_bodies 完成后依次: naive v1 -> v2 temporal -> compute_bscc
set -u
cd /home/e/LF_projects/Risk_analysis
PY=/home/e/miniconda3/bin/python
KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT
CORPUS=data/by_event_body/_background.jsonl
OUTD=results/_background
mkdir -p "$OUTD"

# 1) 等抓取进程退出 (语料写完)
while pgrep -f fetch_background_bodies > /dev/null; do sleep 20; done
echo "[chain] fetch done at $(date); body corpus:"
wc -l "$CORPUS"

# 2) naive v1 提取
$PY pipeline/extract_events.py \
  --input "$CORPUS" \
  --output "$OUTD/signals_postevent_naive_body.jsonl" \
  --prompt prompts/event_extraction_prompt.md \
  --model qwen3.6-27b --base-url http://localhost:8080 \
  --api-key "$KEY" --no-thinking --resume

# 3) v2 temporal 提取
$PY pipeline/extract_events.py \
  --input "$CORPUS" \
  --output "$OUTD/signals_postevent_v2_body.jsonl" \
  --prompt prompts/event_extraction_prompt_v2_temporal.md \
  --model qwen3.6-27b --base-url http://localhost:8080 \
  --api-key "$KEY" --no-thinking --resume

# 4) β₀(body)
$PY pipeline/compute_beta0_body.py
echo "[chain] ALL DONE $(date)"
