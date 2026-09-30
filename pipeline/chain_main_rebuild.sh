#!/bin/bash
# 主池正文重建链 (pass5 完成后): 回填id -> 重建语料 -> 失效 -> 4分片重提取
#   -> validate+metrics -> compare_title_body
set -u
cd /home/e/LF_projects/Risk_analysis
PY=/home/e/miniconda3/bin/python

# 1) 等 pass5 退出
while pgrep -f fetch_pass5_wayback > /dev/null; do sleep 30; done
echo "[main-chain] pass5 exited $(date)"

# 2) 等背景池提取链完成 (避免 llama-server slot 竞争)
while ! grep -q "ALL DONE" data/body_fetch/chain_bg_extract.log 2>/dev/null; do sleep 30; done
echo "[main-chain] background extract chain done $(date)"

# 3) 回填 wayback5 行的 event_id/article_id
$PY pipeline/backfill_wayback5_ids.py

# 4) 重建 18 事件正文语料 (干净优先合并)
$PY pipeline/build_body_corpus.py 2>&1 | tail -22

# 5) 失效被恢复正文的文章 -> resume 重抽
$PY pipeline/invalidate_changed_ids.py

# 6) 4 分片重提取 (论文同款 v2 temporal prompt)
bash pipeline/launch_body_extraction.sh

# 7) validate + metrics
bash pipeline/validate_body_extraction.sh 2>&1 | tail -25

# 8) title vs body 对比刷新
$PY pipeline/compare_title_body.py
echo "[main-chain] ALL DONE $(date)"
