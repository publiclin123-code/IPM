#!/bin/bash
# Plan B: 等 4 片提取全部结束 → validate+metrics → title vs body 对比
set -u
cd /home/e/LF_projects/Risk_analysis
LOG=data/body_fetch/watch_extract.log
echo "[watch] waiting for extraction shards ..." >> $LOG
while pgrep -f run_v2_preonset >/dev/null; do sleep 60; done
echo "[watch] shards done at $(date)" >> $LOG

# 兜底再等 2 分钟防僵尸收尾
sleep 120

echo "[watch] validate + metrics" >> $LOG
bash pipeline/validate_body_extraction.sh >> $LOG 2>&1

echo "[watch] comparison" >> $LOG
/home/e/miniconda3/bin/python pipeline/compare_title_body.py >> $LOG 2>&1

echo "[watch] ALL DONE $(date)" >> $LOG
