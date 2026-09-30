#!/bin/bash
# Plan B todo 7b: body 提取完成后的 validate + metrics (论文同款 --window 180 --loose)
set -u
cd /home/e/LF_projects/Risk_analysis
export LLM_API_KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT
PY=/home/e/miniconda3/bin/python
OUT=results/v2_body
EVENTS=$(python3 -c "
import json
G=json.load(open('validation/gt_events.json'))
print(','.join(e['event_id'] for e in G['events']))")

for eid in ${EVENTS//,/ }; do
  sig="$OUT/${eid}_signals.jsonl"
  [ -f "$sig" ] || { echo "SKIP $eid (no signals)"; continue; }
  n=$(wc -l < "$sig")
  echo ">>> $eid ($n rows)"
  $PY validation/validate.py --signals "$sig" --gt validation/gt_events.json \
      --report "$OUT/${eid}_validation_loose.json" --window 180 --loose || echo "  validate FAIL $eid"
  $PY validation/metrics.py --signals "$sig" --gt validation/gt_events.json \
      --news "data/by_event_body/${eid}.jsonl" --out "$OUT/${eid}_metrics.json" --window 180 || echo "  metrics FAIL $eid"
done
echo "VALIDATION+METRICS DONE $(date)"
