#!/bin/bash
# pass4 完成后的全自动接力: 重建语料 → 重提取 → validate → 对比
set -u
cd /home/e/LF_projects/Risk_analysis
LOG=data/body_fetch/watch_pass4.log
echo "[watch] waiting for pass4 ..." >> $LOG
while pgrep -f fetch_bodies_pass4 >/dev/null; do sleep 60; done
echo "[watch] pass4 done $(date)" >> $LOG
tail -3 data/body_fetch/pass4.log >> $LOG

echo "[watch] rebuild corpus + worksheet" >> $LOG
/home/e/miniconda3/bin/python pipeline/build_body_corpus.py >> $LOG 2>&1
/home/e/miniconda3/bin/python pipeline/build_audit_body_worksheet.py >> $LOG 2>&1
# no_fetch 行预填旧标签 + 正文行排前 (同上次整理)
/home/e/miniconda3/bin/python - <<'EOF' >> $LOG 2>&1
import csv
src='data/annotation/retrieval_audit_sample_body.csv'
rows=list(csv.DictReader(open(src,encoding='utf-8')))
fields=list(rows[0].keys())
for r in rows:
    if r['body_status']=='no_fetch' and not (r.get('relevant_label_body') or '').strip():
        r['relevant_label_body']=r.get('relevant_label','')
        r['note_body']='carried over (no body fetched)'
rows.sort(key=lambda r:(0 if r['body_status']!='no_fetch' else 1, r['event_id']))
with open(src,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('worksheet sorted+prefilled')
EOF

echo "[watch] invalidate stale rows for pass4-recovered articles" >> $LOG
/home/e/miniconda3/bin/python pipeline/invalidate_changed_ids.py >> $LOG 2>&1

echo "[watch] relaunch extraction" >> $LOG
bash pipeline/launch_body_extraction.sh > results/v2_body/_launch2.log 2>&1
echo "[watch] extraction done $(date)" >> $LOG

echo "[watch] validate + compare" >> $LOG
bash pipeline/validate_body_extraction.sh >> $LOG 2>&1
rm -f results/v2_body/_aggregate_18events.json
/home/e/miniconda3/bin/python pipeline/compare_title_body.py >> $LOG 2>&1

echo "[watch] LLM prefill annotation suggestions" >> $LOG
/home/e/miniconda3/bin/python pipeline/prefill_annotation_body.py >> $LOG 2>&1
echo "[watch] ALL DONE $(date)" >> $LOG
