#!/bin/bash
# Plan B: fetch 收尾后全自动推进 ① + ②
set -u
cd /home/e/LF_projects/Risk_analysis
PY=/home/e/miniconda3/bin/python

echo "[1/4] waiting for pass3 ..."
while pgrep -f fetch_bodies_pass3 >/dev/null; do sleep 20; done
tail -2 data/body_fetch/pass3.log

echo "[2/4] rebuild corpus + worksheet"
$PY pipeline/build_body_corpus.py
$PY pipeline/build_audit_body_worksheet.py

echo "[3/4] final fetch stats"
$PY - <<'EOF'
import json, collections
recs=[json.loads(l) for l in open('data/body_fetch/bodies.jsonl',encoding='utf-8')]
best={}
for r in recs:
    h=r['url_hash']; o=best.get(h)
    if o is None or (r['status'].startswith('ok') and not o['status'].startswith('ok')) or (r['status'].startswith('ok') and r['text_len']>o['text_len']): best[h]=r
st=collections.Counter(v['status'] for v in best.values())
ok=sum(v for k,v in st.items() if k.startswith('ok'))
print('unique url_hash:', len(best), dict(st))
print(f'FETCH SUCCESS: {ok}/{len(best)} = {ok/len(best):.1%}')
EOF

echo "[4/4] launch body extraction shards"
bash pipeline/launch_body_extraction.sh
echo "CHAIN DONE $(date)"
