#!/bin/bash
# Phases 3 and 4 of the qwen3.8 re-extraction.
#
# Phase 3: the post-onset negative-control pool, which produces beta_0. Four
#   cells, because beta_0 is a naive-versus-clock-split comparison and the
#   manuscript reports it on both corpora:
#     slug x single-field, slug x clock-split,
#     body x single-field, body x clock-split
#   340 articles each.
#
# Phase 4: the SEC EDGAR filings, which use a separate prompt
#   (event_extraction_prompt_financial_distress.md). The manuscript reports 39
#   forward signals at precision 1.000 on these; that number is model-dependent
#   and must be regenerated on the same model as everything else.
#
# Run this after `reextract_q38.sh` has finished.
#
# Usage:
#   bash pipeline/reextract_q38_phases34.sh
set -u
cd /home/e/LF_projects/Risk_analysis
export LLM_API_KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT
PY=/home/e/miniconda3/bin/python
MODEL=qwen3.8-27b
BASE=http://localhost:8080

V1=prompts/event_extraction_prompt.md
V2=prompts/event_extraction_prompt_v2_temporal.md
VF=prompts/event_extraction_prompt_financial_distress.md

echo "=== PHASE 3: post-onset negative-control pool (beta_0) ==="
mkdir -p results/q38_bg_slug_v1 results/q38_bg_slug_v2 results/q38_bg_body_v1 results/q38_bg_body_v2

for spec in "data/by_event:$V1:results/q38_bg_slug_v1:slug x single-field" \
            "data/by_event:$V2:results/q38_bg_slug_v2:slug x clock-split" \
            "data/by_event_body:$V1:results/q38_bg_body_v1:body x single-field" \
            "data/by_event_body:$V2:results/q38_bg_body_v2:body x clock-split"; do
  IFS=: read -r DATA PROMPT OUT LABEL <<< "$spec"
  mkdir -p "$OUT"
  printf "%s\ncorpus=%s\nprompt=%s\nmodel=%s\n" "$LABEL" "$DATA" "$PROMPT" "$MODEL" > "$OUT/_model.txt"
  echo "  $LABEL"
  $PY pipeline/extract_events.py \
    --input "$DATA/_background.jsonl" \
    --output "$OUT/_background_signals.jsonl" \
    --model "$MODEL" --prompt "$PROMPT" \
    --base-url "$BASE" --no-thinking --resume \
    > "$OUT/_extract.log" 2>&1
  tail -1 "$OUT/_extract.log" | sed 's/^/    /'
done

echo
echo "=== PHASE 4: SEC EDGAR (financial-distress prompt) ==="
mkdir -p results/q38_edgar
printf "edgar\nmodel=%s\nprompt=%s\n" "$MODEL" "$VF" > results/q38_edgar/_model.txt
for i in 1 2 3; do
  case $i in
    1) EV="bbby_bankruptcy_2023,rite_aid_bankruptcy_2023,party_city_bankruptcy_2023,yellow_bankruptcy_2023" ;;
    2) EV="joann_bankruptcy_2024,big_lots_bankruptcy_2024,express_bankruptcy_2024,wework_bankruptcy_2023" ;;
    3) EV="lordstown_bankruptcy_2023,tuesday_morning_bankruptcy_2023,smiledirect_bankruptcy_2023,audacy_bankruptcy_2024" ;;
  esac
  $PY pipeline/run_edgar_extract.py \
    --model "$MODEL" --prompt "$VF" --base-url "$BASE" \
    --out-dir results/q38_edgar --events "$EV" \
    > "results/q38_edgar/_shard$i.log" 2>&1 &
  echo "  edgar shard$i PID $! : $EV"
done
wait

echo
echo "PHASES 3+4 DONE $(date)"
