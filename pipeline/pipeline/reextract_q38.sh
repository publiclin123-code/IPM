#!/bin/bash
# Full re-extraction on the currently served model (qwen3.8-27b).
#
# Why: every existing result came from qwen3.6-27b, whose weights were deleted on
# 2026-09-15 (see /home/e/models/README.md). Mixing a new model into one cell
# would confound "changed the schema" with "changed the model", so the whole grid
# is re-run on one model.
#
# Grid (4 cells x 2 corpora):
#   slug  x {single-field, clock-split}
#   body  x {single-field, clock-split}
#
# Outputs go to new directories. Nothing under results/v2 or results/v2_body is
# touched, so the 3.6 results remain available for a model-robustness comparison.
#
# Usage:
#   bash pipeline/reextract_q38.sh slug      # phase 1, ~20 min
#   bash pipeline/reextract_q38.sh body      # phase 2, ~40 min
#   bash pipeline/reextract_q38.sh all       # both
set -u
cd /home/e/LF_projects/Risk_analysis
export LLM_API_KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT
PY=/home/e/miniconda3/bin/python
MODEL=qwen3.8-27b
BASE=http://localhost:8080

V1=prompts/event_extraction_prompt.md
V2=prompts/event_extraction_prompt_v2_temporal.md

# 17 events, sharded by article count so the four slots finish together.
# suez (0 articles) and naka (1) are kept for completeness; the runner skips
# zero-article events and logs it.
S1="us_chip_export_controls_2022,hurricane_maria_2017,beirut_port_explosion_2020,renesas_naka_plant_fire_2021"
S2="russia_ukraine_war_2022,port_los_angeles_backlog_2021,black_sea_grain_exit_2023,suez_ever_given_2021"
S3="renesas_earthquake_2016,covid_supply_disruption_2020,india_wheat_export_ban_2022,uaw_auto_strike_2023"
S4="europe_energy_crisis_2022,egg_shortage_birdflu_2025,red_sea_crisis_2023,taiwan_strait_crisis_2022,us_china_tariff_war_2018"

run_cell () {   # $1=corpus_dir  $2=prompt  $3=out_dir  $4=label
  local DATA=$1 PROMPT=$2 OUT=$3 LABEL=$4
  mkdir -p "$OUT"
  echo "=== $LABEL ===  corpus=$DATA  prompt=$(basename $PROMPT)  model=$MODEL"
  echo "$LABEL  $(date)" > "$OUT/_model.txt"
  echo "corpus=$DATA"     >> "$OUT/_model.txt"
  echo "prompt=$PROMPT"   >> "$OUT/_model.txt"
  echo "model=$MODEL"     >> "$OUT/_model.txt"
  for i in 1 2 3 4; do
    eval "EV=\$S$i"
    $PY pipeline/run_v2_preonset.py \
      --model "$MODEL" \
      --data-dir "$DATA" \
      --prompt "$PROMPT" \
      --base-url "$BASE" \
      --out-dir "$OUT" \
      --events "$EV" \
      --no-thinking --skip-validate \
      > "$OUT/_shard$i.log" 2>&1 &
    echo "  shard$i PID $! : $EV"
  done
  wait
  echo "  $LABEL DONE $(date)"
}

WHICH=${1:-all}

if [ "$WHICH" = "slug" ] || [ "$WHICH" = "all" ]; then
  run_cell data/by_event      $V1 results/q38_slug_v1 "slug x single-field"
  run_cell data/by_event      $V2 results/q38_slug_v2 "slug x clock-split"
fi

if [ "$WHICH" = "body" ] || [ "$WHICH" = "all" ]; then
  run_cell data/by_event_body $V1 results/q38_body_v1 "body x single-field"
  run_cell data/by_event_body $V2 results/q38_body_v2 "body x clock-split"
fi

echo "ALL DONE $(date)"
