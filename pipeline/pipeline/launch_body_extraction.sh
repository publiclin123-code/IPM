#!/bin/bash
# Plan B todo 7: body 语料重提取，4 片并行对齐 llama-server 4 slot
# 论文同款配置: qwen3.6-27b + event_extraction_prompt_v2_temporal.md + --no-thinking
set -u
cd /home/e/LF_projects/Risk_analysis
export LLM_API_KEY=DpeOwNpzPUYiFBpOajdWMQYJjgllcVQT
PY=/home/e/miniconda3/bin/python
OUT=results/v2_body
mkdir -p "$OUT"

# 18 事件按文章数粗均衡分 4 片 (大事件错开)
S1="us_chip_export_controls_2022,egg_shortage_birdflu_2025,hurricane_maria_2017"
S2="toyota_steel_explosion_2019,renesas_naka_plant_fire_2021,beirut_port_explosion_2020,covid_supply_disruption_2020"
S3="russia_ukraine_war_2022,renesas_earthquake_2016,port_los_angeles_backlog_2021,black_sea_grain_exit_2023"
S4="red_sea_crisis_2023,europe_energy_crisis_2022,taiwan_strait_crisis_2022,us_china_tariff_war_2018,uaw_auto_strike_2023,india_wheat_export_ban_2022"

for i in 1 2 3 4; do
  eval "EV=\$S$i"
  $PY pipeline/run_v2_preonset.py \
    --data-dir data/by_event_body \
    --base-url http://localhost:8080 \
    --out-dir "$OUT" \
    --events "$EV" \
    --no-thinking --skip-validate \
    > "$OUT/_shard$i.log" 2>&1 &
  echo "shard$i PID $! : $EV"
done
wait
echo "ALL SHARDS DONE $(date)"
