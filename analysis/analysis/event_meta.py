"""17-event shared metadata: event -> type (slow_burn/creeping/sudden) + short label.

Single source of truth for figure scripts. Type classification is a priori
(before seeing LLM scores) and must match Table 1 in the paper.

RETRACTION 2026-09-18: `toyota_steel_explosion_2019` was removed from the ground
truth; the event could not be corroborated from any source and its recorded hits
were US steel-tariff articles reached via the commodity string "steel". Evidence
in `pipeline/verify_toyota_event.py`, reasoning in draft/ipm/README.md. The event
is archived in `validation/gt_excluded.json`; do not re-add it without sources.

Counts are now slow_burn=9, creeping=2, sudden=6 (17 total).
"""
from __future__ import annotations

# eid -> a priori type. slow_burn=9, creeping=2, sudden=6 (17 total).
EVENT_TYPES = {
    # slow-burn / policy (9)
    "red_sea_crisis_2023": "slow_burn",
    "us_chip_export_controls_2022": "slow_burn",
    "us_china_tariff_war_2018": "slow_burn",
    "taiwan_strait_crisis_2022": "slow_burn",
    "russia_ukraine_war_2022": "slow_burn",
    "black_sea_grain_exit_2023": "slow_burn",
    "uaw_auto_strike_2023": "slow_burn",
    "india_wheat_export_ban_2022": "slow_burn",
    "egg_shortage_birdflu_2025": "slow_burn",
    # creeping (2)
    "port_los_angeles_backlog_2021": "creeping",
    "europe_energy_crisis_2022": "creeping",
    # sudden (6) -- toyota_steel_explosion_2019 retracted 2026-09-18
    "suez_ever_given_2021": "sudden",
    "beirut_port_explosion_2020": "sudden",
    "renesas_earthquake_2016": "sudden",
    "covid_supply_disruption_2020": "sudden",
    "hurricane_maria_2017": "sudden",
    "renesas_naka_plant_fire_2021": "sudden",
}

# eid -> short display label (for figure lane labels).
EVENT_SHORT = {
    "red_sea_crisis_2023": "Red Sea crisis",
    "us_chip_export_controls_2022": "U.S. chip controls",
    "us_china_tariff_war_2018": "U.S.–China tariff war",
    "taiwan_strait_crisis_2022": "Taiwan Strait",
    "russia_ukraine_war_2022": "Russia–Ukraine war",
    "black_sea_grain_exit_2023": "Black Sea grain",
    "uaw_auto_strike_2023": "UAW strike",
    "india_wheat_export_ban_2022": "India wheat ban",
    "egg_shortage_birdflu_2025": "Egg shortage",
    "port_los_angeles_backlog_2021": "LA/LB port backlog",
    "europe_energy_crisis_2022": "European energy",
    "suez_ever_given_2021": "Suez (Ever Given)",
    "beirut_port_explosion_2020": "Beirut explosion",
    "renesas_earthquake_2016": "Renesas quake",
    "covid_supply_disruption_2020": "COVID-19",
    "hurricane_maria_2017": "Hurricane Maria",
    "renesas_naka_plant_fire_2021": "Renesas fire",
}

# fig3 swimlane / fig8 cumulative expect {eid: (short, type)}.
EVENT_META = {
    eid: (EVENT_SHORT[eid], EVENT_TYPES[eid]) for eid in EVENT_TYPES
}
