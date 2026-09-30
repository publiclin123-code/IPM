# True-positive evidence audit

Signals from `results/v2`, window 180 days, loose matcher.

`company` = a company name matched (strong).
`commodity_only` / `geo_only` = only a generic string matched (weak).

| event | forward | TP | company | geo+commodity | commodity_only | geo_only | weak % |
|---|---|---|---|---|---|---|---|
| red_sea_crisis_2023 | 8 | 7 | 0 | 3 | 4 | 0 | 57% |
| us_chip_export_controls_2022 | 19 | 14 | 3 | 8 | 2 | 1 | 21% |
| covid_supply_disruption_2020 | 2 | 2 | 0 | 2 | 0 | 0 | 0% |
| renesas_earthquake_2016 | 9 | 0 | 0 | 0 | 0 | 0 | 0% |
| port_los_angeles_backlog_2021 | 11 | 5 | 0 | 0 | 5 | 0 | 100% |
| toyota_steel_explosion_2019 | 32 | 27 | 1 | 0 | 26 | 0 | 96% |
| europe_energy_crisis_2022 | 24 | 24 | 2 | 19 | 3 | 0 | 12% |
| renesas_naka_plant_fire_2021 | 0 | 0 | 0 | 0 | 0 | 0 | 0% |
| taiwan_strait_crisis_2022 | 14 | 13 | 8 | 5 | 0 | 0 | 0% |
| us_china_tariff_war_2018 | 35 | 34 | 0 | 24 | 0 | 10 | 29% |
| hurricane_maria_2017 | 12 | 12 | 0 | 10 | 1 | 1 | 17% |
| uaw_auto_strike_2023 | 12 | 12 | 10 | 0 | 0 | 2 | 17% |
| black_sea_grain_exit_2023 | 24 | 23 | 1 | 22 | 0 | 0 | 0% |
| india_wheat_export_ban_2022 | 6 | 6 | 0 | 2 | 4 | 0 | 67% |
| egg_shortage_birdflu_2025 | 1 | 1 | 0 | 0 | 1 | 0 | 100% |
| russia_ukraine_war_2022 | 28 | 27 | 0 | 18 | 0 | 9 | 33% |
| beirut_port_explosion_2020 | 0 | 0 | 0 | 0 | 0 | 0 | 0% |

**Total TP 207**, of which weak (no company evidence): 69 (33.3%).