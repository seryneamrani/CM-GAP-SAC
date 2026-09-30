# Table — Zone Analysis (intra vs cross-zone breakdown)

Spawn/goal zones are auto-derived from spawn_xy/goal_xy at analysis
time using geometry.which_zone(). Episodes with spawn or goal outside
any zone box are labeled 'outside'.

## A) Success Rate by spawn zone

| Method | Scenario | Spawn Zone | N | SR (95% CI) | Coll_Ped | Coll_Static |
|---|---|---|---|---|---|---|
| cm_gap_sac_full | S1_static_only | outside | 500 | 83.2% [79.7, 86.2] | 0.0% | 12.4% |
| cm_gap_sac_full | S2_density3_low | outside | 500 | 79.6% [75.8, 82.9] | 4.0% | 10.2% |
| cm_gap_sac_full | S4_density5_low | outside | 500 | 77.8% [74.0, 81.2] | 2.4% | 14.2% |
| cm_gap_sac_full | S6_density7_low | outside | 499 | 79.4% [75.6, 82.7] | 2.6% | 12.6% |
| cm_gap_sac_full | S7_density7_high | outside | 499 | 73.3% [69.3, 77.0] | 1.6% | 18.4% |
| nav2_dwb | S1_static_only | outside | 500 | 94.0% [91.6, 95.8] | 0.0% | 0.4% |
| nav2_dwb | S7_density7_high | outside | 500 | 72.0% [67.9, 75.8] | 9.6% | 18.4% |

## B) Intra-zone vs Cross-zone

| Method | Scenario | Type | N | SR (95% CI) | Coll_Ped | Coll_Static | SPL |
|---|---|---|---|---|---|---|---|
| cm_gap_sac_full | S1_static_only | intra | 500 | 83.2% [79.7, 86.2] | 0.0% | 12.4% | 0.830 |
| cm_gap_sac_full | S2_density3_low | intra | 500 | 79.6% [75.8, 82.9] | 4.0% | 10.2% | 0.792 |
| cm_gap_sac_full | S4_density5_low | intra | 500 | 77.8% [74.0, 81.2] | 2.4% | 14.2% | 0.774 |
| cm_gap_sac_full | S6_density7_low | intra | 499 | 79.4% [75.6, 82.7] | 2.6% | 12.6% | 0.791 |
| cm_gap_sac_full | S7_density7_high | intra | 499 | 73.3% [69.3, 77.0] | 1.6% | 18.4% | 0.731 |
| nav2_dwb | S1_static_only | intra | 500 | 94.0% [91.6, 95.8] | 0.0% | 0.4% | 0.933 |
| nav2_dwb | S7_density7_high | intra | 500 | 72.0% [67.9, 75.8] | 9.6% | 18.4% | 0.720 |

## C) Zone transitions (top-10 most frequent per method)

| Method | Transition | N | SR (95% CI) | Coll_Ped |
|---|---|---|---|---|
| cm_gap_sac_full | outside→outside | 2498 | 78.7% [77.0, 80.2] | 2.1% |
| nav2_dwb | outside→outside | 1000 | 83.0% [80.5, 85.2] | 4.8% |