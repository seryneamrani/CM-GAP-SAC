# Table — Zone Analysis (intra vs cross-zone breakdown)

Spawn/goal zones are auto-derived from spawn_xy/goal_xy at analysis
time using geometry.which_zone(). Episodes with spawn or goal outside
any zone box are labeled 'outside'.

## A) Success Rate by spawn zone

| Method | Scenario | Spawn Zone | N | SR (95% CI) | Coll_Ped | Coll_Static |
|---|---|---|---|---|---|---|
| cm_gap_sac_full | S1_static_only | outside | 86 | 82.6% [73.2, 89.1] | 0.0% | 5.8% |

## B) Intra-zone vs Cross-zone

| Method | Scenario | Type | N | SR (95% CI) | Coll_Ped | Coll_Static | SPL |
|---|---|---|---|---|---|---|---|
| cm_gap_sac_full | S1_static_only | intra | 86 | 82.6% [73.2, 89.1] | 0.0% | 5.8% | 0.819 |

## C) Zone transitions (top-10 most frequent per method)

| Method | Transition | N | SR (95% CI) | Coll_Ped |
|---|---|---|---|---|
| cm_gap_sac_full | outside→outside | 86 | 82.6% [73.2, 89.1] | 0.0% |