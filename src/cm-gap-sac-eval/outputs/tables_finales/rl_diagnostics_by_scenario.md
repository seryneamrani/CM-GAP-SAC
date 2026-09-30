# Table 5 — RL Diagnostics by Scenario

Reward and policy behavior. TFM = Time-to-First-Motion (custom).
AS = Action Smoothness. PS = Path Smoothness. GPE = Goal Progress Efficiency.

| Method | Scenario | Ep Return | Ep Length | Entropy | TFM (s) | AS | PS (rad/m) | GPE |
|---|---|---|---|---|---|---|---|---|
| cm_gap_sac_full | S1_static_only | -34.77 | 150 | 0.000 | 0.20 | 0.067 | 0.79 | 0.000 |
| cm_gap_sac_full | S2_density3_low | -40.19 | 140 | 0.240 | 0.20 | 0.072 | 1.01 | 0.000 |
| cm_gap_sac_full | S4_density5_low | -45.91 | 146 | 0.238 | 0.20 | 0.075 | 1.07 | 0.000 |
| cm_gap_sac_full | S6_density7_low | -41.99 | 138 | 0.253 | 0.20 | 0.077 | 0.99 | 0.000 |
| cm_gap_sac_full | S7_density7_high | -44.85 | 132 | 0.287 | 0.20 | 0.081 | 1.06 | 0.000 |
| nav2_dwb | S1_static_only | +0.00 | 140 | — | 0.03 | 0.052 | 0.44 | 0.956 |
| nav2_dwb | S7_density7_high | +0.00 | 121 | — | 0.03 | 0.057 | 0.45 | 0.843 |