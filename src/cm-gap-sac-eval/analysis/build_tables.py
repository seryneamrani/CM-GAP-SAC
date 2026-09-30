"""
Build thesis tables from results.jsonl (scenario-based eval, v3+).

Auto-detects which methods are present in results.jsonl and generates the
appropriate tables:

  ALWAYS generated (single or multi-method):
    - main_by_scenario.md          Scenario × metrics (SR + safety + efficiency)
    - density_dynamicity.md        SR heatmap: density × pedestrian speed
    - failure_by_scenario.md       Failure-mode breakdown per scenario
    - safety_by_scenario.md        Safety-focused profile (CBF, clearances)
    - rl_diagnostics_by_scenario.md  Return, entropy, action smoothness

  GENERATED IF MULTIPLE METHODS PRESENT:
    - method_comparison.md         SR per (method × scenario) side-by-side

  GENERATED IF cm_gap_sac_full + ablation_no_cbf BOTH PRESENT:
    - cbf_ablation.md              Paired McNemar on same episodes

  GENERATED IF cm_gap_sac_full + nav2_dwb BOTH PRESENT:
    - nav2_comparison.md           RL vs classical (paired McNemar)

Usage:
    python build_tables.py --results outputs/results.jsonl --output-dir outputs/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats import wilson_ci, paired_binary_from_frames, format_rate_ci

# Zone assignment for spawn/goal points — same source of truth as train.py
try:
    from geometry import which_zone
    _GEOMETRY_AVAILABLE = True
except ImportError:
    _GEOMETRY_AVAILABLE = False
    def which_zone(x, y):
        return None


# ==============================================================================
# Load
# ==============================================================================
def load_results(path: Path) -> pd.DataFrame:
    """Flatten nested results.jsonl into a pandas DataFrame.

    Auto-derives spawn_zone and goal_zone from spawn_xy/goal_xy coordinates
    using geometry.which_zone() — enables intra-zone vs cross-zone breakdowns
    without requiring these fields in the JSONL itself.
    """
    records = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            spawn_xy = r.get("spawn_xy", [0.0, 0.0])
            goal_xy = r.get("goal_xy", [0.0, 0.0])
            spawn_zone = which_zone(spawn_xy[0], spawn_xy[1]) or "outside"
            goal_zone = which_zone(goal_xy[0], goal_xy[1]) or "outside"

            flat = {
                "episode_id":            r["episode_id"],
                "scenario_id":           r.get("scenario_id", "unknown"),
                "n_pedestrians":         r.get("n_pedestrians", 0),
                "pedestrian_speed_mps":  r.get("pedestrian_speed_mps", 0.0),
                "method":                r["method"],
                "checkpoint":            r.get("checkpoint"),
                "episode_seed":          r["episode_seed"],
                "wall_time_s":           r.get("wall_time_s"),
                "path_length_m":         r.get("path_length_m"),
                "spawn_x":               spawn_xy[0],
                "spawn_y":               spawn_xy[1],
                "goal_x":                goal_xy[0],
                "goal_y":                goal_xy[1],
                "spawn_zone":            spawn_zone,
                "goal_zone":             goal_zone,
                "is_intra_zone":         spawn_zone == goal_zone,
                "zone_transition":       f"{spawn_zone}→{goal_zone}",
            }
            flat.update({f"outcome_{k}": v for k, v in r["outcome"].items()})
            for k, v in r["metrics"].items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"m_{k}_{kk}"] = vv
                else:
                    flat[f"m_{k}"] = v
            records.append(flat)
    return pd.DataFrame(records)


def _mean(df: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in df.columns or df[col].isna().all():
        return default
    return float(df[col].mean())


def _median(df: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if col not in df.columns or df[col].isna().all():
        return default
    return float(df[col].replace([np.inf, -np.inf], np.nan).median())


def _fmt(v: float, spec: str = ".3f", na: str = "—") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return na
    return f"{v:{spec}}"


def _scenario_order(df: pd.DataFrame) -> list[str]:
    """Return scenario_ids in canonical order (S1 → S7 if named that way)."""
    ids = sorted(df["scenario_id"].unique())
    # Sort so S1, S2, ... come out in numeric order even with letter prefix
    return sorted(ids, key=lambda s: (s[0] if s else "", int(s[1:].split("_")[0]) if s[1:2].isdigit() else 999))


# ==============================================================================
# Table 1 — Main results by scenario (compact, all key metrics)
# ==============================================================================
def main_by_scenario(df: pd.DataFrame) -> str:
    lines = ["# Table 1 — Main Results by Scenario", ""]
    lines.append("Success Rate reported with Wilson 95% CI. One row per (method, scenario) combination.")
    lines.append("")
    lines.append("| Method | Scenario | n_ped | speed (m/s) | N | SR (95% CI) | SPL | PLR | Coll_Ped | Coll_Static | Timeout | Freeze |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            g = m_df[m_df["scenario_id"] == sid]
            if len(g) == 0:
                continue
            n = len(g)
            s = int(g["outcome_success"].sum())
            sr, lo, hi = wilson_ci(s, n)
            n_ped = int(g["n_pedestrians"].iloc[0])
            speed = float(g["pedestrian_speed_mps"].iloc[0])
            lines.append(
                f"| {method} | {sid} | {n_ped} | {speed} | {n} | "
                f"{format_rate_ci(sr, lo, hi)} | "
                f"{_fmt(_mean(g, 'm_spl'), '.3f')} | "
                f"{_fmt(_median(g, 'm_path_length_ratio'), '.2f')} | "
                f"{100*_mean(g, 'outcome_collision_pedestrian'):.1f}% | "
                f"{100*_mean(g, 'outcome_collision_static'):.1f}% | "
                f"{100*_mean(g, 'outcome_timeout'):.1f}% | "
                f"{100*_mean(g, 'outcome_freeze_at_start'):.1f}% |"
            )
    return "\n".join(lines)


# ==============================================================================
# Table 2 — Density × Dynamicity heatmap (SR per method)
# ==============================================================================
def density_dynamicity_heatmap(df: pd.DataFrame) -> str:
    lines = ["# Table 2 — SR Heatmap: Density × Dynamicity", ""]
    lines.append("Rows = pedestrian density, columns = pedestrian speed.")
    lines.append("Cell value = Success Rate with Wilson 95% CI.")
    lines.append("")

    densities = sorted(df["n_pedestrians"].unique())
    speeds = sorted(df["pedestrian_speed_mps"].unique())

    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        lines.append(f"## {method}\n")
        header = "| Density \\ Speed | " + " | ".join(f"{s} m/s" for s in speeds) + " |"
        sep = "|---|" + "|".join("---" for _ in speeds) + "|"
        lines.append(header)
        lines.append(sep)
        for d in densities:
            row = [f"{d} peds"]
            for s in speeds:
                cell_df = m_df[
                    (m_df["n_pedestrians"] == d) &
                    (m_df["pedestrian_speed_mps"] == s)
                ]
                if len(cell_df) == 0:
                    row.append("—")
                else:
                    n = len(cell_df)
                    succ = int(cell_df["outcome_success"].sum())
                    sr, lo, hi = wilson_ci(succ, n)
                    row.append(f"{100*sr:.1f}% [{100*lo:.0f}, {100*hi:.0f}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Table 3 — Failure taxonomy by scenario
# ==============================================================================
def failure_by_scenario(df: pd.DataFrame) -> str:
    lines = ["# Table 3 — Failure Mode Taxonomy by Scenario", ""]
    lines.append("Per-scenario breakdown of episode outcomes. Rows may sum to slightly")
    lines.append("more than 100% because an episode can be flagged for multiple failure")
    lines.append("modes (e.g., freeze then timeout).")
    lines.append("")
    lines.append("| Method | Scenario | N | Success | Coll_Ped | Coll_Static | Timeout | Freeze |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            g = m_df[m_df["scenario_id"] == sid]
            if len(g) == 0:
                continue
            lines.append(
                f"| {method} | {sid} | {len(g)} | "
                f"{100*_mean(g, 'outcome_success'):.1f}% | "
                f"{100*_mean(g, 'outcome_collision_pedestrian'):.1f}% | "
                f"{100*_mean(g, 'outcome_collision_static'):.1f}% | "
                f"{100*_mean(g, 'outcome_timeout'):.1f}% | "
                f"{100*_mean(g, 'outcome_freeze_at_start'):.1f}% |"
            )
    return "\n".join(lines)


# ==============================================================================
# Table 4 — Safety profile by scenario (CBF, clearances, social)
# ==============================================================================
def safety_by_scenario(df: pd.DataFrame) -> str:
    lines = ["# Table 4 — Safety Profile by Scenario", ""]
    lines.append("Detailed safety metrics — CBF intervention/infeasibility, clearances, social discomfort.")
    lines.append("Clearances reported as median across episodes (min per-episode).")
    lines.append("")
    lines.append("| Method | Scenario | Min_Clr_Ped (m) | Min_Clr_Static (m) | IR | CIR | CM | PSV | SDS |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            g = m_df[m_df["scenario_id"] == sid]
            if len(g) == 0:
                continue
            lines.append(
                f"| {method} | {sid} | "
                f"{_fmt(_median(g, 'm_min_clearance_pedestrian_m'), '.2f')} | "
                f"{_fmt(_median(g, 'm_min_clearance_static_m'), '.2f')} | "
                f"{100*_mean(g, 'm_cbf_intervention_rate', 0.0):.1f}% | "
                f"{100*_mean(g, 'm_cbf_infeasibility_rate', 0.0):.2f}% | "
                f"{_fmt(_mean(g, 'm_cbf_correction_magnitude_mean', 0.0), '.3f')} | "
                f"{100*_mean(g, 'm_personal_space_violation_rate'):.1f}% | "
                f"{_fmt(_mean(g, 'm_social_discomfort_score'), '.3f')} |"
            )
    lines.append("")
    lines.append("IR = CBF Intervention Rate. CIR = CBF Infeasibility Rate. CM = CBF Correction Magnitude.")
    lines.append("PSV = Personal Space Violation Rate. SDS = Social Discomfort Score.")
    return "\n".join(lines)


# ==============================================================================
# Table 5 — RL diagnostics by scenario
# ==============================================================================
def rl_diagnostics_by_scenario(df: pd.DataFrame) -> str:
    lines = ["# Table 5 — RL Diagnostics by Scenario", ""]
    lines.append("Reward and policy behavior. TFM = Time-to-First-Motion (custom).")
    lines.append("AS = Action Smoothness. PS = Path Smoothness. GPE = Goal Progress Efficiency.")
    lines.append("")
    lines.append("| Method | Scenario | Ep Return | Ep Length | Entropy | TFM (s) | AS | PS (rad/m) | GPE |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            g = m_df[m_df["scenario_id"] == sid]
            if len(g) == 0:
                continue
            lines.append(
                f"| {method} | {sid} | "
                f"{_fmt(_mean(g, 'm_episode_return'), '+.2f')} | "
                f"{_fmt(_median(g, 'm_episode_length_steps'), '.0f')} | "
                f"{_fmt(_mean(g, 'm_policy_entropy_mean'), '.3f')} | "
                f"{_fmt(_median(g, 'm_time_to_first_motion_s'), '.2f')} | "
                f"{_fmt(_mean(g, 'm_action_smoothness'), '.3f')} | "
                f"{_fmt(_mean(g, 'm_path_smoothness_rad_per_m'), '.2f')} | "
                f"{_fmt(_mean(g, 'm_goal_progress_efficiency'), '.3f')} |"
            )
    return "\n".join(lines)


# ==============================================================================
# Conditional Table — Method comparison (only if ≥2 methods)
# ==============================================================================
def method_comparison(df: pd.DataFrame) -> str:
    lines = ["# Table 6 — Method Comparison (SR × Scenario)", ""]
    methods = sorted(df["method"].unique())
    lines.append(f"Methods present: {', '.join(methods)}")
    lines.append("")

    header = "| Scenario | " + " | ".join(f"SR — {m}" for m in methods) + " |"
    sep = "|---|" + "|".join("---" for _ in methods) + "|"
    lines.append(header)
    lines.append(sep)

    for sid in _scenario_order(df):
        row = [sid]
        for m in methods:
            g = df[(df["method"] == m) & (df["scenario_id"] == sid)]
            if len(g) == 0:
                row.append("—")
            else:
                n = len(g)
                s = int(g["outcome_success"].sum())
                sr, lo, hi = wilson_ci(s, n)
                row.append(format_rate_ci(sr, lo, hi))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ==============================================================================
# Conditional Table — CBF ablation (only if full + no-CBF both present)
# ==============================================================================
def cbf_ablation(df: pd.DataFrame) -> str:
    lines = ["# Table 7 — CBF Ablation: cm_gap_sac_full vs ablation_no_cbf", ""]
    lines.append("Paired McNemar test on same episode_seed. Δ = (no-CBF − full).")
    lines.append("")
    lines.append("| Scenario | N | SR (full) | SR (no-CBF) | Δ SR | Coll_Ped Δ | Clr_Ped Δ | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|---|")

    full = df[df["method"] == "cm_gap_sac_full"]
    ab = df[df["method"] == "ablation_no_cbf"]

    for sid in _scenario_order(df):
        f_g = full[full["scenario_id"] == sid]
        a_g = ab[ab["scenario_id"] == sid]
        if len(f_g) == 0 or len(a_g) == 0:
            continue
        n = min(len(f_g), len(a_g))
        sr_f = _mean(f_g, "outcome_success")
        sr_a = _mean(a_g, "outcome_success")
        cp_f = _mean(f_g, "outcome_collision_pedestrian")
        cp_a = _mean(a_g, "outcome_collision_pedestrian")
        clr_f = _median(f_g, "m_min_clearance_pedestrian_m")
        clr_a = _median(a_g, "m_min_clearance_pedestrian_m")
        mcn = paired_binary_from_frames(f_g, a_g)
        star = "*" if mcn["significant"] else ""
        lines.append(
            f"| {sid} | {n} | "
            f"{100*sr_f:.1f}% | {100*sr_a:.1f}% | "
            f"{100*(sr_a - sr_f):+.1f} pp | "
            f"{100*(cp_a - cp_f):+.1f} pp | "
            f"{_fmt(clr_a - clr_f, '+.2f')}m | "
            f"{mcn['p_value']:.3f}{star} |"
        )
    lines.append("")
    lines.append("`*` = p < 0.05.")
    return "\n".join(lines)


# ==============================================================================
# Conditional Table — RL vs Nav2 baseline (only if full + nav2_dwb both present)
# ==============================================================================
def nav2_comparison(df: pd.DataFrame) -> str:
    lines = ["# Table 8 — RL vs Classical: cm_gap_sac_full vs nav2_dwb", ""]
    lines.append("Answers the 'why not just Nav2?' defense question. Paired McNemar.")
    lines.append("")
    lines.append("| Scenario | N | SR (RL) | SR (Nav2) | Δ SR | SPL Δ | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|")

    full = df[df["method"] == "cm_gap_sac_full"]
    nav = df[df["method"] == "nav2_dwb"]

    for sid in _scenario_order(df):
        f_g = full[full["scenario_id"] == sid]
        n_g = nav[nav["scenario_id"] == sid]
        if len(f_g) == 0 or len(n_g) == 0:
            continue
        n = min(len(f_g), len(n_g))
        sr_f = _mean(f_g, "outcome_success")
        sr_n = _mean(n_g, "outcome_success")
        spl_f = _mean(f_g, "m_spl")
        spl_n = _mean(n_g, "m_spl")
        mcn = paired_binary_from_frames(f_g, n_g)
        star = "*" if mcn["significant"] else ""
        lines.append(
            f"| {sid} | {n} | "
            f"{100*sr_f:.1f}% | {100*sr_n:.1f}% | "
            f"{100*(sr_f - sr_n):+.1f} pp | "
            f"{_fmt(spl_f - spl_n, '+.3f')} | "
            f"{mcn['p_value']:.3f}{star} |"
        )
    return "\n".join(lines)


# ==============================================================================
# Zone analysis — intra vs cross-zone breakdown (derived from spawn_xy/goal_xy)
# ==============================================================================
def zone_analysis(df: pd.DataFrame) -> str:
    lines = ["# Table — Zone Analysis (intra vs cross-zone breakdown)", ""]
    lines.append("Spawn/goal zones are auto-derived from spawn_xy/goal_xy at analysis")
    lines.append("time using geometry.which_zone(). Episodes with spawn or goal outside")
    lines.append("any zone box are labeled 'outside'.")
    lines.append("")

    # ---- Part A: SR by spawn zone (regardless of goal) ----
    lines.append("## A) Success Rate by spawn zone")
    lines.append("")
    lines.append("| Method | Scenario | Spawn Zone | N | SR (95% CI) | Coll_Ped | Coll_Static |")
    lines.append("|---|---|---|---|---|---|---|")
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            s_df = m_df[m_df["scenario_id"] == sid]
            for zone in sorted(s_df["spawn_zone"].unique()):
                g = s_df[s_df["spawn_zone"] == zone]
                if len(g) < 3:  # skip near-empty cells
                    continue
                n = len(g)
                s = int(g["outcome_success"].sum())
                sr, lo, hi = wilson_ci(s, n)
                lines.append(
                    f"| {method} | {sid} | {zone} | {n} | "
                    f"{format_rate_ci(sr, lo, hi)} | "
                    f"{100*_mean(g, 'outcome_collision_pedestrian'):.1f}% | "
                    f"{100*_mean(g, 'outcome_collision_static'):.1f}% |"
                )
    lines.append("")

    # ---- Part B: Intra-zone vs cross-zone ----
    lines.append("## B) Intra-zone vs Cross-zone")
    lines.append("")
    lines.append("| Method | Scenario | Type | N | SR (95% CI) | Coll_Ped | Coll_Static | SPL |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        for sid in _scenario_order(m_df):
            s_df = m_df[m_df["scenario_id"] == sid]
            for kind, mask in [("intra", s_df["is_intra_zone"]),
                               ("cross", ~s_df["is_intra_zone"])]:
                g = s_df[mask]
                if len(g) < 3:
                    continue
                n = len(g)
                s = int(g["outcome_success"].sum())
                sr, lo, hi = wilson_ci(s, n)
                lines.append(
                    f"| {method} | {sid} | {kind} | {n} | "
                    f"{format_rate_ci(sr, lo, hi)} | "
                    f"{100*_mean(g, 'outcome_collision_pedestrian'):.1f}% | "
                    f"{100*_mean(g, 'outcome_collision_static'):.1f}% | "
                    f"{_fmt(_mean(g, 'm_spl'), '.3f')} |"
                )
    lines.append("")

    # ---- Part C: Top-N specific zone transitions ----
    lines.append("## C) Zone transitions (top-10 most frequent per method)")
    lines.append("")
    lines.append("| Method | Transition | N | SR (95% CI) | Coll_Ped |")
    lines.append("|---|---|---|---|---|")
    for method in sorted(df["method"].unique()):
        m_df = df[df["method"] == method]
        top_transitions = m_df["zone_transition"].value_counts().head(10)
        for transition, count in top_transitions.items():
            g = m_df[m_df["zone_transition"] == transition]
            n = len(g)
            s = int(g["outcome_success"].sum())
            sr, lo, hi = wilson_ci(s, n)
            lines.append(
                f"| {method} | {transition} | {n} | "
                f"{format_rate_ci(sr, lo, hi)} | "
                f"{100*_mean(g, 'outcome_collision_pedestrian'):.1f}% |"
            )
    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.results)

    methods = sorted(df["method"].unique())
    scenarios = _scenario_order(df)
    print(f"Loaded {len(df)} episodes from {args.results}")
    print(f"Methods present:   {methods}")
    print(f"Scenarios present: {scenarios}")
    print()

    # Always generate these
    tables = {
        "main_by_scenario.md":         main_by_scenario(df),
        "density_dynamicity.md":       density_dynamicity_heatmap(df),
        "failure_by_scenario.md":      failure_by_scenario(df),
        "safety_by_scenario.md":       safety_by_scenario(df),
        "rl_diagnostics_by_scenario.md": rl_diagnostics_by_scenario(df),
        "zone_analysis.md":            zone_analysis(df),
    }

    # Conditional tables — only meaningful if multiple methods
    if len(methods) >= 2:
        tables["method_comparison.md"] = method_comparison(df)

    if "cm_gap_sac_full" in methods and "ablation_no_cbf" in methods:
        tables["cbf_ablation.md"] = cbf_ablation(df)

    if "cm_gap_sac_full" in methods and "nav2_dwb" in methods:
        tables["nav2_comparison.md"] = nav2_comparison(df)

    for name, content in tables.items():
        out = args.output_dir / name
        out.write_text(content)
        print(f"  wrote {out}  ({len(content.splitlines())} lines)")

    print(f"\nDone. {len(tables)} tables generated.")
    if len(methods) == 1:
        print(f"Note: only 1 method present ({methods[0]}). Cross-method")
        print(f"comparison tables will auto-generate when you rerun after")
        print(f"evaluating additional methods.")


if __name__ == "__main__":
    main()
