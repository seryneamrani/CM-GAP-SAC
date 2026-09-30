"""
analyse.py — Genere toutes les analyses (eval + training) pour CM-GAP-SAC.

Schemas attendus (colonnes verifiees) :

  --eval : un .jsonl (ou dossier de .jsonl) avec un objet JSON par episode,
           champs : method, scenario_group/scenario_id, zone_type, spawn_zone,
           goal_zone, n_pedestrians, pedestrian_speed_mps, outcome{success,
           collision_pedestrian, collision_static, timeout, freeze_at_start},
           metrics{...}, path_length_m, wall_time_s.

  --training : episodes_combined_clean.csv avec les colonnes :
           episode, step, outcome, collision_type, spawn_zone, spawn_x, spawn_y,
           goal_x, goal_y, d_goal_init, n_waypoints, waypoints_reached, reward,
           steps, success_roll, collision_roll, collision_ped_roll,
           collision_static_roll, shield_frozen_roll, mean_reward_roll,
           cbf_omega_decay_ep, sps, __source_file, d_min_ped_terminal,
           min_lidar_terminal, collision_source

Usage:
    python analyse.py \
        --eval src/cm-gap-sac-eval/outputs/results_backup_20260727_1157.jsonl \
        --training runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_combined_clean.csv \
        --outdir analyse

    python analyse.py                      # utilise les chemins par defaut ci-dessous
    python analyse.py --no-plots            # ne genere que les CSV, pas les figures PDF

Sorties :
    analyse/eval/
        task_difficulty.csv, zone_transition.csv, clearance_by_outcome.csv,
        throughput.txt, shield_behavior.csv, failure_modes.csv,
        zone_pair_heatmap_S6.csv
        figures/*.pdf
    analyse/training/
        learning_curves.csv, outcome_distribution.csv,
        collision_type_breakdown.csv, collision_source_breakdown.csv,
        task_difficulty_training.csv, waypoint_completion.csv,
        spawn_zone_distribution.csv, shield_cbf_trend.csv,
        terminal_clearance_by_outcome.csv, source_file_summary.csv,
        throughput_training.txt, training_summary.txt
        figures/*.pdf
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTROL_PERIOD_S = 0.05  # 20 Hz

DEFAULT_EVAL_PATH = "src/cm-gap-sac-eval/outputs/results_backup_20260727_1157.jsonl"
DEFAULT_TRAINING_PATH = "runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_combined_clean.csv"
DEFAULT_OUTDIR = "analyse"

DIST_BINS = [0, 2, 4, 6, 8, 100]
DIST_LABELS = ["[0,2]", "[2,4]", "[4,6]", "[6,8]", "[8,+]"]


# ---------------------------------------------------------------------------
# UTILITAIRES FIGURES (vectoriel, pret pour inclusion LaTeX/XeLaTeX)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig, outdir, name):
    figdir = Path(outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure -> {path}")


# ---------------------------------------------------------------------------
# PARTIE EVAL (results_backup*.jsonl)
# ---------------------------------------------------------------------------

def load_all_episodes(eval_path):
    """Charge un fichier JSONL (ou un dossier de .jsonl) dans un DataFrame."""
    eval_path = Path(eval_path)
    if eval_path.is_dir():
        paths = glob.glob(str(eval_path / "**/*.jsonl"), recursive=True)
    else:
        paths = [str(eval_path)]

    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                flat = {
                    "method": rec.get("method"),
                    "scenario": rec.get("scenario_group") or rec.get("scenario_id"),
                    "zone_type": rec.get("zone_type"),
                    "spawn_zone": rec.get("spawn_zone"),
                    "goal_zone": rec.get("goal_zone"),
                    "n_ped": rec.get("n_pedestrians"),
                    "ped_speed": rec.get("pedestrian_speed_mps"),
                    "success": rec["outcome"]["success"],
                    "coll_ped": rec["outcome"]["collision_pedestrian"],
                    "coll_stat": rec["outcome"]["collision_static"],
                    "timeout": rec["outcome"]["timeout"],
                    "freeze": rec["outcome"]["freeze_at_start"],
                    "wall_time_s": rec.get("wall_time_s"),
                    "path_length_m": rec.get("path_length_m"),
                }
                m = rec.get("metrics", {})
                for k, v in m.items():
                    if isinstance(v, (int, float)) or v is None:
                        flat[k] = v if v != float("inf") else np.nan
                rows.append(flat)
    return pd.DataFrame(rows)


def outcome_label(row):
    if row["success"]:
        return "success"
    if row["coll_stat"]:
        return "static_collision"
    if row["coll_ped"]:
        return "pedestrian_collision"
    if row["timeout"]:
        return "timeout"
    if row["freeze"]:
        return "freeze"
    return "other"


def task_difficulty(df, method="cm_gap_sac_full",
                     scenarios=("S2_density3_low", "S4_density5_low",
                                "S6_density7_low", "S7_density7_high")):
    sub = df[(df.method == method) & (df.scenario.isin(scenarios))].copy()
    sub["bin"] = pd.cut(sub["initial_goal_distance_m"], bins=DIST_BINS, labels=DIST_LABELS)
    grp = sub.groupby("bin", observed=True)["success"].agg(["count", "mean"])
    grp.columns = ["N", "SR"]
    grp["SR"] = (grp["SR"] * 100).round(1)
    return grp


def zone_transition(df, method="cm_gap_sac_full"):
    sub = df[df.method == method].copy()
    grp = sub.groupby(["scenario", "zone_type"])["success"].agg(["count", "mean"]).reset_index()
    grp["mean"] = (grp["mean"] * 100).round(1)
    piv = grp.pivot(index="scenario", columns="zone_type", values="mean")
    piv["gap_pp"] = (piv.get("intra", 0) - piv.get("cross", 0)).round(1)
    return piv


def clearance_by_outcome(df, method="cm_gap_sac_full", scenario="S6_density7_low"):
    sub = df[(df.method == method) & (df.scenario == scenario)].copy()
    sub["outcome_label"] = sub.apply(outcome_label, axis=1)
    grp = sub.groupby("outcome_label")["min_clearance_static_m"].agg(
        ["count", lambda x: x.median(), lambda x: x.quantile(0.1)]
    )
    grp.columns = ["N", "median_m", "p10_m"]
    grp = grp.round(2)
    return grp


def throughput(df, method="cm_gap_sac_full"):
    sub = df[(df.method == method) & df.wall_time_s.notna()].copy()
    mean_wall = sub["wall_time_s"].mean()
    mean_steps = sub["episode_length_steps"].mean()
    total_sim_time = mean_steps * CONTROL_PERIOD_S
    rtf = total_sim_time / mean_wall
    sps = mean_steps / mean_wall
    return {
        "mean_wall_time_s": round(mean_wall, 2),
        "mean_episode_length_steps": round(mean_steps, 1),
        "control_period_s": CONTROL_PERIOD_S,
        "mean_sim_time_s": round(total_sim_time, 2),
        "real_time_factor": round(rtf, 2),
        "steps_per_sec": round(sps, 1),
    }


def shield_behavior(df, method="cm_gap_sac_full"):
    sub = df[df.method == method].copy()
    grp = sub.groupby("scenario").agg(
        median_IR=("cbf_intervention_rate", lambda x: round(x.median() * 100, 1)),
        mean_CM=("cbf_correction_magnitude_mean", lambda x: round(x.mean(), 3)),
        infeas_rate=("cbf_infeasibility_rate", lambda x: round(x.mean() * 100, 3)),
    )
    return grp


def failure_mode_by_scenario(df):
    df = df.copy()
    df["outcome_label"] = df.apply(outcome_label, axis=1)
    grp = df.groupby(["method", "scenario", "outcome_label"]).size().unstack(fill_value=0)
    tot = grp.sum(axis=1)
    grp_pct = (grp.div(tot, axis=0) * 100).round(1)
    return grp_pct


def zone_pair_heatmap(df, method="cm_gap_sac_full", scenario=None):
    sub = df[df.method == method].copy()
    if scenario:
        sub = sub[sub.scenario == scenario]
    grp = sub.groupby(["spawn_zone", "goal_zone"])["success"].agg(["count", "mean"]).reset_index()
    grp["SR_pct"] = (grp["mean"] * 100).round(1)
    return grp.pivot(index="spawn_zone", columns="goal_zone", values="SR_pct")


# --- figures eval ---

def plot_task_difficulty(t1, outdir):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(t1.index.astype(str), t1["SR"], color="#2b6cb0")
    for i, (n, sr) in enumerate(zip(t1["N"], t1["SR"])):
        ax.text(i, sr + 1.5, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xlabel("Distance initiale au but (m)")
    ax.set_ylabel("Taux de succes (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SR par tranche de distance initiale au but")
    save_fig(fig, outdir, "task_difficulty")


def plot_zone_transition(t2, outdir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(t2.index))
    width = 0.35
    if "intra" in t2.columns:
        ax.bar(x - width / 2, t2["intra"], width, label="intra-zone", color="#2b6cb0")
    if "cross" in t2.columns:
        ax.bar(x + width / 2, t2["cross"], width, label="cross-zone", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(t2.index, rotation=30, ha="right")
    ax.set_ylabel("Taux de succes (%)")
    ax.set_title("SR intra-zone vs cross-zone par scenario")
    ax.legend()
    save_fig(fig, outdir, "zone_transition")


def plot_failure_modes(t6, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(t6))
    x_labels = [f"{m}\n{s}" for m, s in t6.index]
    colors = {"success": "#2f855a", "static_collision": "#c05621",
              "pedestrian_collision": "#c53030", "timeout": "#805ad5",
              "freeze": "#718096", "other": "#a0aec0"}
    for col in t6.columns:
        vals = t6[col].values
        ax.bar(x_labels, vals, bottom=bottom, label=col, color=colors.get(col, None))
        bottom += vals
    ax.set_ylabel("Part des episodes (%)")
    ax.set_title("Repartition des modes d'echec par scenario")
    ax.legend(fontsize=7, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    save_fig(fig, outdir, "failure_modes")


def plot_shield_behavior(t5, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(t5.index.astype(str), t5["median_IR"], color="#2b6cb0")
    axes[0].set_ylabel("Taux d'intervention CBF median (%)")
    axes[0].set_title("Intervention du shield")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)

    axes[1].bar(t5.index.astype(str), t5["mean_CM"], color="#c05621")
    axes[1].set_ylabel("Magnitude moyenne de correction")
    axes[1].set_title("Amplitude des corrections CBF")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    save_fig(fig, outdir, "shield_behavior")


def plot_zone_pair_heatmap(t7, outdir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(t7.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(t7.columns)))
    ax.set_xticklabels(t7.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(t7.index)))
    ax.set_yticklabels(t7.index)
    ax.set_xlabel("goal_zone")
    ax.set_ylabel("spawn_zone")
    for i in range(t7.shape[0]):
        for j in range(t7.shape[1]):
            v = t7.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("SR (%) par paire (spawn_zone, goal_zone) — S6")
    fig.colorbar(im, ax=ax, label="SR (%)")
    save_fig(fig, outdir, "zone_pair_heatmap_S6")


def run_eval_analysis(eval_path, outdir, make_plots=True):
    """Execute toute l'analyse d'eval et ecrit les CSV (+figures) dans outdir."""
    print(f"[EVAL] Chargement de {eval_path} ...")
    df = load_all_episodes(eval_path)
    print(f"  -> {len(df)} episodes charges")
    if df.empty:
        print("  !! Aucun episode trouve, verifie le chemin du fichier.")
        return
    print(f"  -> methods: {sorted(df.method.dropna().unique())}")
    print(f"  -> scenarios: {sorted(df.scenario.dropna().unique())}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n[EVAL] Table 1 - Difficulte par distance initiale au but")
    t1 = task_difficulty(df)
    print(t1)
    t1.to_csv(outdir / "task_difficulty.csv")
    if make_plots and not t1.empty:
        plot_task_difficulty(t1, outdir)

    print("\n[EVAL] Table 2 - SR intra-zone vs cross-zone")
    t2 = zone_transition(df)
    print(t2)
    t2.to_csv(outdir / "zone_transition.csv")
    if make_plots and not t2.empty:
        plot_zone_transition(t2, outdir)

    print("\n[EVAL] Table 3 - Clearance par outcome (S6)")
    t3 = clearance_by_outcome(df)
    print(t3)
    t3.to_csv(outdir / "clearance_by_outcome.csv")

    print("\n[EVAL] Bloc 4 - Throughput")
    t4 = throughput(df)
    for k, v in t4.items():
        print(f"  {k}: {v}")
    with open(outdir / "throughput.txt", "w") as f:
        for k, v in t4.items():
            f.write(f"{k}: {v}\n")

    print("\n[EVAL] Table 5 - Comportement du shield par scenario")
    t5 = shield_behavior(df)
    print(t5)
    t5.to_csv(outdir / "shield_behavior.csv")
    if make_plots and not t5.empty:
        plot_shield_behavior(t5, outdir)

    print("\n[EVAL] Bonus - Repartition des modes d'echec")
    t6 = failure_mode_by_scenario(df)
    print(t6)
    t6.to_csv(outdir / "failure_modes.csv")
    if make_plots and not t6.empty:
        plot_failure_modes(t6, outdir)

    print("\n[EVAL] Bonus - Heatmap SR par (spawn_zone, goal_zone), S6")
    t7 = zone_pair_heatmap(df, scenario="S6_density7_low")
    print(t7)
    t7.to_csv(outdir / "zone_pair_heatmap_S6.csv")
    if make_plots and not t7.empty:
        plot_zone_pair_heatmap(t7, outdir)

    print(f"\n[EVAL] Tous les fichiers ont ete ecrits dans {outdir}/")


# ---------------------------------------------------------------------------
# PARTIE TRAINING (episodes_combined_clean.csv — schema exact)
# ---------------------------------------------------------------------------

TRAIN_NUMERIC_COLS = [
    "step", "spawn_x", "spawn_y", "goal_x", "goal_y", "d_goal_init",
    "n_waypoints", "waypoints_reached", "reward", "steps", "success_roll",
    "collision_roll", "collision_ped_roll", "collision_static_roll",
    "shield_frozen_roll", "mean_reward_roll", "cbf_omega_decay_ep", "sps",
    "d_min_ped_terminal", "min_lidar_terminal",
]


def load_training_log(training_path):
    df = pd.read_csv(training_path)
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def training_learning_curves(df):
    """Extrait les courbes deja lissees par le logger (colonnes *_roll)."""
    cols = ["step", "episode", "success_roll", "collision_roll", "collision_ped_roll",
            "collision_static_roll", "shield_frozen_roll", "mean_reward_roll"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def outcome_distribution(df, by="__source_file"):
    """Repartition des outcomes par segment d'entrainement (fichier source)."""
    grp = df.groupby(by)["outcome"].value_counts(normalize=True).unstack(fill_value=0) * 100
    return grp.round(1)


def collision_type_breakdown(df, by="__source_file"):
    sub = df[df["collision_type"].notna() & (df["collision_type"] != "none")]
    if sub.empty:
        return pd.DataFrame()
    grp = sub.groupby(by)["collision_type"].value_counts(normalize=True).unstack(fill_value=0) * 100
    return grp.round(1)


def collision_source_breakdown(df):
    sub = df[df["collision_source"].notna() & (df["collision_source"] != "none")]
    if sub.empty:
        return pd.Series(dtype=float)
    return (sub["collision_source"].value_counts(normalize=True) * 100).round(1)


def task_difficulty_training(df):
    """SR (outcome == 'success') par tranche de distance initiale au but."""
    sub = df.copy()
    sub["bin"] = pd.cut(sub["d_goal_init"], bins=DIST_BINS, labels=DIST_LABELS)
    sub["is_success"] = (sub["outcome"] == "success").astype(int)
    grp = sub.groupby("bin", observed=True)["is_success"].agg(["count", "mean"])
    grp.columns = ["N", "SR"]
    grp["SR"] = (grp["SR"] * 100).round(1)
    return grp


def waypoint_completion(df, by="__source_file"):
    sub = df.copy()
    sub["waypoint_ratio"] = sub["waypoints_reached"] / sub["n_waypoints"].replace(0, np.nan)
    grp = sub.groupby(by)["waypoint_ratio"].agg(["mean", "median", "count"]).round(3)
    return grp


def spawn_zone_distribution(df, by="__source_file"):
    """Distribution de spawn_zone au fil de l'entrainement (verif rehearsal weighting)."""
    grp = df.groupby(by)["spawn_zone"].value_counts(normalize=True).unstack(fill_value=0) * 100
    return grp.round(1)


def shield_and_cbf_trend(df, by="__source_file"):
    agg = {}
    if "shield_frozen_roll" in df.columns:
        agg["shield_frozen_roll_mean"] = ("shield_frozen_roll", "mean")
    if "cbf_omega_decay_ep" in df.columns:
        agg["cbf_omega_decay_ep_mean"] = ("cbf_omega_decay_ep", "mean")
        agg["cbf_omega_decay_ep_last"] = ("cbf_omega_decay_ep", "last")
    if "collision_static_roll" in df.columns:
        agg["collision_static_roll_mean"] = ("collision_static_roll", "mean")
    if not agg:
        return pd.DataFrame()
    return df.groupby(by).agg(**agg).round(4)


def terminal_clearance_by_outcome(df):
    cols = [c for c in ["d_min_ped_terminal", "min_lidar_terminal"] if c in df.columns]
    if not cols or "outcome" not in df.columns:
        return pd.DataFrame()
    frames = []
    for c in cols:
        g = df.groupby("outcome")[c].agg(["count", "median", lambda x: x.quantile(0.1)])
        g.columns = [f"{c}_N", f"{c}_median", f"{c}_p10"]
        frames.append(g)
    return pd.concat(frames, axis=1).round(3)


def throughput_training(df):
    if "sps" not in df.columns:
        return {}
    return {
        "mean_sps": round(df["sps"].mean(), 1),
        "median_sps": round(df["sps"].median(), 1),
        "min_sps": round(df["sps"].min(), 1),
        "max_sps": round(df["sps"].max(), 1),
    }


def source_file_summary(df):
    """Un segment par __source_file : n episodes, plage de steps, reward moyen, SR."""
    sub = df.copy()
    sub["is_success"] = (sub["outcome"] == "success").astype(int)
    grp = sub.groupby("__source_file").agg(
        n_episodes=("episode", "count"),
        step_min=("step", "min"),
        step_max=("step", "max"),
        mean_reward=("reward", "mean"),
    )
    sr = sub.groupby("__source_file")["is_success"].mean() * 100
    grp["success_rate_pct"] = sr.round(1)
    return grp.sort_values("step_min")


# --- figures training ---

def plot_learning_curves(curves, outdir):
    if "step" not in curves.columns:
        x = curves.index
        xlabel = "episode (ordre)"
    else:
        x = curves["step"]
        xlabel = "step d'entrainement"

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    if "success_roll" in curves.columns:
        axes[0].plot(x, curves["success_roll"] * (100 if curves["success_roll"].max() <= 1 else 1),
                     color="#2f855a")
        axes[0].set_ylabel("Taux de succes lisse (%)")
    if "mean_reward_roll" in curves.columns:
        axes[1].plot(x, curves["mean_reward_roll"], color="#2b6cb0")
        axes[1].set_ylabel("Reward moyen lisse")
    if "collision_ped_roll" in curves.columns:
        axes[2].plot(x, curves["collision_ped_roll"], label="pieton", color="#c53030")
    if "collision_static_roll" in curves.columns:
        axes[2].plot(x, curves["collision_static_roll"], label="statique", color="#c05621")
    if "collision_roll" in curves.columns:
        axes[2].plot(x, curves["collision_roll"], label="total", color="#4a5568", linestyle="--")
    axes[2].set_ylabel("Taux de collision lisse")
    axes[2].set_xlabel(xlabel)
    axes[2].legend(fontsize=8)

    axes[0].set_title("Courbes d'apprentissage (moyennes glissantes)")
    save_fig(fig, outdir, "learning_curves")


def plot_task_difficulty_training(t1, outdir):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(t1.index.astype(str), t1["SR"], color="#2b6cb0")
    for i, (n, sr) in enumerate(zip(t1["N"], t1["SR"])):
        ax.text(i, sr + 1.5, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xlabel("Distance initiale au but (m)")
    ax.set_ylabel("Taux de succes (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SR (entrainement) par tranche de distance initiale")
    save_fig(fig, outdir, "task_difficulty_training")


def plot_spawn_zone_distribution(t_spawn, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(t_spawn))
    for col in t_spawn.columns:
        vals = t_spawn[col].values
        ax.bar(t_spawn.index.astype(str), vals, bottom=bottom, label=col)
        bottom += vals
    ax.set_ylabel("Part des episodes (%)")
    ax.set_title("Distribution de spawn_zone par segment d'entrainement")
    ax.legend(fontsize=7, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    save_fig(fig, outdir, "spawn_zone_distribution")


def plot_shield_cbf_trend(t_shield, outdir):
    cols = [c for c in t_shield.columns if c in
            ("shield_frozen_roll_mean", "cbf_omega_decay_ep_mean", "collision_static_roll_mean")]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(t_shield.index))
    width = 0.8 / max(len(cols), 1)
    for i, c in enumerate(cols):
        ax.bar(x + i * width, t_shield[c], width, label=c)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(t_shield.index, rotation=20, ha="right", fontsize=8)
    ax.set_title("Tendance shield / CBF par segment d'entrainement")
    ax.legend(fontsize=7)
    save_fig(fig, outdir, "shield_cbf_trend")


def plot_terminal_clearance(t_clear, outdir):
    if t_clear.empty:
        return
    median_cols = [c for c in t_clear.columns if c.endswith("_median")]
    if not median_cols:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(t_clear.index))
    width = 0.8 / len(median_cols)
    for i, c in enumerate(median_cols):
        ax.bar(x + i * width, t_clear[c], width, label=c.replace("_median", ""))
    ax.set_xticks(x + width * (len(median_cols) - 1) / 2)
    ax.set_xticklabels(t_clear.index, rotation=20, ha="right")
    ax.set_ylabel("Distance mediane (m)")
    ax.set_title("Clearance terminale par outcome")
    ax.legend(fontsize=8)
    save_fig(fig, outdir, "terminal_clearance_by_outcome")


def run_training_analysis(training_path, outdir, make_plots=True):
    """Analyse complete du log d'entrainement (episodes_combined_clean.csv)."""
    print(f"\n[TRAINING] Chargement de {training_path} ...")
    df = load_training_log(training_path)
    print(f"  -> {len(df)} episodes, {len(df.columns)} colonnes")
    missing = [c for c in TRAIN_NUMERIC_COLS if c not in df.columns]
    if missing:
        print(f"  !! Colonnes numeriques attendues absentes: {missing}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = [f"Episodes: {len(df)}", f"Colonnes: {list(df.columns)}"]
    if "step" in df.columns:
        lines.append(f"Plage de steps: {int(df['step'].min())} -> {int(df['step'].max())}")

    print("\n[TRAINING] Courbes d'apprentissage (colonnes *_roll deja lissees)")
    curves = training_learning_curves(df)
    curves.to_csv(outdir / "learning_curves.csv", index=False)
    if make_plots and not curves.empty:
        plot_learning_curves(curves, outdir)
    if "success_roll" in df.columns:
        lines.append(f"Taux de succes lisse, dernier point: {df['success_roll'].iloc[-1]:.3f}")
    if "mean_reward_roll" in df.columns:
        lines.append(f"Reward moyen lisse, dernier point: {df['mean_reward_roll'].iloc[-1]:.2f}")

    print("\n[TRAINING] Repartition des outcomes par segment (__source_file)")
    if {"outcome", "__source_file"}.issubset(df.columns):
        t_outcome = outcome_distribution(df)
        print(t_outcome)
        t_outcome.to_csv(outdir / "outcome_distribution.csv")

    print("\n[TRAINING] Repartition des types de collision par segment")
    if "collision_type" in df.columns:
        t_coll = collision_type_breakdown(df)
        if not t_coll.empty:
            print(t_coll)
            t_coll.to_csv(outdir / "collision_type_breakdown.csv")

    print("\n[TRAINING] Source des detections de collision")
    if "collision_source" in df.columns:
        t_collsrc = collision_source_breakdown(df)
        if not t_collsrc.empty:
            print(t_collsrc)
            t_collsrc.to_csv(outdir / "collision_source_breakdown.csv")

    print("\n[TRAINING] Difficulte (SR) par tranche de distance initiale au but")
    if {"d_goal_init", "outcome"}.issubset(df.columns):
        t_diff = task_difficulty_training(df)
        print(t_diff)
        t_diff.to_csv(outdir / "task_difficulty_training.csv")
        if make_plots and not t_diff.empty:
            plot_task_difficulty_training(t_diff, outdir)

    print("\n[TRAINING] Taux de waypoints atteints par segment")
    if {"waypoints_reached", "n_waypoints"}.issubset(df.columns):
        t_wp = waypoint_completion(df)
        print(t_wp)
        t_wp.to_csv(outdir / "waypoint_completion.csv")

    print("\n[TRAINING] Distribution spawn_zone par segment (verif rehearsal weighting)")
    if "spawn_zone" in df.columns:
        t_spawn = spawn_zone_distribution(df)
        print(t_spawn)
        t_spawn.to_csv(outdir / "spawn_zone_distribution.csv")
        if make_plots and not t_spawn.empty:
            plot_spawn_zone_distribution(t_spawn, outdir)

    print("\n[TRAINING] Tendance shield / CBF par segment")
    t_shield = shield_and_cbf_trend(df)
    if not t_shield.empty:
        print(t_shield)
        t_shield.to_csv(outdir / "shield_cbf_trend.csv")
        if make_plots:
            plot_shield_cbf_trend(t_shield, outdir)

    print("\n[TRAINING] Clearance terminale par outcome")
    t_clear = terminal_clearance_by_outcome(df)
    if not t_clear.empty:
        print(t_clear)
        t_clear.to_csv(outdir / "terminal_clearance_by_outcome.csv")
        if make_plots:
            plot_terminal_clearance(t_clear, outdir)

    print("\n[TRAINING] Throughput (sps)")
    t_thr = throughput_training(df)
    if t_thr:
        for k, v in t_thr.items():
            print(f"  {k}: {v}")
        with open(outdir / "throughput_training.txt", "w") as f:
            for k, v in t_thr.items():
                f.write(f"{k}: {v}\n")

    print("\n[TRAINING] Resume par fichier source (segments d'entrainement)")
    if "__source_file" in df.columns:
        t_src = source_file_summary(df)
        print(t_src)
        t_src.to_csv(outdir / "source_file_summary.csv")

    with open(outdir / "training_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[TRAINING] Tous les fichiers ont ete ecrits dans {outdir}/")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse eval + training pour CM-GAP-SAC")
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH,
                         help="Chemin vers le fichier/dossier .jsonl d'eval")
    parser.add_argument("--training", default=DEFAULT_TRAINING_PATH,
                         help="Chemin vers le CSV de training (episodes_combined_clean.csv)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                         help="Dossier racine de sortie (contiendra eval/ et training/)")
    parser.add_argument("--no-plots", action="store_true",
                         help="Ne genere que les CSV/TXT, sans les figures PDF")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    make_plots = not args.no_plots

    run_eval_analysis(args.eval, root / "eval", make_plots=make_plots)
    run_training_analysis(args.training, root / "training", make_plots=make_plots)

    print(f"\nTermine. Resultats dans {root}/eval et {root}/training")


if __name__ == "__main__":
    main()