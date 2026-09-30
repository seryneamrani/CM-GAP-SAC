"""
Generate 1 snapshot PNG per scenario, showing:
    - Hospital walls (rectangles)
    - Static obstacles (circles)
    - Pedestrian starting positions for that scenario's density (dots with speed arrows)
    - Grid overlay (1m × 1m cells) with occupancy shading
    - Petite room centers (annotated)

Usage:
    python generate_snapshots.py --config eval_config.yaml --output-dir outputs/snapshots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

from geometry import (
    WALLS, STATIC_OBSTACLES, PED_STARTS, ZONE_BOXES,
    HOSPITAL_X_MIN, HOSPITAL_X_MAX, HOSPITAL_Y_MIN, HOSPITAL_Y_MAX,
    is_navigable,
)


GRID_CELL_SIZE = 1.0  # meters — 1m × 1m occupancy grid


def is_cell_occupied(cx: float, cy: float, cell_size: float) -> bool:
    """Sample cell center + corners; if majority blocked, mark occupied."""
    half = cell_size / 2
    samples = [
        (cx, cy),
        (cx - half + 0.05, cy - half + 0.05),
        (cx + half - 0.05, cy - half + 0.05),
        (cx - half + 0.05, cy + half - 0.05),
        (cx + half - 0.05, cy + half - 0.05),
    ]
    blocked = sum(1 for (x, y) in samples if not is_navigable(x, y))
    return blocked >= 3


def draw_scenario(scenario: dict, ax: plt.Axes) -> None:
    ax.set_xlim(HOSPITAL_X_MIN - 0.5, HOSPITAL_X_MAX + 0.5)
    ax.set_ylim(HOSPITAL_Y_MIN - 0.5, HOSPITAL_Y_MAX + 0.5)
    ax.set_aspect("equal")

    # ---- Occupancy grid (background shading) ----
    x_cells = np.arange(HOSPITAL_X_MIN, HOSPITAL_X_MAX + GRID_CELL_SIZE, GRID_CELL_SIZE)
    y_cells = np.arange(HOSPITAL_Y_MIN, HOSPITAL_Y_MAX + GRID_CELL_SIZE, GRID_CELL_SIZE)
    for i in range(len(x_cells) - 1):
        for j in range(len(y_cells) - 1):
            cx = (x_cells[i] + x_cells[i + 1]) / 2
            cy = (y_cells[j] + y_cells[j + 1]) / 2
            if is_cell_occupied(cx, cy, GRID_CELL_SIZE):
                rect = mpatches.Rectangle(
                    (x_cells[i], y_cells[j]), GRID_CELL_SIZE, GRID_CELL_SIZE,
                    facecolor="lightgray", edgecolor="none", alpha=0.6, zorder=0,
                )
                ax.add_patch(rect)

    # ---- Grid lines ----
    for x in x_cells:
        ax.axvline(x, color="gray", alpha=0.2, linewidth=0.4, zorder=1)
    for y in y_cells:
        ax.axhline(y, color="gray", alpha=0.2, linewidth=0.4, zorder=1)

    # ---- Zone boundaries ----
    zone_colors = {
        "corridor": "#4A90E2", "longue_1": "#7ED321", "longue_2": "#7ED321",
        "petite_1": "#F5A623", "petite_2": "#F5A623",
    }
    for zone_name, (xmin, xmax, ymin, ymax) in ZONE_BOXES.items():
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=zone_colors.get(zone_name, "lightblue"),
            edgecolor=zone_colors.get(zone_name, "blue"),
            alpha=0.12, linewidth=1.2, linestyle="--", zorder=2,
        )
        ax.add_patch(rect)
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, zone_name,
                fontsize=7, ha="center", va="center", color="darkslategray",
                alpha=0.8, zorder=3)

    # ---- Walls (black rectangles) ----
    for xmin, xmax, ymin, ymax in WALLS:
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor="black", edgecolor="black", zorder=4,
        )
        ax.add_patch(rect)

    # ---- Static obstacles (red circles) ----
    for ox, oy, oradius in STATIC_OBSTACLES:
        circle = mpatches.Circle(
            (ox, oy), oradius, facecolor="firebrick", edgecolor="darkred",
            alpha=0.7, zorder=5,
        )
        ax.add_patch(circle)
    if STATIC_OBSTACLES:
        ax.plot([], [], "o", color="firebrick", markersize=8, label="Static obstacle")

    # ---- Pedestrians for this scenario (first N from PED_STARTS) ----
    # ---- Pedestrians for this scenario (first N from PED_STARTS) ----
    n_ped = scenario["n_pedestrians"]
    speed = scenario["pedestrian_speed_mps"]
    if n_ped > 0:
        selected = PED_STARTS[:n_ped]
        for i, (sx, sy) in enumerate(selected):
            circle = mpatches.Circle(
                (sx, sy), 0.25, facecolor="royalblue", edgecolor="navy",
                alpha=0.85, zorder=6,
            )
            ax.add_patch(circle)
            ax.text(sx, sy, str(i + 1), fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold", zorder=7)
            # Velocity arrow (arbitrary direction for viz — actual direction is
            # SFM-generated at runtime)
            angle = 2 * np.pi * i / n_ped
            dx, dy = speed * np.cos(angle), speed * np.sin(angle)
            ax.arrow(sx, sy, dx * 1.5, dy * 1.5, head_width=0.15,
                     head_length=0.15, fc="navy", ec="navy", alpha=0.6, zorder=6)

            # --- AJOUT : vitesse écrite au bout de la flèche ---
            tip_x, tip_y = sx + dx * 1.5, sy + dy * 1.5
            ax.text(
                tip_x, tip_y, f"{speed:.1f} m/s",
                fontsize=6, ha="center", va="bottom",
                color="navy", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

        ax.plot([], [], "o", color="royalblue", markersize=8,
                label=f"Pedestrian ({n_ped} @ {speed} m/s)")
    else:
        ax.plot([], [], " ", label="No pedestrians (static-only)")



    # ---- Title + labels ----
    ax.set_title(
        f"{scenario['id']}\n{scenario.get('description', '')}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/snapshots"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print(f"Generating {len(cfg['scenarios'])} snapshots to {args.output_dir}/")
    for scenario in cfg["scenarios"]:
        fig, ax = plt.subplots(figsize=(9, 9))
        draw_scenario(scenario, ax)
        out = args.output_dir / f"{scenario['id']}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    # Also generate a combined 4x2 grid overview
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    scenarios = cfg["scenarios"]
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        draw_scenario(scenario, ax)
    # Hide unused subplots
    for idx in range(len(scenarios), 8):
        axes[idx // 2, idx % 2].axis("off")
    fig.suptitle("CM-GAP-SAC Evaluation — 7 Scenarios", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = args.output_dir / "all_scenarios_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    print(f"\nDone. {len(cfg['scenarios']) + 1} images generated.")


if __name__ == "__main__":
    main()
