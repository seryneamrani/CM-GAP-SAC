#!/usr/bin/env python3
"""Generate results figures from the CM-GAP-SAC evaluation manifest.

Produces three PDF figures used in the results section:
  - task_difficulty_training.pdf
  - zone_transition.pdf
  - terminal_clearance_by_outcome.pdf
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------
MANIFEST_PATH = Path(
    "/home/seryne/limo_jazzy_ws/src/cm-gap-sac-eval/outputs/"
    "results_backup_20260727_1157.jsonl"
)
OUTPUT_DIR = Path(
    "/home/seryne/limo_jazzy_ws/src/cm-gap-sac-eval/outputs/figures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Academic style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Academic blue / orange / green palette (muted, print-friendly)
COL_BLUE   = "#2E5C8A"
COL_ORANGE = "#D97706"
COL_GREEN  = "#3F7D3F"
COL_INK    = "#1f1f1f"   # for annotation text — dark, readable


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Python's json.loads accepts Infinity / -Infinity / NaN by default,
            # which matches how the manifest writes min_clearance_pedestrian_m
            # when no pedestrians are present.
            episodes.append(json.loads(line))
    return episodes


def outcome_label(ep: dict) -> str:
    o = ep["outcome"]
    if o.get("success"):
        return "Success"
    if o.get("collision_pedestrian") or o.get("collision_static"):
        return "Collision"
    if o.get("timeout"):
        return "Timeout"
    return "Other"


def sanitize(v, cap: float | None = None):
    """Return a finite float, capping Infinity at `cap`, else None."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return cap
    return float(v)


episodes = load_episodes(MANIFEST_PATH)
print(f"Loaded {len(episodes)} episodes from {MANIFEST_PATH.name}")


# ===========================================================================
# Figure 1: Success rate by initial goal distance (training)
# ===========================================================================
# NOTE: adapt the filter below if training and evaluation are mixed in the
# same manifest. Common markers: `checkpoint`, `scenario_group`, `phase`.
training_eps = episodes

distances = np.array(
    [
        math.dist(e["spawn_xy"], e["goal_xy"])
        for e in training_eps
    ],
    dtype=float,
)
successes = np.array(
    [bool(e["outcome"]["success"]) for e in training_eps], dtype=bool
)

mask = np.isfinite(distances) & (distances > 0)
distances = distances[mask]
successes = successes[mask]

step = 1.0
d_min = max(0.0, math.floor(distances.min()))
d_max = math.ceil(distances.max())
bin_edges = np.arange(d_min, d_max + step, step)
bin_idx = np.digitize(distances, bin_edges) - 1

centers, rates, counts = [], [], []
for b in range(len(bin_edges) - 1):
    sel = bin_idx == b
    n = int(sel.sum())
    if n == 0:
        continue
    centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
    rates.append(100.0 * successes[sel].mean())
    counts.append(n)

centers = np.array(centers)
rates = np.array(rates)
counts = np.array(counts)

fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
ax1.bar(
    centers, rates, width=0.85 * step,
    color=COL_BLUE, edgecolor="white", linewidth=0.8, alpha=0.92,
    label="Success rate", zorder=3,
)
ax1.set_xlabel("Initial goal distance (m)")
ax1.set_ylabel("Success rate (%)")
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax1.yaxis.grid(True)
ax1.xaxis.grid(False)

fig.tight_layout()
out_path = OUTPUT_DIR / "task_difficulty_training.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"  wrote {out_path}")


# ===========================================================================
# Figure 2: Intra-zone vs cross-zone success rate per scenario
# ===========================================================================
def scen_sort_key(s: str) -> int:
    try:
        return int(s.split("_")[0][1:])
    except Exception:
        return 999

scenarios = sorted({e["scenario_group"] for e in episodes}, key=scen_sort_key)

intra_rates, cross_rates, intra_ns, cross_ns = [], [], [], []
for s in scenarios:
    intra = [e for e in episodes if e["scenario_group"] == s
             and e.get("zone_type") == "intra"]
    cross = [e for e in episodes if e["scenario_group"] == s
             and e.get("zone_type") == "cross"]
    intra_rates.append(100.0 * np.mean([e["outcome"]["success"] for e in intra]) if intra else 0.0)
    cross_rates.append(100.0 * np.mean([e["outcome"]["success"] for e in cross]) if cross else 0.0)
    intra_ns.append(len(intra))
    cross_ns.append(len(cross))

x = np.arange(len(scenarios))
w = 0.38

fig, ax = plt.subplots(figsize=(7.5, 4.3))
b1 = ax.bar(x - w / 2, intra_rates, w, color=COL_BLUE,
            edgecolor="white", linewidth=0.8, label="Intra-zone", zorder=3)
b2 = ax.bar(x + w / 2, cross_rates, w, color=COL_ORANGE,
            edgecolor="white", linewidth=0.8, label="Cross-zone", zorder=3)

# Short scenario labels on x-axis
short_labels = [s.split("_")[0] for s in scenarios]
ax.set_xticks(x)
ax.set_xticklabels(short_labels)
ax.set_xlabel("Scenario")
ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 115)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Percentage ABOVE each bar; sample count INSIDE the bar (bottom) in white
for bar, val, n in zip(b1, intra_rates, intra_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=9, color=COL_INK, zorder=4)
    ax.text(bar.get_x() + bar.get_width() / 2, 3,
            f"n={n}", ha="center", va="bottom",
            fontsize=8, color="white", zorder=5)
for bar, val, n in zip(b2, cross_rates, cross_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=9, color=COL_INK, zorder=4)
    ax.text(bar.get_x() + bar.get_width() / 2, 3,
            f"n={n}", ha="center", va="bottom",
            fontsize=8, color="white", zorder=5)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
          ncol=2, frameon=False)
fig.tight_layout()
out_path = OUTPUT_DIR / "zone_transition.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"  wrote {out_path}")


# ===========================================================================
# Figure 3: Terminal clearances by outcome
# ===========================================================================
CAP = 5.0  # cap Infinity (no pedestrian in scene) at 5 m for plotting

groups = ["Success", "Timeout", "Collision"]
ped_data, sta_data, group_counts = [], [], []
for g in groups:
    eps = [e for e in episodes if outcome_label(e) == g]
    ped = [sanitize(e["metrics"].get("min_clearance_pedestrian_m"), CAP) for e in eps]
    sta = [sanitize(e["metrics"].get("min_clearance_static_m"), CAP) for e in eps]
    ped = [v for v in ped if v is not None]
    sta = [v for v in sta if v is not None]
    ped_data.append(ped)
    sta_data.append(sta)
    group_counts.append(len(eps))

fig, ax = plt.subplots(figsize=(7.5, 4.3))
positions_ped = np.arange(len(groups)) * 2.5
positions_sta = positions_ped + 0.9

def style_box(bx, color):
    for patch in bx["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for whisker in bx["whiskers"]:
        whisker.set_color(color)
        whisker.set_linewidth(1.0)
    for cap in bx["caps"]:
        cap.set_color(color)
        cap.set_linewidth(1.0)
    for median in bx["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

box_ped = ax.boxplot(
    ped_data, positions=positions_ped, widths=0.75,
    patch_artist=True, showfliers=False,
)
box_sta = ax.boxplot(
    sta_data, positions=positions_sta, widths=0.75,
    patch_artist=True, showfliers=False,
)
style_box(box_ped, COL_BLUE)
style_box(box_sta, COL_ORANGE)

# Sample counts baked into x-tick labels — no more out-of-axis text
tick_labels = [f"{g}\n(n={n})" for g, n in zip(groups, group_counts)]
ax.set_xticks(positions_ped + 0.45)
ax.set_xticklabels(tick_labels)
ax.set_ylabel("Terminal minimum clearance (m)")
ax.set_xlabel("Episode outcome")
ax.set_ylim(bottom=0)
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Annotate medians above each box for readability
def annotate_medians(data_list, positions, color):
    for d, xpos in zip(data_list, positions):
        if not d:
            continue
        med = float(np.median(d))
        ax.text(xpos, med + 0.12, f"{med:.2f}",
                ha="center", va="bottom", fontsize=8.5,
                color=color, fontweight="bold", zorder=5)

annotate_medians(ped_data, positions_ped, COL_BLUE)
annotate_medians(sta_data, positions_sta, COL_ORANGE)

legend_elems = [
    Patch(facecolor=COL_BLUE, alpha=0.78, label="Pedestrian clearance"),
    Patch(facecolor=COL_ORANGE, alpha=0.78, label="Static clearance"),
]
ax.legend(handles=legend_elems, loc="upper right", frameon=False)

fig.tight_layout()
out_path = OUTPUT_DIR / "terminal_clearance_by_outcome.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"  wrote {out_path}")

print(f"\nDone. Figures in: {OUTPUT_DIR}")