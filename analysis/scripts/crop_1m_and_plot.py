"""
Crop episodes.csv at 1M environment steps and recompute rolling metrics.

Usage:
    python crop_1m_and_plot.py --csv path/to/episodes.csv --outdir ./out_1M

Adjust OUTCOME_* and COLLISION_TYPE_* below if your logger uses different strings.
The script first prints the unique values it finds so you can verify.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Config — verify against the printout at the start of the run
# ------------------------------------------------------------
STEP_CUTOFF = 1_000_000
ROLL_WINDOW = 100                    # matches the "_100" TB tags

OUTCOME_SUCCESS = "success"
OUTCOME_COLLISION_ANY_PREFIX = "collision"   # any outcome starting with this
COLLISION_TYPE_PED = "pedestrian"
COLLISION_TYPE_STATIC = "static"


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True, help="Path to episodes.csv")
parser.add_argument("--outdir", type=str, default="./out_1M",
                    help="Where to write the cropped CSV and figure")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load and inspect
# ------------------------------------------------------------
df = pd.read_csv(args.csv)
print(f"Loaded {len(df):,} episodes from {args.csv}")
print(f"Step range: {df['step'].min():,} → {df['step'].max():,}")
print(f"Unique outcomes: {sorted(df['outcome'].dropna().unique().tolist())}")
print(f"Unique collision_type: {sorted(df['collision_type'].dropna().unique().tolist())}")
print()


# ------------------------------------------------------------
# Crop at 1M steps
# ------------------------------------------------------------
df_c = df[df["step"] <= STEP_CUTOFF].reset_index(drop=True).copy()
print(f"After cropping at step {STEP_CUTOFF:,}: {len(df_c):,} episodes kept, "
      f"{len(df) - len(df_c):,} dropped")


# ------------------------------------------------------------
# Per-episode binary indicators
# ------------------------------------------------------------
df_c["_is_success"] = (df_c["outcome"] == OUTCOME_SUCCESS).astype(int)
df_c["_is_coll"] = (
    df_c["outcome"].fillna("").str.startswith(OUTCOME_COLLISION_ANY_PREFIX).astype(int)
)
df_c["_is_coll_ped"] = (df_c["collision_type"].fillna("") == COLLISION_TYPE_PED).astype(int)
df_c["_is_coll_static"] = (df_c["collision_type"].fillna("") == COLLISION_TYPE_STATIC).astype(int)


# ------------------------------------------------------------
# Recompute the six rolling columns
# ------------------------------------------------------------
def roll(series):
    return series.rolling(ROLL_WINDOW, min_periods=1).mean()

df_c["success_roll"] = roll(df_c["_is_success"])
df_c["collision_roll"] = roll(df_c["_is_coll"])
df_c["collision_ped_roll"] = roll(df_c["_is_coll_ped"])
df_c["collision_static_roll"] = roll(df_c["_is_coll_static"])
df_c["mean_reward_roll"] = roll(df_c["reward"])
# shield_frozen_roll: original column kept as-is (no per-episode source in the schema)


# ------------------------------------------------------------
# Cleanup helper columns
# ------------------------------------------------------------
df_c = df_c.drop(columns=[c for c in df_c.columns if c.startswith("_")])


# ------------------------------------------------------------
# Save cropped CSV
# ------------------------------------------------------------
csv_out = outdir / "episodes_1M.csv"
df_c.to_csv(csv_out, index=False)
print(f"\nSaved cropped CSV: {csv_out}")


# ------------------------------------------------------------
# Numbers to quote in Chapter 5
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL NUMBERS at step 1M (rolling mean over last 100 episodes)")
print("=" * 60)
print(f"  Success rate:               {df_c['success_roll'].iloc[-1]:.4f}")
print(f"  Collision rate (total):     {df_c['collision_roll'].iloc[-1]:.4f}")
print(f"  Collision rate (static):    {df_c['collision_static_roll'].iloc[-1]:.4f}")
print(f"  Collision rate (pedestrian):{df_c['collision_ped_roll'].iloc[-1]:.4f}")
print(f"  Mean episode reward:        {df_c['mean_reward_roll'].iloc[-1]:.4f}")

# Also print the mean over the final 100 episodes (a smoother "final" number)
tail = df_c.tail(ROLL_WINDOW)
print(f"\nMean over the last {ROLL_WINDOW} episodes (for the abstract / conclusion):")
print(f"  Success rate:               {tail['success_roll'].mean():.4f}")
print(f"  Collision rate (total):     {tail['collision_roll'].mean():.4f}")
print(f"  Collision rate (static):    {tail['collision_static_roll'].mean():.4f}")
print(f"  Collision rate (pedestrian):{tail['collision_ped_roll'].mean():.4f}")
print(f"  Mean episode reward:        {tail['mean_reward_roll'].mean():.4f}")


# ------------------------------------------------------------
# Sanity check: recomputed vs original rolling values
# ------------------------------------------------------------
orig = df[df["step"] <= STEP_CUTOFF].reset_index(drop=True)
diff = (df_c["success_roll"] - orig["success_roll"]).abs().max()
print(f"\nMax abs diff between recomputed and original success_roll: {diff:.6f}")
print("(A value near zero means your original logger was causal — safe to use either.)")


# ------------------------------------------------------------
# Plots for Chapter 5
# ------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
x = df_c["step"]

# Rolling success
axes[0, 0].plot(x, df_c["success_roll"], color="C0", linewidth=1.1)
axes[0, 0].set_ylabel("Success rate")
axes[0, 0].set_title("Rolling success rate (W = 100)")
axes[0, 0].set_ylim(0, 1)

# Rolling collision, split
axes[0, 1].plot(x, df_c["collision_static_roll"], color="C3", linewidth=1.1, label="static")
axes[0, 1].plot(x, df_c["collision_ped_roll"], color="C1", linewidth=1.1, label="pedestrian")
axes[0, 1].set_ylabel("Collision rate")
axes[0, 1].set_title("Rolling collision rate, split (W = 100)")
axes[0, 1].legend(loc="upper right")

# Rolling mean reward
axes[0, 2].plot(x, df_c["mean_reward_roll"], color="C2", linewidth=1.1)
axes[0, 2].set_ylabel("Mean episode reward")
axes[0, 2].set_title("Rolling mean reward (W = 100)")

# PER beta schedule (deterministic — not from CSV)
beta_x = pd.Series(range(0, STEP_CUTOFF + 1, 1000))
beta_y = (0.4 + (1.0 - 0.4) * (beta_x / STEP_CUTOFF)).clip(upper=1.0)
axes[1, 0].plot(beta_x, beta_y, color="C4", linewidth=1.5)
axes[1, 0].set_ylabel(r"$\beta$")
axes[1, 0].set_xlabel("Environment step")
axes[1, 0].set_title(r"PER importance-sampling exponent $\beta$")
axes[1, 0].set_ylim(0.35, 1.05)

# Success vs total collision, more smoothed
axes[1, 1].plot(x, df_c["success_roll"], color="C0", alpha=0.25, linewidth=0.8)
axes[1, 1].plot(x, df_c["collision_roll"], color="C3", alpha=0.25, linewidth=0.8)
axes[1, 1].plot(
    x, df_c["success_roll"].rolling(500, min_periods=1).mean(),
    color="C0", linewidth=1.6, label="success (smooth)",
)
axes[1, 1].plot(
    x, df_c["collision_roll"].rolling(500, min_periods=1).mean(),
    color="C3", linewidth=1.6, label="collision (smooth)",
)
axes[1, 1].set_ylabel("Rate")
axes[1, 1].set_xlabel("Environment step")
axes[1, 1].set_title("Success vs collision, extra smoothing")
axes[1, 1].legend(loc="center right")
axes[1, 1].set_ylim(0, 1)

# Cumulative successes
df_c["_cum_success"] = (df_c["outcome"] == OUTCOME_SUCCESS).cumsum()
axes[1, 2].plot(x, df_c["_cum_success"], color="C0", linewidth=1.3)
axes[1, 2].set_ylabel("Cumulative successes")
axes[1, 2].set_xlabel("Environment step")
axes[1, 2].set_title("Cumulative success count")

for ax in axes.flatten():
    ax.set_xlim(0, STEP_CUTOFF)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig_out = outdir / "training_curves_1M.png"
plt.savefig(fig_out, dpi=200, bbox_inches="tight")
print(f"\nSaved figure: {fig_out}")
