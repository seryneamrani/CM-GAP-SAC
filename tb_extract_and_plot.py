"""
Generate the two training-result figures for Chapter 5, Section 5.3.

Figure 1 (fig:training_convergence): 3 panels
  (a) rolling mean episode reward (from CSV)
  (b) rolling success rate         (from CSV)
  (c) SAC entropy temperature alpha (from TB event files)

Figure 2 (fig:training_stability): 2 panels
  (a) PER importance-sampling exponent beta (from TB event files)
  (b) critic q_min_mean                     (from TB event files)

Usage:
    python make_ch5_figures.py \
        --logdir runs/cm_gap_sac/20260629-nav2-wide-doors/ \
        --outdir ./ch5_figures

The CSV loading uses the stitched last-session merge (same logic as before).
"""

import argparse
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit("Install tensorboard: pip install tensorboard")


BASE = Path("runs/cm_gap_sac/20260629-nav2-wide-doors")
CSV_FILES = [
    BASE / "episodes_v1_pre_logging_fix.csv",
    BASE / "episodes_run_v1_abandoned.csv",
    BASE / "episodes.csv",
]

STEP_CUTOFF = 1_150_000
ROLL_WINDOW = 100
SMOOTH_W = 300

TAG_ALPHA = "train/alpha"
TAG_BETA  = "per/beta"
TAG_QMIN  = "train/q_min_mean"

OUTCOME_SUCCESS = "success"

EXPECTED_COLS = [
    "episode", "step", "outcome", "collision_type", "spawn_zone",
    "spawn_x", "spawn_y", "goal_x", "goal_y", "d_goal_init",
    "n_waypoints", "waypoints_reached", "reward", "steps",
    "success_roll", "collision_roll", "collision_ped_roll",
    "collision_static_roll", "shield_frozen_roll", "mean_reward_roll",
    "cbf_omega_decay_ep", "sps", "d_min_ped_terminal",
    "min_lidar_terminal", "collision_source",
]


parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default=str(BASE))
parser.add_argument("--outdir", type=str, default="./ch5_figures")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


# ============================================================
# CSV loading and last-session stitching
# ============================================================
def load_fragment(path: Path) -> pd.DataFrame:
    raw = path.read_text(errors="replace")
    lines = raw.splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0]
    cols_in_header = [c.strip() for c in header.split(",")]
    if cols_in_header[0] == "sps":
        data_fields = len(lines[1].split(",")) if len(lines) > 1 else 0
        n_missing = data_fields - len(cols_in_header)
        raw = (",".join(EXPECTED_COLS[:n_missing]) + "," + header + "\n"
               + "\n".join(lines[1:]))
    df = pd.read_csv(StringIO(raw), low_memory=False)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episode"] = pd.to_numeric(df.get("episode", pd.Series()), errors="coerce")
    df = df.dropna(subset=["step", "episode"]).reset_index(drop=True)
    df["step"] = df["step"].astype(int)
    df["episode"] = df["episode"].astype(int)
    return df


def keep_last_session(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    ep = df["episode"].values
    last_restart = 0
    for i in range(1, len(ep)):
        if ep[i] < ep[i-1] and ep[i] <= 5:
            last_restart = i
    return df.iloc[last_restart:].reset_index(drop=True)


print("Loading CSV fragments...")
kept = []
for p in CSV_FILES:
    if not p.exists():
        continue
    df = keep_last_session(load_fragment(p))
    kept.append(df)
    print(f"  {p.name}: kept {len(df):,} rows "
          f"(steps {df['step'].min():,} → {df['step'].max():,})")

df_csv = pd.concat(kept, ignore_index=True).sort_values("step")
df_csv = df_csv.drop_duplicates(subset="step", keep="first").reset_index(drop=True)
df_csv = df_csv[df_csv["step"] <= STEP_CUTOFF].reset_index(drop=True)
print(f"Merged CSV: {len(df_csv):,} episodes, "
      f"steps {df_csv['step'].min():,} → {df_csv['step'].max():,}")

# Recompute rolling metrics
df_csv["_suc"] = (df_csv["outcome"].fillna("") == OUTCOME_SUCCESS).astype(float)
df_csv["success_roll"]     = df_csv["_suc"].rolling(ROLL_WINDOW, min_periods=1).mean()
df_csv["mean_reward_roll"] = df_csv["reward"].rolling(ROLL_WINDOW, min_periods=1).mean()


# ============================================================
# TB event file extraction with stitching
# ============================================================
def load_tb_scalars(logdir: Path, tags: list) -> dict:
    event_files = sorted(logdir.rglob("events.out.tfevents.*"))
    print(f"\nFound {len(event_files)} TB event files")
    per_tag = defaultdict(list)
    for f in event_files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
            continue
        available = ea.Tags().get("scalars", [])
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                if events:
                    per_tag[tag].append(pd.DataFrame({
                        "step": [e.step for e in events],
                        "value": [e.value for e in events],
                        "wall_time": [e.wall_time for e in events],
                    }))
    out = {}
    for tag, frames in per_tag.items():
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values(["step", "wall_time"], kind="mergesort")
        merged = merged.drop_duplicates(subset="step", keep="last").reset_index(drop=True)
        merged = merged[merged["step"] <= STEP_CUTOFF].reset_index(drop=True)
        out[tag] = merged
        print(f"  {tag}: {len(merged):,} samples, "
              f"steps {merged['step'].min():,} → {merged['step'].max():,}")
    return out


tb = load_tb_scalars(Path(args.logdir), [TAG_ALPHA, TAG_BETA, TAG_QMIN])

if not all(t in tb for t in [TAG_ALPHA, TAG_BETA, TAG_QMIN]):
    missing = [t for t in [TAG_ALPHA, TAG_BETA, TAG_QMIN] if t not in tb]
    print(f"\nWARNING: missing TB tags: {missing}")
    print("Check the exact tag names in your TB — edit TAG_ALPHA/TAG_BETA/TAG_QMIN "
          "at the top of this script if they differ.")


# ============================================================
# Plotting
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")

def smooth(s, w=SMOOTH_W):
    return s.rolling(w, min_periods=1).mean()

def smooth_arr(values, w=50):
    return pd.Series(values).rolling(w, min_periods=1).mean().values


# --- Figure 1: convergence ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x_ep = df_csv["step"] / 1e6

# (a) Rolling mean reward
ax = axes[0]
ax.plot(x_ep, df_csv["mean_reward_roll"], color="seagreen",
        linewidth=0.7, alpha=0.30)
ax.plot(x_ep, smooth(df_csv["mean_reward_roll"]),
        color="seagreen", linewidth=2.2)
ax.set_ylabel("Mean episode reward")
ax.set_xlabel("Environment step (millions)")
ax.set_title("(a) Rolling mean episode reward")
ax.axvline(1.15, color="gray", linestyle="--", linewidth=0.8)

# (b) Rolling success rate
ax = axes[1]
ax.plot(x_ep, df_csv["success_roll"], color="steelblue",
        linewidth=0.7, alpha=0.30)
ax.plot(x_ep, smooth(df_csv["success_roll"]),
        color="steelblue", linewidth=2.2)
ax.set_ylabel("Success rate")
ax.set_xlabel("Environment step (millions)")
ax.set_title("(b) Rolling success rate")
ax.set_ylim(0, 1.0)
ax.axvline(1.15, color="gray", linestyle="--", linewidth=0.8)

# (c) alpha
ax = axes[2]
if TAG_ALPHA in tb:
    df = tb[TAG_ALPHA]
    x = df["step"] / 1e6
    ax.plot(x, df["value"], color="C0", linewidth=0.7, alpha=0.30)
    ax.plot(x, smooth_arr(df["value"], w=100), color="C0", linewidth=2.2)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel("Environment step (millions)")
ax.set_title(r"(c) Entropy temperature $\alpha$")
ax.axvline(1.15, color="gray", linestyle="--", linewidth=0.8)

for ax in axes:
    ax.set_xlim(0, 1.20)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig1_path = outdir / "training_convergence.png"
plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig1_path}")


# --- Figure 2: stability ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) beta
ax = axes[0]
if TAG_BETA in tb:
    df = tb[TAG_BETA]
    ax.plot(df["step"] / 1e6, df["value"],
            color="mediumpurple", linewidth=2.2)
ax.set_ylabel(r"$\beta$")
ax.set_xlabel("Environment step (millions)")
ax.set_title(r"(a) PER exponent $\beta$ annealing schedule")
ax.set_ylim(0.35, 1.08)
ax.axvline(1.15, color="gray", linestyle="--", linewidth=0.8)
ax.text(1.16, 0.42, "1.15M\n(eval)", fontsize=8, color="gray")

# (b) q_min
ax = axes[1]
if TAG_QMIN in tb:
    df = tb[TAG_QMIN]
    x = df["step"] / 1e6
    ax.plot(x, df["value"], color="C1", linewidth=0.7, alpha=0.30)
    ax.plot(x, smooth_arr(df["value"], w=100), color="C1", linewidth=2.2)
ax.set_ylabel(r"$\min_i Q_{\theta_i}$ (batch mean)")
ax.set_xlabel("Environment step (millions)")
ax.set_title("(b) Critic Q-value (twin-minimum)")
ax.axvline(1.15, color="gray", linestyle="--", linewidth=0.8)

for ax in axes:
    ax.set_xlim(0, 1.20)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig2_path = outdir / "training_stability.png"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")


# ============================================================
# Print the numbers to quote in the text
# ============================================================
print("\n" + "=" * 60)
print("NUMBERS TO QUOTE IN SECTION 5.3")
print("=" * 60)
tail = df_csv.tail(ROLL_WINDOW)
print(f"  Reward at ~200k plateau start : "
      f"{smooth(df_csv['mean_reward_roll']).iloc[len(df_csv)//5]:.1f}")
print(f"  Reward at end of training     : "
      f"{smooth(df_csv['mean_reward_roll']).iloc[-1]:.1f}")
print(f"  Success rate (last 100 ep)    : {tail['success_roll'].mean():.3f}")
if TAG_ALPHA in tb:
    a_start = tb[TAG_ALPHA]["value"].iloc[0]
    a_end   = tb[TAG_ALPHA]["value"].iloc[-1]
    print(f"  Alpha: initial {a_start:.3f} → final {a_end:.3f}")
if TAG_QMIN in tb:
    q_end = smooth_arr(tb[TAG_QMIN]["value"].values, w=100)[-1]
    print(f"  Q-value plateau (final smoothed): {q_end:.2f}")