"""
Render Figure 1 (main figure) as 3 separate panel PNGs, then paste them
together side by side into a single combined image.

Panels:
    1. Rolling mean reward
    2. Rolling success rate, smoothed (shows convergence toward a target)
    3. train/alpha (SAC entropy-coefficient auto-tuning)

Expects CSVs with columns: Wall time,Step,Value
Matched by filename suffix inside --indir, e.g.:
    <run>_mean_reward.csv
    <run>_success_rate.csv
    <run>_train_alpha.csv   (or <run>_alpha.csv, tried as fallback)

Usage:
    python figure1_main.py --outdir ./figure1_out
    python figure1_main.py --indir . --outdir ./figure1_out --smooth-window 200 --success-target 0.65
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


REWARD_SUFFIX = "_mean_reward.csv"
SUCCESS_SUFFIX = "_success_rate.csv"
ALPHA_SUFFIXES = ["_train_alpha.csv", "_alpha.csv"]  # tried in order


def find_file(indir: Path, suffixes) -> Path:
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    for suffix in suffixes:
        matches = sorted(indir.glob(f"*{suffix}"))
        if matches:
            if len(matches) > 1:
                print(f"  Warning: multiple files match '{suffix}', using {matches[0].name}")
            return matches[0]
    raise FileNotFoundError(f"No file found for suffix(es) {suffixes} in {indir}")


def load_metric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Wall time", "Step", "Value"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path.name}: expected columns {expected}, got {list(df.columns)}")
    return df


def save_panel(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved panel: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default=".",
                         help="Directory containing the CSV exports (default: current dir)")
    parser.add_argument("--outdir", type=str, default="./figure1_out",
                         help="Where to write the panel PNGs and the combined figure")
    parser.add_argument("--smooth-window", type=int, default=200,
                         help="Rolling window (in points) used to smooth the success-rate panel")
    parser.add_argument("--success-target", type=float, default=0.65,
                         help="Reference line drawn on the success-rate panel")
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    print(f"Looking for CSVs in {indir}")
    reward_path = find_file(indir, REWARD_SUFFIX)
    success_path = find_file(indir, SUCCESS_SUFFIX)
    alpha_path = find_file(indir, ALPHA_SUFFIXES)

    reward_df = load_metric(reward_path)
    success_df = load_metric(success_path)
    alpha_df = load_metric(alpha_path)

    for name, path, df in [
        ("mean_reward", reward_path, reward_df),
        ("success_rate", success_path, success_df),
        ("alpha", alpha_path, alpha_df),
    ]:
        print(f"  {name:14s} <- {path.name:60s} "
              f"({len(df):,} pts, step range {df['Step'].min():,}-{df['Step'].max():,})")

    panel_paths = []

    # ------------------------------------------------------------
    # Panel 1 — Rolling mean reward
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(reward_df["Step"], reward_df["Value"], color="C2", linewidth=1.3)
    ax1.set_xlabel("Environment step")
    ax1.set_ylabel("Mean episode reward")
    ax1.set_title("Rolling mean reward")
    ax1.grid(alpha=0.3)
    p1 = outdir / "panel1_mean_reward.png"
    save_panel(fig1, p1)
    panel_paths.append(p1)
    plt.close(fig1)

    # ------------------------------------------------------------
    # Panel 2 — Rolling success rate, smoothed (+ target line)
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    smoothed = success_df["Value"].rolling(args.smooth_window, min_periods=1).mean()
    ax2.plot(success_df["Step"], success_df["Value"], color="C0", alpha=0.25, linewidth=0.8)
    ax2.plot(success_df["Step"], smoothed, color="C0", linewidth=1.8)
    ax2.axhline(args.success_target, color="gray", linestyle="--", linewidth=1)
    ax2.text(success_df["Step"].iloc[-1], args.success_target, f" target: {args.success_target:.2f}",
              va="bottom", ha="right", color="gray", fontsize=9)
    ax2.set_xlabel("Environment step")
    ax2.set_ylabel("Success rate")
    ax2.set_title(f"Rolling success rate (smoothed, w={args.smooth_window})")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    p2 = outdir / "panel2_success_rate.png"
    save_panel(fig2, p2)
    panel_paths.append(p2)
    plt.close(fig2)

    # ------------------------------------------------------------
    # Panel 3 — train/alpha (SAC auto-tuning)
    # ------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(alpha_df["Step"], alpha_df["Value"], color="C4", linewidth=1.3)
    ax3.set_xlabel("Environment step")
    ax3.set_ylabel(r"$\alpha$")
    ax3.set_title("train/alpha (SAC entropy coefficient)")
    ax3.grid(alpha=0.3)
    p3 = outdir / "panel3_alpha.png"
    save_panel(fig3, p3)
    panel_paths.append(p3)
    plt.close(fig3)

    # ------------------------------------------------------------
    # Paste the 3 panels together into the main Figure 1
    # ------------------------------------------------------------
    images = [Image.open(p) for p in panel_paths]
    heights = [im.height for im in images]
    max_h = max(heights)
    # pad any shorter image to align tops
    padded = []
    for im in images:
        if im.height < max_h:
            canvas = Image.new("RGB", (im.width, max_h), "white")
            canvas.paste(im, (0, 0))
            padded.append(canvas)
        else:
            padded.append(im)

    total_w = sum(im.width for im in padded)
    combined = Image.new("RGB", (total_w, max_h), "white")
    x_offset = 0
    for im in padded:
        combined.paste(im, (x_offset, 0))
        x_offset += im.width

    fig_out = outdir / "figure1_main.png"
    combined.save(fig_out)
    print(f"\nSaved combined Figure 1: {fig_out}")


if __name__ == "__main__":
    main()