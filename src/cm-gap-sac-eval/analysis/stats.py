"""
Statistical utilities for CM-GAP-SAC eval analysis.

Depends only on numpy + scipy — no ROS/Gazebo.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes: int, trials: int,
              confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Reliable at small N and extreme rates (unlike normal approximation).

    Returns (proportion, lower_bound, upper_bound).
    """
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (p, max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def mcnemar_test(a_wins: int, b_wins: int) -> dict:
    """McNemar's test for paired binary outcomes.

    a_wins = # pairs where method A succeeded and method B failed
    b_wins = # pairs where B succeeded and A failed
    (Pairs where both agree are ignored — that's the point of McNemar.)

    Uses exact binomial for small N (<25 discordant), chi-square with
    continuity correction otherwise.

    Returns {statistic, p_value, significant, a_wins, b_wins, discordant}.
    """
    discordant = a_wins + b_wins
    if discordant == 0:
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "a_wins": a_wins, "b_wins": b_wins, "discordant": 0,
        }

    if discordant < 25:
        # Exact binomial (two-sided)
        k = min(a_wins, b_wins)
        p = 2 * float(stats.binom.cdf(k, discordant, 0.5))
        p = min(p, 1.0)
        return {
            "statistic": None, "p_value": p, "significant": p < 0.05,
            "a_wins": a_wins, "b_wins": b_wins, "discordant": discordant,
        }
    else:
        # Chi-square with continuity correction
        stat = (abs(a_wins - b_wins) - 1) ** 2 / discordant
        p = 1 - float(stats.chi2.cdf(stat, 1))
        return {
            "statistic": float(stat), "p_value": p, "significant": p < 0.05,
            "a_wins": a_wins, "b_wins": b_wins, "discordant": discordant,
        }


def paired_binary_from_frames(df_a: pd.DataFrame, df_b: pd.DataFrame,
                              join_key: str = "episode_seed",
                              outcome_col: str = "outcome_success") -> dict:
    """Compute McNemar between two frames aligned on join_key."""
    merged = df_a[[join_key, outcome_col]].merge(
        df_b[[join_key, outcome_col]], on=join_key, suffixes=("_a", "_b")
    )
    a_col = f"{outcome_col}_a"
    b_col = f"{outcome_col}_b"
    a_wins = int((merged[a_col] & ~merged[b_col]).sum())
    b_wins = int((~merged[a_col] & merged[b_col]).sum())
    return mcnemar_test(a_wins, b_wins)


def format_rate_ci(mean: float, lo: float, hi: float, pct: bool = True) -> str:
    """Format proportion with CI for tables: '87.0% [80.0, 92.0]'."""
    if pct:
        return f"{100*mean:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def bootstrap_median_ci(values: np.ndarray, confidence: float = 0.95,
                        n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for median — used for continuous metrics (time-to-goal, etc.)."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    medians = np.empty(n_resamples)
    n = len(values)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)

    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(medians, [alpha, 1 - alpha])
    return (float(np.median(values)), float(lo), float(hi))
