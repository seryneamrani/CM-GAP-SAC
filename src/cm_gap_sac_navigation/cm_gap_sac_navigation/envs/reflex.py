"""Hand-crafted geometric reflex for warm-start regularization.

Inspired by Khatib 1986 (potential fields) and kept intentionally
minimal. The reflex provides SAC with a reasonable baseline behavior
during early training; its influence on the policy decays exponentially
(see reflex_shaping below), so the asymptotic policy is pure SAC and
the optimality guarantees of Haarnoja et al. 2018 are preserved
(Ng et al. 1999 on potential-based shaping invariance).
"""
import numpy as np

def reflex_action(
    obs: dict,
    v_max: float = 0.5,
    omega_max: float = 1.0,
    k_omega: float = 1.0,
    r_slowdown: float = 1.5,
    n_front: int = 30,
) -> np.ndarray:
    """P-controller on heading, speed gated by alignment and front clearance."""
    theta_goal = float(obs["goal"][1])
    omega = float(np.clip(k_omega * theta_goal, -omega_max, omega_max))

    lidar = obs["lidar"]
    mid = len(lidar) // 2
    front_clearance = float(lidar[mid - n_front: mid + n_front].min())
    speed_factor = float(np.clip(front_clearance / r_slowdown, 0.0, 1.0))
    align_factor = max(0.0, float(np.cos(theta_goal)))

    v = v_max * speed_factor * align_factor
    return np.array([v, omega], dtype=np.float32)


def reflex_shaping(
    action_sac: np.ndarray,
    action_reflex: np.ndarray,
    global_step: int,
    beta_0: float = 0.5,
    t_anneal: int = 50000,
) -> float:
    """Exponentially-decaying penalty on deviation from the reflex."""
    lam = beta_0 * np.exp(-global_step / max(t_anneal, 1))
    delta = float(np.linalg.norm(action_sac - action_reflex))
    return -lam * delta