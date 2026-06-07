"""Hand-crafted geometric reflex for warm-start regularization.

Inspired by Khatib 1986 (potential fields) and kept intentionally
minimal. The reflex provides SAC with a reasonable baseline behavior
during early training; its influence on the policy decays exponentially
(see reflex_shaping below), so the asymptotic policy is pure SAC and
the optimality guarantees of Haarnoja et al. 2018 are preserved
(Ng et al. 1999 on potential-based shaping invariance).

v0.3 change:
    Added an obstacle-repulsion term to omega. Previously omega only
    pointed toward the goal (pure attraction), so when an obstacle sat
    between the robot and the goal the reflex drove straight at it and
    only decelerated, producing a stop-and-go pattern with no lateral
    escape. The repulsion term steers the robot AROUND the nearest
    front obstacle, giving SAC a sensible avoidance prior to imitate.
"""
import numpy as np


def reflex_action(
    obs: dict,
    v_max: float = 0.5,
    omega_max: float = 1.0,
    k_omega: float = 1.0,
    r_slowdown: float = 1.5,
    n_front: int = 30,
    k_repulse: float = 1.5,
    r_influence: float = 1.0,
) -> np.ndarray:
    """P-controller on heading (attraction + repulsion), speed gated.

    omega = attraction toward goal + repulsion from nearest front obstacle.
        attraction : k_omega * theta_goal
        repulsion  : turns away from the closest obstacle within
                     r_influence; magnitude grows as it gets closer.

    Speed is gated by front clearance (slow near obstacles) and by goal
    alignment (slow when not yet facing the goal).

    LiDAR convention assumed (matches the SDF gpu_lidar and the original
    reflex): the scan spans [-pi, +pi], index 0 = rear, the middle index
    = front (angle 0). The front cone is the n_front beams either side of
    the middle, identical to the slice the original reflex used.
    """
    theta_goal = float(obs["goal"][1])
    lidar = np.asarray(obs["lidar"], dtype=np.float32)
    n = len(lidar)

    angles = np.linspace(-np.pi, np.pi, n)

    mid = n // 2
    lo = max(0, mid - n_front)
    hi = min(n, mid + n_front)
    front_ranges = lidar[lo:hi]
    front_angles = angles[lo:hi]

    # --- Attraction: steer toward the goal ---
    omega_attract = k_omega * theta_goal

    # --- Repulsion: steer away from the nearest front obstacle ---
    omega_repulse = 0.0
    j = int(np.argmin(front_ranges))
    d_closest = float(front_ranges[j])
    a_closest = float(front_angles[j])
    if d_closest < r_influence:
        # Linear ramp: 0 at r_influence, 1 when touching.
        strength = (r_influence - d_closest) / r_influence
        if abs(a_closest) < 0.10:
            # Obstacle dead ahead: no natural side to turn, so escape
            # toward the goal side (turn the short way around).
            escape = np.sign(theta_goal) if abs(theta_goal) > 1e-3 else 1.0
            omega_repulse = k_repulse * strength * escape
        else:
            # Obstacle on the left (a_closest > 0) -> turn right (omega < 0).
            # Obstacle on the right (a_closest < 0) -> turn left (omega > 0).
            omega_repulse = -k_repulse * strength * np.sign(a_closest)

    omega = float(np.clip(omega_attract + omega_repulse, -omega_max, omega_max))

    # --- Speed: slow near obstacles and when misaligned with goal ---
    front_clearance = float(front_ranges.min())
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