"""Control Barrier Function safety shield for CM-GAP_SAC.

Given a SAC-proposed action u_SAC = (v, omega), this module solves a small
quadratic program to project u_SAC onto the safe set defined by Control
Barrier Functions for each obstacle:

    h_k(x) = ||p_robot - p_obstacle_k||^2 - r_safe_k^2 >= 0

The CBF condition that guarantees forward invariance of the safe set is:

    h_dot_k(x, u) + gamma * h_k(x) >= 0

For a unicycle robot with state (x, y, theta) and control u = (v, omega):

    p_robot_dot = (v cos(theta), v sin(theta))
    h_dot_k(x, u) = 2 * (p_robot - p_obstacle_k)^T * p_robot_dot
                  = 2 * v * [(x - x_k) cos(theta) + (y - y_k) sin(theta)]

This is linear in v (and independent of omega for a static obstacle in the
position constraint). For pedestrians with known velocity p_obstacle_dot_k,
we extend:

    h_dot_k = 2 * (p_robot - p_obstacle_k)^T * (p_robot_dot - p_obstacle_dot_k)

The QP minimizes ||u - u_SAC||^2 subject to one linear inequality per
nearby obstacle. OSQP solves this in well under 1 ms on CPU.

Per-obstacle-class safety radius (NEW):
    A single r_safe for walls AND pedestrians is geometrically wrong here.
    A pedestrian is a moving, unpredictable, human-safety concern: it
    deserves a generous margin (r_safe_ped ~ 0.50). A wall is a fixed,
    perfectly-known structure with no human stake: the robot may pass close
    to it safely, so a tight margin (r_safe_static ~ 0.25) suffices. With a
    single large r_safe, a 1.2 m doorway leaves only 1.2 - 2*0.50 = 0.20 m
    of free corridor at its center, which the robot (~0.32 m wide) cannot
    fit through -> the QP becomes infeasible and the robot freezes. Since
    pedestrians stay out of the doorways (they remain in their rooms), the
    doorway constraint is purely static, so a tight r_safe_static reopens
    the passage (1.2 - 2*0.25 = 0.70 m free) while r_safe_ped keeps full
    margin around people in the corridor.

Optimal-decay CBF-QP:
    On top of the per-class radii, the barrier gain is a *decision variable*
    of the QP (Zeng et al. 2021):

        A * u + omega_decay * (gamma * h) - moving_term >= 0
        cost += omega_decay_penalty * (omega_decay - 1)^2

    omega_decay in [omega_decay_min, 1]. The solver keeps it ~ 1 whenever
    feasible, and relaxes it toward omega_decay_min only when that is the
    sole way to stay feasible. The barrier is never removed.

References:
    - Ames et al. 2017, 2019 (Control Barrier Functions theory)
    - Stellato et al. 2020 (OSQP solver)
    - Cheng et al. 2019, Emam et al. 2022 (CBF + RL integration)
    - Zeng et al. 2021 (Optimal-decay CBF-QP, the decision-variable gain)

Limitations and honest design choices:
    - Pure position CBF, not full second-order. For unicycle, omega does
      NOT directly constrain h_dot in our simple formulation. Conscious
      simplification avoiding relative-degree-2 (HOCBF). The shield filters
      v but lets omega pass through; avoidance happens via velocity cut.
    - An anti-pivot post-step bounds omega when v is crushed to ~0, to
      avoid the degenerate spin-in-place mode (v~0, omega~max).
    - LiDAR rays beyond `active_radius` are pruned to keep the QP small.
    - Static obstacles assume p_obstacle_dot = 0. Pedestrian velocity is
      taken from the tracker.
    - Emergency stop u = (0, 0) only if the QP is infeasible even with the
      barrier relaxed to its minimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp


# ======================================================================
@dataclass
class ShieldResult:
    """Output of one shield call."""
    safe_action: np.ndarray          # (2,)
    was_modified: bool               # True if shield changed the action
    n_active_constraints: int        # how many obstacles were considered
    infeasible: bool                 # True if QP failed -> emergency stop
    delta_action: np.ndarray         # u_safe - u_SAC, shape (2,)
    omega_decay: float               # barrier relaxation in [min, 1]


# ======================================================================
class CbfSafetyShield:
    """OSQP-based CBF QP shield with per-class radii and optimal-decay.

    Usage:
        shield = CbfSafetyShield(
            r_safe_static=0.25, r_safe_ped=0.50, gamma=2.0, active_radius=1.0)
        result = shield.filter(
            u_sac=action,
            robot_xy_yaw=(x, y, theta),
            lidar_scan=lidar_array,        # (n_beams,) ranges
            pedestrians_rel=ped_features,  # (K, 5)
            ped_mask=ped_mask,             # (K,)
        )
        env.step(result.safe_action)
    """

    def __init__(
        self,
        r_safe_static: float = 0.25,   # tight margin vs walls / furniture
        r_safe_ped: float = 0.50,      # generous margin vs pedestrians
        gamma: float = 2.0,
        active_radius: float = 1.0,
        v_min: float = -0.3,
        v_max: float = 0.5,
        omega_min: float = -1.0,
        omega_max: float = 1.0,
        n_lidar_keep: int = 12,
        emergency_stop_on_infeasible: bool = True,
        osqp_max_iter: int = 200,
        osqp_eps_abs: float = 1e-4,
        cbf_v_min_shield: float = 0.0,
        omega_gate_v: float = 0.15,
        omega_min_floor: float = 0.2,
        omega_decay_min: float = 0.1,
        omega_decay_penalty: float = 10.0,
    ) -> None:
        self.r_safe_static = float(r_safe_static)
        self.r_safe_ped = float(r_safe_ped)
        self.r_safe_static_sq = float(r_safe_static) ** 2
        self.r_safe_ped_sq = float(r_safe_ped) ** 2
        self.gamma = float(gamma)
        self.active_radius = float(active_radius)
        self.v_bounds = (float(v_min), float(v_max))
        self.cbf_v_min_shield = float(cbf_v_min_shield)
        self.omega_bounds = (float(omega_min), float(omega_max))
        self.n_lidar_keep = int(n_lidar_keep)
        self.emergency_stop_on_infeasible = bool(emergency_stop_on_infeasible)
        self.osqp_max_iter = int(osqp_max_iter)
        self.osqp_eps_abs = float(osqp_eps_abs)

        # Anti rotation-sur-place (post-QP bound on omega).
        self._omega_gate_v = float(omega_gate_v)
        self._omega_min_floor = float(omega_min_floor)

        # Optimal-decay parameters.
        self._omega_decay_min = float(omega_decay_min)
        self._omega_decay_penalty = float(omega_decay_penalty)

        try:
            import osqp
        except ImportError as e:
            raise ImportError(
                "osqp is required for CbfSafetyShield. "
                "Install with: pip install osqp"
            ) from e
        self._osqp = osqp

    # ------------------------------------------------------------------
    def _lidar_to_obstacles_robot_frame(
        self, lidar: np.ndarray,
    ) -> np.ndarray:
        """Convert downsampled lidar scan to (N, 2) obstacle points in robot frame.

        Scan spans 360 deg uniformly, counter-clockwise from forward (matches
        the env's downsample_lidar convention). Returns only points within
        active_radius; keeps the n_lidar_keep nearest if too many.
        """
        n = lidar.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
        mask = (lidar > 0.05) & (lidar < self.active_radius)
        if not mask.any():
            return np.zeros((0, 2), dtype=np.float32)

        ranges = lidar[mask]
        ang = angles[mask]
        xs = ranges * np.cos(ang)
        ys = ranges * np.sin(ang)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        if pts.shape[0] > self.n_lidar_keep:
            d2 = (pts ** 2).sum(axis=1)
            idx = np.argpartition(d2, self.n_lidar_keep)[: self.n_lidar_keep]
            pts = pts[idx]

        return pts

    # ------------------------------------------------------------------
    def _build_constraints(
        self,
        robot_xy_yaw: Tuple[float, float, float],
        static_obstacles_xy: np.ndarray,    # (N_s, 2) WORLD frame (walls/furniture)
        ped_xy_world:        np.ndarray,    # (N_p, 2) WORLD frame
        ped_vel_world:       np.ndarray,    # (N_p, 2) WORLD frame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the CBF constraints for the optimal-decay QP.

        Each obstacle row, with the barrier gain as decision var omega_decay:
            C_uw_row . (v, omega) + c_dec_row * omega_decay >= rhs_row

        where:
            coef_v      = 2 [(x-x_k)cos(theta) + (y-y_k)sin(theta)]
            barrier     = gamma * h_k,  h_k = ||p-p_k||^2 - r_safe_k^2
                          -> r_safe_k is r_safe_static for the first N_s rows
                             (walls/furniture), r_safe_ped for the rest (peds)
            moving_term = 2 (p-p_k)^T v_k   (physical, NOT scaled by omega_decay)

        Returns C_uw (N,2), c_dec (N,), rhs (N,).
        """
        x, y, theta = robot_xy_yaw
        cth, sth = float(np.cos(theta)), float(np.sin(theta))

        N_s = static_obstacles_xy.shape[0]
        N_p = ped_xy_world.shape[0]
        N = N_s + N_p
        if N == 0:
            return (np.zeros((0, 2), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64))

        all_xy = np.zeros((N, 2), dtype=np.float64)
        all_v  = np.zeros((N, 2), dtype=np.float64)
        # Per-row squared safety radius: static for walls, ped for pedestrians.
        r_safe_sq = np.empty(N, dtype=np.float64)
        if N_s > 0:
            all_xy[:N_s] = static_obstacles_xy
            r_safe_sq[:N_s] = self.r_safe_static_sq
        if N_p > 0:
            all_xy[N_s:] = ped_xy_world
            all_v[N_s:]  = ped_vel_world
            r_safe_sq[N_s:] = self.r_safe_ped_sq

        dx = x - all_xy[:, 0]
        dy = y - all_xy[:, 1]

        h = dx * dx + dy * dy - r_safe_sq                     # (N,) per-class
        coef_v = 2.0 * (dx * cth + dy * sth)                  # (N,)
        moving_term = 2.0 * (dx * all_v[:, 0] + dy * all_v[:, 1])  # (N,)

        C_uw = np.zeros((N, 2), dtype=np.float64)
        C_uw[:, 0] = coef_v
        C_uw[:, 1] = 0.0                                      # omega unconstrained
        c_dec = self.gamma * h                                # scaled by omega_decay
        rhs = moving_term

        return C_uw, c_dec, rhs

    # ------------------------------------------------------------------
    def filter(
        self,
        u_sac: np.ndarray,                        # (2,)
        robot_xy_yaw: Tuple[float, float, float],
        lidar_scan: Optional[np.ndarray] = None,  # (n_beams,) robot frame
        pedestrians_rel: Optional[np.ndarray] = None,  # (K, 5)
        ped_mask:        Optional[np.ndarray] = None,  # (K,)
    ) -> ShieldResult:
        """Project u_sac onto the (per-class, optimal-decay) safe set.

        Decision variable z = (v, omega, omega_decay), 3-dimensional.
        Cost: ||(v,omega) - u_sac||^2 + p * (omega_decay - 1)^2.
        """
        u_sac = np.asarray(u_sac, dtype=np.float64).reshape(2)

        # 1. Static obstacles from lidar (robot frame -> world frame).
        x, y, theta = robot_xy_yaw
        if lidar_scan is not None:
            obs_robot = self._lidar_to_obstacles_robot_frame(
                np.asarray(lidar_scan, dtype=np.float32)
            )
            if obs_robot.shape[0] > 0:
                cth, sth = np.cos(theta), np.sin(theta)
                R = np.array([[cth, -sth], [sth, cth]], dtype=np.float64)
                obs_world = obs_robot.astype(np.float64) @ R.T + np.array([x, y])
            else:
                obs_world = np.zeros((0, 2), dtype=np.float64)
        else:
            obs_world = np.zeros((0, 2), dtype=np.float64)

        # 2. Pedestrians (relative -> world).
        ped_world_xy  = np.zeros((0, 2), dtype=np.float64)
        ped_world_vel = np.zeros((0, 2), dtype=np.float64)
        if pedestrians_rel is not None and ped_mask is not None:
            mask_b = np.asarray(ped_mask, dtype=bool)
            if mask_b.any():
                peds = np.asarray(pedestrians_rel)[mask_b]      # (M, 5)
                cth, sth = np.cos(theta), np.sin(theta)
                R = np.array([[cth, -sth], [sth, cth]], dtype=np.float64)
                ped_world_xy  = peds[:, :2].astype(np.float64) @ R.T + np.array([x, y])
                ped_world_vel = peds[:, 2:4].astype(np.float64) @ R.T

        # 3. Build constraints (per-class radii, optimal-decay form).
        C_uw, c_dec, rhs = self._build_constraints(
            robot_xy_yaw=robot_xy_yaw,
            static_obstacles_xy=obs_world,
            ped_xy_world=ped_world_xy,
            ped_vel_world=ped_world_vel,
        )
        n_active = int(C_uw.shape[0])

        # 4. Solve QP. z = (v, omega, omega_decay).
        #    min 0.5 z^T P z + q^T z :
        #      ||(v,omega)-u_sac||^2 -> P_uw = 2 I_2, q_uw = -2 u_sac
        #      p (omega_decay-1)^2   -> P_dd = 2 p,   q_dd = -2 p
        #    Box: v in [v_min_shield, v_max], omega in [w_min,w_max],
        #         omega_decay in [omega_decay_min, 1]
        #    Safety: C_uw u + c_dec * omega_decay >= rhs

        _, v_max = self.v_bounds
        v_min = self.cbf_v_min_shield
        w_min, w_max = self.omega_bounds
        p = self._omega_decay_penalty
        wd_min = self._omega_decay_min

        P = sp.csc_matrix(np.diag([2.0, 2.0, 2.0 * p]))
        q = np.array([-2.0 * u_sac[0], -2.0 * u_sac[1], -2.0 * p],
                     dtype=np.float64)

        box = sp.eye(3, format="csc")
        box_l = np.array([v_min, w_min, wd_min], dtype=np.float64)
        box_u = np.array([v_max, w_max, 1.0], dtype=np.float64)

        if n_active > 0:
            safety = sp.csc_matrix(
                np.hstack([C_uw, c_dec.reshape(-1, 1)])
            )
            C = sp.vstack([box, safety], format="csc")
            l = np.concatenate([box_l, rhs])
            u = np.concatenate([box_u, np.full(n_active, np.inf)])
        else:
            C = box
            l = box_l
            u = box_u

        prob = self._osqp.OSQP()
        prob.setup(
            P=P, q=q, A=C, l=l, u=u,
            verbose=False,
            max_iter=self.osqp_max_iter,
            eps_abs=self.osqp_eps_abs,
            eps_rel=self.osqp_eps_abs,
            polishing=True,
        )
        sol = prob.solve()

        status = getattr(sol.info, "status", "unknown")
        success = ("solved" in str(status).lower())

        if success and sol.x is not None:
            z = np.asarray(sol.x, dtype=np.float64).reshape(3)
            safe = z[:2].copy()
            omega_decay = float(np.clip(z[2], wd_min, 1.0))
            safe[0] = np.clip(safe[0], v_min, v_max)
            safe[1] = np.clip(safe[1], w_min, w_max)
            infeasible = False
        else:
            if self.emergency_stop_on_infeasible:
                safe = np.array([0.0, 0.0], dtype=np.float64)
            else:
                safe = u_sac.copy()
            omega_decay = wd_min
            infeasible = True

        # ── Anti rotation-sur-place ──────────────────────────────────
        v_abs = abs(safe[0])
        if v_abs < self._omega_gate_v:
            scale = v_abs / self._omega_gate_v
            omega_cap = (self._omega_min_floor
                         + scale * (w_max - self._omega_min_floor))
            safe[1] = float(np.clip(safe[1], -omega_cap, omega_cap))
        # ─────────────────────────────────────────────────────────────

        delta = safe - u_sac
        was_modified = bool(np.linalg.norm(delta) > 1e-4)

        return ShieldResult(
            safe_action=safe.astype(np.float32),
            was_modified=was_modified,
            n_active_constraints=n_active,
            infeasible=infeasible,
            delta_action=delta.astype(np.float32),
            omega_decay=omega_decay,
        )


# ======================================================================
# Factory
# ======================================================================
def build_shield_from_config(cfg) -> Optional[CbfSafetyShield]:
    """Build a shield from config; return None if cbf.enabled is False.

    Backward-compatible: if the YAML still has a single `r_safe`, it is used
    for BOTH classes. Prefer the new `r_safe_static` / `r_safe_ped` keys.
    """
    cbf_cfg = cfg.cbf
    if not cbf_cfg.get("enabled", True):
        return None

    # Per-class radii with fallback to a legacy single r_safe.
    legacy = cbf_cfg.get("r_safe", 0.50)
    r_safe_static = cbf_cfg.get("r_safe_static", min(legacy, 0.25))
    r_safe_ped = cbf_cfg.get("r_safe_ped", legacy)

    return CbfSafetyShield(
        r_safe_static=r_safe_static,
        r_safe_ped=r_safe_ped,
        gamma=cbf_cfg["gamma_cbf"],
        active_radius=cbf_cfg["active_radius"],
        v_min=cfg.robot.v_min, v_max=cfg.robot.v_max,
        omega_min=cfg.robot.omega_min, omega_max=cfg.robot.omega_max,
        emergency_stop_on_infeasible=cbf_cfg.get(
            "emergency_stop_on_infeasible", True,
        ),
        cbf_v_min_shield=cbf_cfg.get("v_min_shield", 0.0),
        omega_gate_v=cbf_cfg.get("omega_gate_v", 0.15),
        omega_min_floor=cbf_cfg.get("omega_min_floor", 0.2),
        omega_decay_min=cbf_cfg.get("omega_decay_min", 0.1),
        omega_decay_penalty=cbf_cfg.get("omega_decay_penalty", 10.0),
    )