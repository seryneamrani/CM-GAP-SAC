"""Control Barrier Function safety shield for CM-GAP_SAC.

Given a SAC-proposed action u_SAC = (v, omega), this module solves a small
quadratic program to project u_SAC onto the safe set defined by Control
Barrier Functions for each obstacle:

    h_k(x) = ||p_robot - p_obstacle_k||^2 - r_safe^2 >= 0

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

References:
    - Ames et al. 2017, 2019 (Control Barrier Functions theory)
    - Stellato et al. 2020 (OSQP solver)
    - Cheng et al. 2019, Emam et al. 2022 (CBF + RL integration)

Limitations and honest design choices:
    - Pure position CBF, not full second-order. For unicycle, omega does
      NOT directly constrain h_dot in our simple formulation. This is a
      conscious simplification that avoids relative-degree-2 (HOCBF)
      complications. In practice this means the shield filters v but
      lets omega pass through; collision avoidance happens via velocity
      reduction, which is what we want.
    - LiDAR rays beyond `active_radius` are pruned to keep the QP small.
    - Static obstacles assume p_obstacle_dot = 0. Pedestrian velocity is
      taken from the tracker.
    - If the QP is infeasible (no safe action exists), we issue an
      emergency stop u = (0, 0). This is the safest default.
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


# ======================================================================
class CbfSafetyShield:
    """OSQP-based CBF QP shield.

    Usage:
        shield = CbfSafetyShield(r_safe=0.30, gamma=2.0, active_radius=2.0)
        result = shield.filter(
            u_sac=action,
            robot_xy_yaw=(x, y, theta),
            lidar_scan=lidar_array,        # (n_beams,) ranges
            pedestrians_rel=ped_features,  # (K, 5) [x_rel, y_rel, vx_rel, vy_rel, age]
            ped_mask=ped_mask,             # (K,) 1=valid
        )
        env.step(result.safe_action)
    """

    def __init__(
        self,
        r_safe: float = 0.30,
        gamma: float = 2.0,
        active_radius: float = 2.0,
        v_min: float = -0.3,
        v_max: float = 0.6,
        omega_min: float = -1.0,
        omega_max: float = 1.0,
        n_lidar_keep: int = 12,
        emergency_stop_on_infeasible: bool = True,
        osqp_max_iter: int = 200,
        osqp_eps_abs: float = 1e-4,
    ) -> None:
        self.r_safe = float(r_safe)
        self.r_safe_sq = float(r_safe) ** 2
        self.gamma = float(gamma)
        self.active_radius = float(active_radius)
        self.v_bounds = (float(v_min), float(v_max))
        self.omega_bounds = (float(omega_min), float(omega_max))
        self.n_lidar_keep = int(n_lidar_keep)
        self.emergency_stop_on_infeasible = bool(emergency_stop_on_infeasible)
        self.osqp_max_iter = int(osqp_max_iter)
        self.osqp_eps_abs = float(osqp_eps_abs)

        # Lazy import: only fail if shield is actually instantiated.
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

        Scan is assumed to span 360 degrees uniformly, indexed counter-clockwise
        starting from the robot's forward direction (this matches our env's
        downsample_lidar convention).

        Returns only points within active_radius. If more than n_lidar_keep
        remain, keeps the n_lidar_keep nearest.
        """
        n = lidar.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
        # Filter by active radius first.
        mask = (lidar > 0.05) & (lidar < self.active_radius)
        if not mask.any():
            return np.zeros((0, 2), dtype=np.float32)

        ranges = lidar[mask]
        ang = angles[mask]
        xs = ranges * np.cos(ang)
        ys = ranges * np.sin(ang)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        # Keep the N nearest if too many.
        if pts.shape[0] > self.n_lidar_keep:
            d2 = (pts ** 2).sum(axis=1)
            idx = np.argpartition(d2, self.n_lidar_keep)[: self.n_lidar_keep]
            pts = pts[idx]

        return pts

    # ------------------------------------------------------------------
    def _build_constraints(
        self,
        robot_xy_yaw: Tuple[float, float, float],
        static_obstacles_xy: np.ndarray,    # (N_s, 2) in WORLD frame
        ped_xy_world:        np.ndarray,    # (N_p, 2) in WORLD frame
        ped_vel_world:       np.ndarray,    # (N_p, 2) in WORLD frame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the matrix A and vector b for the CBF constraints A u + b >= 0.

        Each row corresponds to one obstacle:
            row_lhs * u >= row_rhs
            i.e.  A_row * u + b_row >= 0  with A_row = row_lhs, b_row = -row_rhs

        For a unicycle and a moving obstacle k:
            h_k        = ||p - p_k||^2 - r_safe^2
            h_dot_k    = 2 (p - p_k)^T (p_dot - p_dot_k)
                       = 2 (p - p_k)^T (R(theta) [v, 0]^T - v_k)
            row_lhs[v]     = 2 [(x - x_k) cos(theta) + (y - y_k) sin(theta)]
            row_lhs[omega] = 0    (position-only CBF)
            row_rhs        = - gamma * h_k + 2 (p - p_k)^T v_k
        """
        x, y, theta = robot_xy_yaw
        cth, sth = float(np.cos(theta)), float(np.sin(theta))

        # All obstacles (positions + velocities).
        N_s = static_obstacles_xy.shape[0]
        N_p = ped_xy_world.shape[0]
        N = N_s + N_p
        if N == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)

        all_xy = np.zeros((N, 2), dtype=np.float64)
        all_v  = np.zeros((N, 2), dtype=np.float64)
        if N_s > 0:
            all_xy[:N_s] = static_obstacles_xy
        if N_p > 0:
            all_xy[N_s:] = ped_xy_world
            all_v[N_s:]  = ped_vel_world

        # Vectorized: dx = x - x_k
        dx = x - all_xy[:, 0]
        dy = y - all_xy[:, 1]

        h = dx * dx + dy * dy - self.r_safe_sq                # (N,)
        coef_v = 2.0 * (dx * cth + dy * sth)                  # (N,)
        moving_term = 2.0 * (dx * all_v[:, 0] + dy * all_v[:, 1])  # (N,)

        A = np.zeros((N, 2), dtype=np.float64)
        A[:, 0] = coef_v
        A[:, 1] = 0.0                                         # omega unconstrained
        b = self.gamma * h - moving_term                      # >= 0

        return A, b

    # ------------------------------------------------------------------
    def filter(
        self,
        u_sac: np.ndarray,                        # (2,)
        robot_xy_yaw: Tuple[float, float, float],
        lidar_scan: Optional[np.ndarray] = None,  # (n_beams,) in robot frame
        pedestrians_rel: Optional[np.ndarray] = None,  # (K, 5)
        ped_mask:        Optional[np.ndarray] = None,  # (K,)
    ) -> ShieldResult:
        """Project u_sac onto the safe set."""
        u_sac = np.asarray(u_sac, dtype=np.float64).reshape(2)

        # 1. Static obstacles from lidar (in robot frame -> world frame).
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

        # 3. Build constraints.
        A, b = self._build_constraints(
            robot_xy_yaw=robot_xy_yaw,
            static_obstacles_xy=obs_world,
            ped_xy_world=ped_world_xy,
            ped_vel_world=ped_world_vel,
        )
        n_active = int(A.shape[0])

        # 4. Solve QP. Decision variable u = (v, omega).
        #    Minimize 0.5 u^T P u + q^T u + const,
        #    where ||u - u_sac||^2 = u^T u - 2 u^T u_sac + const.
        #    -> P = 2 I, q = -2 u_sac
        # Constraints:
        #   - Box: u_min <= u <= u_max
        #   - Safety:  A u + b >= 0   <=>   A u >= -b   (no upper bound).
        # OSQP form: l <= C u <= u_box.
        v_min, v_max = self.v_bounds
        w_min, w_max = self.omega_bounds

        if n_active > 0:
            C = sp.vstack([sp.eye(2, format="csc"), sp.csc_matrix(A)], format="csc")
            l = np.concatenate([np.array([v_min, w_min]), -b])
            u = np.concatenate([np.array([v_max, w_max]),
                                np.full(n_active, np.inf)])
        else:
            C = sp.eye(2, format="csc")
            l = np.array([v_min, w_min])
            u = np.array([v_max, w_max])

        P = sp.csc_matrix(2.0 * np.eye(2))
        q = -2.0 * u_sac

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
            safe = np.asarray(sol.x, dtype=np.float64).reshape(2)
            # Clip to numerical bounds.
            safe[0] = np.clip(safe[0], v_min, v_max)
            safe[1] = np.clip(safe[1], w_min, w_max)
            infeasible = False
        else:
            # Emergency stop.
            if self.emergency_stop_on_infeasible:
                safe = np.array([0.0, 0.0], dtype=np.float64)
            else:
                safe = u_sac.copy()
            infeasible = True

        delta = safe - u_sac
        was_modified = bool(np.linalg.norm(delta) > 1e-4)

        return ShieldResult(
            safe_action=safe.astype(np.float32),
            was_modified=was_modified,
            n_active_constraints=n_active,
            infeasible=infeasible,
            delta_action=delta.astype(np.float32),
        )


# ======================================================================
# Factory
# ======================================================================
def build_shield_from_config(cfg) -> Optional[CbfSafetyShield]:
    """Build a shield from config; return None if cbf.enabled is False."""
    cbf_cfg = cfg.cbf
    if not cbf_cfg.get("enabled", True):
        return None
    return CbfSafetyShield(
        r_safe=cbf_cfg["r_safe"],
        gamma=cbf_cfg["gamma_cbf"],
        active_radius=cbf_cfg["active_radius"],
        v_min=cfg.robot.v_min, v_max=cfg.robot.v_max,
        omega_min=cfg.robot.omega_min, omega_max=cfg.robot.omega_max,
        emergency_stop_on_infeasible=cbf_cfg.get(
            "emergency_stop_on_infeasible", True,
        ),
    )
