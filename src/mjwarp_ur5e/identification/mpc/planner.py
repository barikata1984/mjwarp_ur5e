from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import minimize

from mjwarp_ur5e.identification.constraints import (
    make_joint_acceleration_constraint,
    make_joint_position_constraint,
    make_joint_velocity_constraint,
)
from mjwarp_ur5e.identification.regressor import (
    compute_condition_number,
    compute_stacked_body_regressor,
)
from mjwarp_ur5e.trajectories.base import TrajectorySample
from mjwarp_ur5e.trajectories.quintic_spline import build_quintic_from_decision_vars

from .config import MPCConfig
from .metrics import acceleration_peak, gravity_sweep_angle

# SLSQP leaves O(1e-9..1e-6) overshoot when a constraint is active at the optimum.
_FEASIBILITY_TOL = 1e-6


class _QuinticCache:
    """Caches the quintic TrajectorySample for the most recent decision vector."""

    def __init__(
        self,
        q0: np.ndarray,
        dq0: np.ndarray,
        num_segments: int,
        num_joints: int,
        duration: float,
        fps: float,
    ) -> None:
        self._q0 = np.asarray(q0, dtype=np.float64).copy()
        self._dq0 = np.asarray(dq0, dtype=np.float64).copy()
        self._num_segments = num_segments
        self._num_joints = num_joints
        self._duration = duration
        self._fps = fps
        self._cache_key: bytes | None = None
        self._cache_value: TrajectorySample | None = None

    def get(self, x: np.ndarray) -> TrajectorySample:
        key = x.tobytes()
        if key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        sample = build_quintic_from_decision_vars(
            x,
            self._q0,
            self._dq0,
            self._num_segments,
            self._num_joints,
            self._duration,
            self._fps,
        )
        self._cache_key = key
        self._cache_value = sample
        return sample


@dataclass
class PlanResult:
    """Result of a single horizon planning step."""

    trajectory: TrajectorySample
    waypoints: np.ndarray  # (num_segments, num_joints) optimized waypoints (excl. q0)
    condition_number: float  # kappa_2(W) for this horizon (or accumulated)
    gravity_sweep: float  # gravity sweep angle [rad]
    acceleration_peak: float  # max |ddq| [rad/s^2]
    cost: float  # optimizer objective value
    feasible: bool  # all constraints satisfied
    constraint_margins: dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0  # seconds
    n_evaluations: int = 0  # total function evaluations across restarts


def _make_objective(
    cache: _QuinticCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: MPCConfig,
    W_accumulated: np.ndarray | None,
) -> Callable[[np.ndarray], float]:
    body_name = config.body_name
    subsample = config.horizon.subsample_factor

    def objective(x: np.ndarray) -> float:
        try:
            sample = cache.get(x)
            W_new = compute_stacked_body_regressor(
                model,
                data,
                sample.position,
                sample.velocity,
                sample.acceleration,
                body_name,
                subsample_factor=subsample,
            )
            W = np.vstack([W_accumulated, W_new]) if W_accumulated is not None else W_new
            return compute_condition_number(W)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12

    return objective


class ExcitationPlanner:
    """Multi-start SLSQP planner for a single excitation horizon."""

    def __init__(
        self,
        config: MPCConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.config = config
        self._model = model
        self._data = data

    def trajectory_from_x(
        self,
        x: np.ndarray,
        q0: np.ndarray,
        dq0: np.ndarray,
    ) -> TrajectorySample:
        """Decision variable -> TrajectorySample (for external use)."""
        hcfg = self.config.horizon
        return build_quintic_from_decision_vars(
            x,
            q0,
            dq0,
            hcfg.num_segments,
            self.config.num_joints,
            hcfg.duration,
            hcfg.fps,
        )

    def plan(
        self,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        W_accumulated: np.ndarray | None = None,
    ) -> PlanResult:
        cfg = self.config
        pcfg = cfg.planner
        hcfg = cfg.horizon
        limits = cfg.joint_limits
        num_joints = cfg.num_joints

        q_current = np.asarray(q_current, dtype=np.float64)
        dq_current = np.asarray(dq_current, dtype=np.float64)

        cache = _QuinticCache(
            q_current, dq_current, hcfg.num_segments, num_joints, hcfg.duration, hcfg.fps
        )
        objective = _make_objective(cache, self._model, self._data, cfg, W_accumulated)
        constraints = [
            {
                "type": "ineq",
                "name": "joint_position",
                "fun": make_joint_position_constraint(cache, limits),
            },
            {
                "type": "ineq",
                "name": "joint_velocity",
                "fun": make_joint_velocity_constraint(cache, limits),
            },
            {
                "type": "ineq",
                "name": "joint_acceleration",
                "fun": make_joint_acceleration_constraint(cache, limits),
            },
        ]

        rng = np.random.default_rng(pcfg.seed)
        best_x: np.ndarray | None = None
        best_cost = float("inf")
        total_evals = 0

        t0 = time.perf_counter()
        for i in range(pcfg.n_restarts):
            x0 = q_current + rng.uniform(
                -pcfg.waypoint_perturbation,
                pcfg.waypoint_perturbation,
                size=(hcfg.num_segments, num_joints),
            )
            x0 = np.clip(x0, limits.q_min, limits.q_max).ravel()

            result = minimize(
                objective,
                x0,
                method=pcfg.method,
                constraints=constraints,
                options={"maxiter": pcfg.max_iter_per_start, "ftol": pcfg.ftol},
            )
            total_evals += result.nfev

            improved = result.fun < best_cost
            if improved:
                best_cost = result.fun
                best_x = result.x.copy()

            dt = time.perf_counter() - t0
            tag = "*" if improved else " "
            print(
                f"    {tag} restart {i + 1}/{pcfg.n_restarts}: "
                f"cond={result.fun:.2f}  best={best_cost:.2f}  "
                f"({dt:.1f}s)",
                flush=True,
            )

        wall_time = time.perf_counter() - t0

        if best_x is None:
            best_x = np.tile(q_current, hcfg.num_segments)
            best_cost = objective(best_x)

        sample = cache.get(best_x)
        margins = {c["name"]: float(c["fun"](best_x)) for c in constraints}
        feasible = all(v >= -_FEASIBILITY_TOL for v in margins.values())

        return PlanResult(
            trajectory=sample,
            waypoints=best_x.reshape(hcfg.num_segments, num_joints),
            condition_number=best_cost,
            gravity_sweep=gravity_sweep_angle(
                self._model,
                self._data,
                sample.position,
                cfg.body_name,
                subsample=hcfg.subsample_factor,
            ),
            acceleration_peak=acceleration_peak(sample),
            cost=best_cost,
            feasible=feasible,
            constraint_margins=margins,
            wall_time=wall_time,
            n_evaluations=total_evals,
        )
