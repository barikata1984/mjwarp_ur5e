from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize

from mjwarp_ur5e.identification.collision import CollisionConfig
from mjwarp_ur5e.identification.constraints import (
    JointLimits,
    _TrajectoryCache,
    build_scipy_constraints,
)
from mjwarp_ur5e.identification.objective import (
    condition_number_objective,
    evaluate_full_resolution,
)
from mjwarp_ur5e.identification.workspace import WorkspaceConstraintConfig

_UR5E_HOME = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


@dataclass
class OptimizerConfig:
    num_joints: int = 6
    num_harmonics: int = 5
    base_freq: float = 0.1
    duration: float = 10.0
    fps: float = 100.0
    q0: np.ndarray | None = None
    subsample_factor: int = 10
    n_monte_carlo: int = 20
    max_iter_per_start: int = 200
    optimizer_method: str = "SLSQP"
    ftol: float = 1e-6
    seed: int = 42
    joint_limits: JointLimits | None = None
    workspace_config: WorkspaceConstraintConfig | None = None
    payload_workspace_config: WorkspaceConstraintConfig | None = None
    collision_config: CollisionConfig | None = None
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"

    def __post_init__(self) -> None:
        if self.q0 is None:
            self.q0 = _UR5E_HOME.copy()
        if self.joint_limits is None:
            self.joint_limits = JointLimits()


@dataclass
class OptimizationResult:
    x_opt: np.ndarray
    condition_number: float
    a_opt: np.ndarray
    b_opt: np.ndarray
    q0: np.ndarray
    config: OptimizerConfig
    n_evaluations: int
    wall_time: float
    n_restarts: int
    best_start_index: int


class ExcitationOptimizer:
    """Multi-start SLSQP optimizer for excitation trajectory design."""

    def __init__(
        self,
        config: OptimizerConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.config = config
        self.model = model
        self.data = data

    def _get_x_size(self) -> int:
        return 2 * self.config.num_joints * self.config.num_harmonics

    def _generate_random_x0(self, rng: np.random.Generator) -> np.ndarray:
        nj = self.config.num_joints
        nh = self.config.num_harmonics
        x = np.zeros(2 * nj * nh, dtype=np.float64)
        for k in range(nh):
            scale = 0.3 / (k + 1)
            x[k * nj : (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
            x[nj * nh + k * nj : nj * nh + (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
        return x

    def _build_cache_and_constraints(self) -> tuple[_TrajectoryCache, list[dict]]:
        """Build trajectory cache and all scipy constraints from config."""
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)
        cache = _TrajectoryCache(
            num_joints=cfg.num_joints,
            num_harmonics=cfg.num_harmonics,
            base_freq=cfg.base_freq,
            duration=cfg.duration,
            fps=cfg.fps,
            q0=q0,
        )
        constraints = build_scipy_constraints(
            cache,
            cfg.joint_limits,
            workspace_config=cfg.workspace_config,
            collision_config=cfg.collision_config,
            model=self.model,
            data=self.data,
            payload_workspace_config=cfg.payload_workspace_config,
            payload_body_name=cfg.body_name,
        )
        return cache, constraints

    def optimize(self) -> OptimizationResult:
        """Run multi-start optimisation and return the best result."""
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)
        cache, constraints = self._build_cache_and_constraints()

        def objective(x: np.ndarray) -> float:
            return condition_number_objective(
                x, cache, self.model, self.data, cfg.body_name, cfg.subsample_factor
            )

        rng = np.random.default_rng(cfg.seed)
        best_x: np.ndarray | None = None
        best_cond = float("inf")
        best_idx = 0
        total_evals = 0

        t0 = time.perf_counter()

        for i in range(cfg.n_monte_carlo):
            x0 = self._generate_random_x0(rng)
            result = minimize(
                objective,
                x0,
                method=cfg.optimizer_method,
                constraints=constraints,
                options={"maxiter": cfg.max_iter_per_start, "ftol": cfg.ftol},
            )
            total_evals += result.nfev
            cond = float(result.fun)
            print(f"  start {i + 1}/{cfg.n_monte_carlo}: cond={cond:.4f}")
            if cond < best_cond:
                best_cond = cond
                best_x = result.x.copy()
                best_idx = i

        wall_time = time.perf_counter() - t0

        n = cfg.num_joints * cfg.num_harmonics
        a_opt = best_x[:n].reshape(cfg.num_joints, cfg.num_harmonics)
        b_opt = best_x[n:].reshape(cfg.num_joints, cfg.num_harmonics)

        return OptimizationResult(
            x_opt=best_x,
            condition_number=best_cond,
            a_opt=a_opt,
            b_opt=b_opt,
            q0=q0,
            config=cfg,
            n_evaluations=total_evals,
            wall_time=wall_time,
            n_restarts=cfg.n_monte_carlo,
            best_start_index=best_idx,
        )

    def validate_trajectory(self, result: OptimizationResult) -> dict:
        """Full-resolution validation of an optimisation result."""
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)

        cond, _ = evaluate_full_resolution(
            result.x_opt,
            self.model,
            self.data,
            cfg.body_name,
            cfg.num_joints,
            cfg.num_harmonics,
            cfg.base_freq,
            cfg.duration,
            cfg.fps,
            q0,
        )

        _, full_constraints = self._build_cache_and_constraints()

        margins: list[float] = []
        all_satisfied = True
        for c in full_constraints:
            val = c["fun"](result.x_opt)
            margins.append(val)
            if val < 0:
                all_satisfied = False

        return {
            "condition_number": cond,
            "all_constraints_satisfied": all_satisfied,
            "constraint_margins": margins,
        }
