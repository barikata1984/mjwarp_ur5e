from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import minimize

from mjwarp_ur5e.identification.collision import CollisionConfig
from scipy.optimize import Bounds

from mjwarp_ur5e.identification.constraints import (
    JointLimits,
    _TrajectoryCache,
    build_scipy_constraints,
    compute_fourier_velocity_bounds,
)
from mjwarp_ur5e.identification.objective import (
    condition_number_objective,
    d_optimal_with_cond,
    evaluate_full_resolution,
)
from mjwarp_ur5e.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig

_UR5E_HOME = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

log = logging.getLogger(__name__)


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
    objective_type: str = "d_optimal"  # "d_optimal" or "condition_number"
    optimizer_method: str = "SLSQP"
    ftol: float = 1e-6
    seed: int = 42
    joint_limits: JointLimits | None = None
    workspace_config: WorkspaceConstraintConfig | None = None
    payload_workspace_config: WorkspaceConstraintConfig | None = None
    collision_config: CollisionConfig | None = None
    ee_velocity_config: EeVelocityConfig | None = None
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"
    enable_velocity_constraint: bool = True
    enable_acceleration_constraint: bool = True
    use_fourier_bounds: bool = False
    include_ft_offset: bool = False
    ft_offset_column_scale: bool = True

    def __post_init__(self) -> None:
        if self.q0 is None:
            self.q0 = _UR5E_HOME.copy()
        if self.joint_limits is None:
            self.joint_limits = JointLimits()


@dataclass
class EarlyStopConfig:
    """Early stopping configuration for multi-start optimization."""

    enabled: bool = False
    patience: int = 5
    min_improvement: float = 1e-3
    target_cond: float = 0.0  # Stop when condition number <= this (0 = disabled)


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "ur5e-excitation"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


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
    # Diagnostics (populated after optimization)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    trajectory_stats: dict[str, float] = field(default_factory=dict)


def _config_to_wandb_dict(cfg: OptimizerConfig) -> dict:
    """Convert OptimizerConfig to a flat dict suitable for wandb.config."""
    d: dict = {
        "num_joints": cfg.num_joints,
        "num_harmonics": cfg.num_harmonics,
        "base_freq": cfg.base_freq,
        "duration": cfg.duration,
        "fps": cfg.fps,
        "subsample_factor": cfg.subsample_factor,
        "n_monte_carlo": cfg.n_monte_carlo,
        "max_iter_per_start": cfg.max_iter_per_start,
        "objective_type": cfg.objective_type,
        "optimizer_method": cfg.optimizer_method,
        "ftol": cfg.ftol,
        "seed": cfg.seed,
        "body_name": cfg.body_name,
        "site_name": cfg.site_name,
        "n_decision_vars": 2 * cfg.num_joints * cfg.num_harmonics,
    }
    if cfg.q0 is not None:
        d["q0"] = cfg.q0.tolist()
    if cfg.joint_limits is not None:
        d["q_min"] = cfg.joint_limits.q_min.tolist()
        d["q_max"] = cfg.joint_limits.q_max.tolist()
        d["dq_max"] = cfg.joint_limits.dq_max.tolist()
        d["ddq_max"] = cfg.joint_limits.ddq_max.tolist()
    if cfg.workspace_config is not None:
        d["max_displacement"] = cfg.workspace_config.max_displacement
    d["collision_enabled"] = cfg.collision_config is not None
    d["payload_workspace_enabled"] = cfg.payload_workspace_config is not None
    d["use_fourier_bounds"] = cfg.use_fourier_bounds
    d["enable_velocity_constraint"] = cfg.enable_velocity_constraint
    d["enable_acceleration_constraint"] = cfg.enable_acceleration_constraint
    d["include_ft_offset"] = cfg.include_ft_offset
    d["ft_offset_column_scale"] = cfg.ft_offset_column_scale
    if cfg.ee_velocity_config is not None:
        d["ee_max_linear_velocity"] = cfg.ee_velocity_config.max_linear_velocity
    return d


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

    def _generate_random_x0(
        self, rng: np.random.Generator, bounds: Bounds | None = None
    ) -> np.ndarray:
        nj = self.config.num_joints
        nh = self.config.num_harmonics
        x = np.zeros(2 * nj * nh, dtype=np.float64)
        for k in range(nh):
            scale = 0.3 / (k + 1)
            x[k * nj : (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
            x[nj * nh + k * nj : nj * nh + (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
        if bounds is not None:
            x = np.clip(x, bounds.lb, bounds.ub)
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
            ee_velocity_config=cfg.ee_velocity_config,
            site_name=cfg.site_name,
            enable_velocity_constraint=cfg.enable_velocity_constraint,
            enable_acceleration_constraint=cfg.enable_acceleration_constraint,
        )
        return cache, constraints

    def _compute_fourier_bounds(self) -> Bounds | None:
        """Compute scipy Bounds from analytical Fourier velocity bounds."""
        cfg = self.config
        if not cfg.use_fourier_bounds or cfg.joint_limits is None:
            return None
        upper = compute_fourier_velocity_bounds(
            num_joints=cfg.num_joints,
            num_harmonics=cfg.num_harmonics,
            base_freq=cfg.base_freq,
            duration=cfg.duration,
            dq_max=cfg.joint_limits.dq_max,
        )
        return Bounds(lb=-upper, ub=upper)

    def _compute_constraint_margin(self, x: np.ndarray, constraints: list[dict]) -> float:
        """Return the minimum constraint margin (>= 0 means all satisfied)."""
        return min(float(c["fun"](x)) for c in constraints)

    @staticmethod
    def _compute_named_margins(x: np.ndarray, constraints: list[dict]) -> dict[str, float]:
        """Compute per-constraint margin using the 'name' key."""
        return {
            c.get("name", f"constraint_{i}"): float(c["fun"](x)) for i, c in enumerate(constraints)
        }

    def _compute_trajectory_stats(self, x: np.ndarray, cache: _TrajectoryCache) -> dict[str, float]:
        """Compute trajectory-level statistics for diagnostics."""
        sample = cache.get(x)
        return {
            "q_max": float(np.max(np.abs(sample.position))),
            "dq_max": float(np.max(np.abs(sample.velocity))),
            "ddq_max": float(np.max(np.abs(sample.acceleration))),
            "dq_per_joint_max": np.max(np.abs(sample.velocity), axis=0).tolist(),
            "ddq_per_joint_max": np.max(np.abs(sample.acceleration), axis=0).tolist(),
        }

    def optimize(
        self,
        wandb_config: WandbConfig | None = None,
        early_stop_config: EarlyStopConfig | None = None,
    ) -> OptimizationResult:
        """Run multi-start optimisation and return the best result.

        Args:
            wandb_config: If provided with enabled=True, log metrics to W&B.
            early_stop_config: If provided with enabled=True, stop early when
                the global best condition number stops improving.
        """
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)
        cache, constraints = self._build_cache_and_constraints()
        fourier_bounds = self._compute_fourier_bounds()

        # --- wandb setup ---
        wb_run = None
        if wandb_config and wandb_config.enabled:
            try:
                import wandb

                wb_run = wandb.init(
                    project=wandb_config.project,
                    name=wandb_config.run_name,
                    tags=wandb_config.tags or None,
                    config=_config_to_wandb_dict(cfg),
                )
            except Exception:
                log.warning("wandb init failed; continuing without logging", exc_info=True)

        # --- early stopping setup ---
        es = early_stop_config or EarlyStopConfig()
        patience_counter = 0

        # Track the latest objective value and condition number.
        # _latest_obj: raw objective value (D-optimal or cond number)
        # _latest_cond: condition number (always tracked for reporting)
        _latest_obj: list[float] = [float("inf")]
        _latest_cond: list[float] = [float("inf")]

        use_d_optimal = cfg.objective_type == "d_optimal"

        _column_scale = cfg.ft_offset_column_scale and cfg.include_ft_offset

        def objective(x: np.ndarray) -> float:
            if use_d_optimal:
                obj_val, cond_val = d_optimal_with_cond(
                    x,
                    cache,
                    self.model,
                    self.data,
                    cfg.body_name,
                    cfg.subsample_factor,
                    include_ft_offset=cfg.include_ft_offset,
                    column_scale=_column_scale,
                )
            else:
                cond_val = condition_number_objective(
                    x,
                    cache,
                    self.model,
                    self.data,
                    cfg.body_name,
                    cfg.subsample_factor,
                    include_ft_offset=cfg.include_ft_offset,
                    column_scale=_column_scale,
                )
                obj_val = cond_val
            _latest_obj[0] = obj_val
            _latest_cond[0] = cond_val
            return obj_val

        rng = np.random.default_rng(cfg.seed)
        best_x: np.ndarray | None = None
        best_cond = float("inf")
        best_idx = 0
        total_evals = 0
        global_step = 0

        t0 = time.perf_counter()

        actual_restarts = 0
        for i in range(cfg.n_monte_carlo):
            x0 = self._generate_random_x0(rng, bounds=fourier_bounds)
            iter_in_restart = [0]
            restart_t0 = time.perf_counter()

            def _callback(xk: np.ndarray) -> None:
                nonlocal global_step
                global_step += 1
                iter_in_restart[0] += 1
                cond_val = _latest_cond[0]
                obj_val = _latest_obj[0]
                if wb_run is not None:
                    log_dict: dict = {
                        "iter/condition_number": cond_val,
                        "iter/objective": obj_val,
                        "iter/restart_index": i,
                        "iter/iter_in_restart": iter_in_restart[0],
                        "iter/wall_time": time.perf_counter() - t0,
                    }
                    wb_run.log(log_dict, step=global_step)

            result = minimize(
                objective,
                x0,
                method=cfg.optimizer_method,
                bounds=fourier_bounds,
                constraints=constraints,
                options={"maxiter": cfg.max_iter_per_start, "ftol": cfg.ftol},
                callback=_callback,
            )
            total_evals += result.nfev
            actual_restarts += 1
            restart_wall = time.perf_counter() - restart_t0
            margin = self._compute_constraint_margin(result.x, constraints)
            feasible = margin >= 0

            # Evaluate condition number for reporting (reuse cached trajectory)
            if use_d_optimal:
                _, cond = d_optimal_with_cond(
                    result.x,
                    cache,
                    self.model,
                    self.data,
                    cfg.body_name,
                    cfg.subsample_factor,
                    include_ft_offset=cfg.include_ft_offset,
                    column_scale=_column_scale,
                )
            else:
                cond = float(result.fun)
            obj_val = float(result.fun)

            if use_d_optimal:
                print(
                    f"  start {i + 1}/{cfg.n_monte_carlo}: "
                    f"cond={cond:.4f}  D-opt={obj_val:.4f}  margin={margin:.4f}  "
                    f"feasible={feasible}  ({restart_wall:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  start {i + 1}/{cfg.n_monte_carlo}: "
                    f"cond={cond:.4f}  margin={margin:.4f}  "
                    f"feasible={feasible}  ({restart_wall:.1f}s)",
                    flush=True,
                )

            improved = cond < best_cond
            if improved:
                best_cond = cond
                best_x = result.x.copy()
                best_idx = i

            # Log restart-level metrics
            if wb_run is not None:
                global_step += 1
                restart_log: dict = {
                    "restart/condition_number": cond,
                    "restart/objective": obj_val,
                    "restart/global_best_cond": best_cond,
                    "restart/constraint_margin_min": margin,
                    "restart/feasible": int(feasible),
                    "restart/n_func_evals": result.nfev,
                    "restart/n_iters": iter_in_restart[0],
                    "restart/wall_time_s": restart_wall,
                    "restart/improved": int(improved),
                    "restart/index": i,
                }
                restart_margins = self._compute_named_margins(result.x, constraints)
                for name, mval in restart_margins.items():
                    restart_log[f"restart/margin/{name}"] = mval
                wb_run.log(restart_log, step=global_step)

            # Early stopping check
            if es.enabled:
                # Target condition number reached (current restart must be feasible)
                if es.target_cond > 0 and feasible and cond <= es.target_cond:
                    print(
                        f"  Early stop: target cond {es.target_cond} reached "
                        f"(cond={cond:.4f}, feasible=True)",
                        flush=True,
                    )
                    break
                # Patience-based stopping
                if improved and (best_cond < float("inf")):
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= es.patience:
                    print(
                        f"  Early stop: no improvement for {es.patience} restarts "
                        f"(best={best_cond:.4f})",
                        flush=True,
                    )
                    break

        wall_time = time.perf_counter() - t0

        n = cfg.num_joints * cfg.num_harmonics
        a_opt = best_x[:n].reshape(cfg.num_joints, cfg.num_harmonics)
        b_opt = best_x[n:].reshape(cfg.num_joints, cfg.num_harmonics)

        # Compute diagnostics on best solution
        named_margins = self._compute_named_margins(best_x, constraints)
        best_feasible = all(v >= 0 for v in named_margins.values())
        traj_stats = self._compute_trajectory_stats(best_x, cache)

        opt_result = OptimizationResult(
            x_opt=best_x,
            condition_number=best_cond,
            a_opt=a_opt,
            b_opt=b_opt,
            q0=q0,
            config=cfg,
            n_evaluations=total_evals,
            wall_time=wall_time,
            n_restarts=actual_restarts,
            best_start_index=best_idx,
            constraint_margins=named_margins,
            feasible=best_feasible,
            trajectory_stats=traj_stats,
        )

        # Log final summary to wandb
        if wb_run is not None:
            summary: dict = {
                "final/condition_number": best_cond,
                "final/best_restart_index": best_idx,
                "final/total_restarts": actual_restarts,
                "final/total_func_evals": total_evals,
                "final/wall_time_s": wall_time,
                "final/feasible": int(best_feasible),
            }
            for name, margin in named_margins.items():
                summary[f"final/margin/{name}"] = margin
            for key, val in traj_stats.items():
                if isinstance(val, list):
                    for j, v in enumerate(val):
                        summary[f"final/traj/{key}_j{j}"] = v
                else:
                    summary[f"final/traj/{key}"] = val
            wb_run.summary.update(summary)
            wb_run.finish()

        return opt_result

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
            include_ft_offset=cfg.include_ft_offset,
            column_scale=cfg.ft_offset_column_scale and cfg.include_ft_offset,
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
