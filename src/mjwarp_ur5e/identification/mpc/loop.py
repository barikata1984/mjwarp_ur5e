"""MPC excitation loop: plan -> execute -> identify -> replan."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from mjwarp_ur5e.identification.estimators.rtls import RecursiveTotalLeastSquares, RTLSConfig
from mjwarp_ur5e.identification.estimators.types import EstimationResult
from mjwarp_ur5e.identification.execution import PlaybackConfig, TrajectoryPlayback
from mjwarp_ur5e.identification.regressor import (
    compute_condition_number,
    compute_stacked_body_regressor,
)
from mjwarp_ur5e.trajectories.base import TrajectorySample

from .config import MPCConfig
from .planner import ExcitationPlanner, PlanResult


@dataclass
class MPCStepLog:
    """Record of a single MPC iteration."""

    step: int
    plan_result: PlanResult
    estimation: EstimationResult
    condition_number_accumulated: float
    q_start: np.ndarray
    q_end: np.ndarray
    n_samples_total: int
    wall_time_plan: float
    wall_time_execute: float
    executed_q: np.ndarray | None = None  # (N, 6) joint positions actually executed


@dataclass
class MPCResult:
    """Result of the full MPC loop."""

    steps: list[MPCStepLog]
    final_estimation: EstimationResult
    total_wall_time: float
    total_samples: int
    final_condition_number: float
    converged: bool

    def executed_trajectory_q(self) -> np.ndarray:
        """Concatenate executed joint positions from all steps into (N_total, nj)."""
        arrays = [s.executed_q for s in self.steps if s.executed_q is not None]
        if not arrays:
            raise ValueError("No executed trajectory data in steps")
        return np.vstack(arrays)


def _slice_trajectory(
    trajectory: TrajectorySample,
    t_start: float,
    t_end: float,
) -> TrajectorySample:
    """Cut the time range [t_start, t_end] out of a TrajectorySample, re-zeroing time."""
    mask = (trajectory.time >= t_start - 1e-9) & (trajectory.time <= t_end + 1e-9)
    return TrajectorySample(
        time=trajectory.time[mask] - t_start,
        position=trajectory.position[mask],
        velocity=trajectory.velocity[mask],
        acceleration=trajectory.acceleration[mask],
    )


class MPCLoop:
    """Closed-loop MPC for inertial-parameter excitation."""

    def __init__(
        self,
        config: MPCConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self._config = config
        self._model = model
        self._data = data
        self._planner = ExcitationPlanner(config, model, data)
        self._rtls = RecursiveTotalLeastSquares(RTLSConfig(n_params=10))
        self._playback = TrajectoryPlayback(
            model,
            data,
            PlaybackConfig(
                use_pd_control=config.use_pd_control,
                body_name=config.body_name,
                site_name=config.site_name,
                noise_std_wrench=config.noise_std_wrench,
                settle_time=0.0,
            ),
        )

    def run(self) -> MPCResult:
        cfg = self._config
        q_current = cfg.q0.copy()
        dq_current = np.zeros(cfg.num_joints)
        W_accumulated: np.ndarray | None = None
        steps: list[MPCStepLog] = []
        prev_cond: float | None = None
        cond_accumulated = float("nan")
        converged = False
        rng = np.random.default_rng(cfg.planner.seed + 1000)

        t0 = time.perf_counter()

        for step_idx in range(cfg.max_mpc_steps):
            print(
                f"[MPC step {step_idx}/{cfg.max_mpc_steps}] planning...",
                flush=True,
            )
            t_plan = time.perf_counter()
            plan_result = self._planner.plan(q_current, dq_current, W_accumulated)
            wall_plan = time.perf_counter() - t_plan

            execute_traj = _slice_trajectory(plan_result.trajectory, 0.0, cfg.replan_period)

            t_exec = time.perf_counter()
            buffer = self._playback.execute(
                execute_traj,
                rng=rng if cfg.noise_std_wrench > 0 else None,
            )
            wall_exec = time.perf_counter() - t_exec

            arrays = buffer.to_arrays()
            q_meas = arrays["q"]
            dq_meas = arrays["dq"]
            ddq_meas = arrays["ddq"]
            wrench = arrays["wrench"]

            W_new = compute_stacked_body_regressor(
                self._model,
                self._data,
                q_meas,
                dq_meas,
                ddq_meas,
                cfg.body_name,
            )
            y_new = wrench.ravel()

            if W_accumulated is None:
                W_accumulated = W_new
            else:
                W_accumulated = np.vstack([W_accumulated, W_new])

            cond_accumulated = compute_condition_number(W_accumulated)

            if not self._rtls._initialized:
                self._rtls.initialize(W_new, y_new)
            else:
                self._rtls.update(W_new, y_new)
            estimation = self._rtls.get_current_estimate()

            q_start = q_meas[0].copy()
            q_current = q_meas[-1].copy()
            dq_current = dq_meas[-1].copy()

            print(
                f"  -> cond={cond_accumulated:.2f}  "
                f"mass={estimation.mass:.4f}  "
                f"samples={W_accumulated.shape[0] // 6}  "
                f"plan={wall_plan:.1f}s  exec={wall_exec:.2f}s",
                flush=True,
            )

            steps.append(
                MPCStepLog(
                    step=step_idx,
                    plan_result=plan_result,
                    estimation=estimation,
                    condition_number_accumulated=cond_accumulated,
                    q_start=q_start,
                    q_end=q_current.copy(),
                    n_samples_total=W_accumulated.shape[0] // 6,
                    wall_time_plan=wall_plan,
                    wall_time_execute=wall_exec,
                    executed_q=q_meas.copy(),
                )
            )

            if prev_cond is not None and prev_cond > 0:
                rel_improvement = (prev_cond - cond_accumulated) / prev_cond
                if rel_improvement < cfg.convergence_threshold:
                    converged = True
                    break
            prev_cond = cond_accumulated

        total_time = time.perf_counter() - t0
        total_samples = W_accumulated.shape[0] // 6 if W_accumulated is not None else 0

        return MPCResult(
            steps=steps,
            final_estimation=self._rtls.get_current_estimate(),
            total_wall_time=total_time,
            total_samples=total_samples,
            final_condition_number=cond_accumulated,
            converged=converged,
        )
