from __future__ import annotations

import mujoco
import numpy as np

from .constraints import _TrajectoryCache, build_trajectory_from_params
from .regressor import compute_condition_number, compute_stacked_body_regressor


def condition_number_objective(
    x: np.ndarray,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    subsample_factor: int,
) -> float:
    """Compute condition number of the stacked body regressor.

    Returns a large finite value (1e12) on numerical failure to avoid
    crashing the optimizer.
    """
    try:
        sample = cache.get(x)
        stacked = compute_stacked_body_regressor(
            model,
            data,
            sample.position,
            sample.velocity,
            sample.acceleration,
            body_name,
            subsample_factor=subsample_factor,
        )
        return compute_condition_number(stacked)
    except (np.linalg.LinAlgError, ValueError):
        return 1e12


def evaluate_full_resolution(
    x: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    num_joints: int,
    num_harmonics: int,
    base_freq: float,
    duration: float,
    fps: float,
    q0: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate objective at full resolution (no subsampling).

    Returns (condition_number, stacked_regressor).
    """
    sample = build_trajectory_from_params(
        x, num_joints, num_harmonics, base_freq, duration, fps, q0
    )
    stacked = compute_stacked_body_regressor(
        model,
        data,
        sample.position,
        sample.velocity,
        sample.acceleration,
        body_name,
        subsample_factor=1,
    )
    cond = compute_condition_number(stacked)
    return cond, stacked
