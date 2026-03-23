from __future__ import annotations

import mujoco
import numpy as np

from .constraints import _TrajectoryCache, build_trajectory_from_params
from .regressor import compute_condition_number, compute_stacked_body_regressor


def _compute_stacked_regressor(
    x: np.ndarray,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    subsample_factor: int,
    with_ft_offset: bool = False,
) -> np.ndarray:
    """Build stacked regressor from coefficient vector (shared by objectives)."""
    sample = cache.get(x)
    return compute_stacked_body_regressor(
        model,
        data,
        sample.position,
        sample.velocity,
        sample.acceleration,
        body_name,
        subsample_factor=subsample_factor,
        with_ft_offset=with_ft_offset,
    )


def _maybe_column_scale(matrix: np.ndarray, column_scale: bool) -> np.ndarray:
    """Apply column L2-norm scaling if requested."""
    if not column_scale:
        return matrix
    norms = np.linalg.norm(matrix, axis=0)
    norms = np.maximum(norms, 1e-30)
    return matrix / norms


def condition_number_objective(
    x: np.ndarray,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    subsample_factor: int,
    with_ft_offset: bool = False,
    column_scale: bool = False,
) -> float:
    """Compute condition number of the stacked body regressor.

    Returns a large finite value (1e12) on numerical failure to avoid
    crashing the optimizer.
    """
    try:
        stacked = _compute_stacked_regressor(
            x, cache, model, data, body_name, subsample_factor, with_ft_offset
        )
        return compute_condition_number(stacked, column_scale=column_scale)
    except (np.linalg.LinAlgError, ValueError):
        return 1e12


def d_optimal_objective(
    x: np.ndarray,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    subsample_factor: int,
    with_ft_offset: bool = False,
    column_scale: bool = False,
) -> float:
    """D-optimal objective: -log det(W^T W) = -2 * sum(log(sigma_i)).

    Smooth (C^inf) alternative to condition number minimization.
    Maximizes the volume of the information ellipsoid, encouraging all
    singular values to be large rather than just minimizing their ratio.

    Returns a large finite value (1e12) on numerical failure.
    """
    try:
        stacked = _compute_stacked_regressor(
            x, cache, model, data, body_name, subsample_factor, with_ft_offset
        )
        stacked = _maybe_column_scale(stacked, column_scale)
        sv = np.linalg.svd(stacked, compute_uv=False)
        sv_floored = np.maximum(sv, 1e-30)
        return -2.0 * np.sum(np.log(sv_floored))
    except (np.linalg.LinAlgError, ValueError):
        return 1e12


def d_optimal_with_cond(
    x: np.ndarray,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    subsample_factor: int,
    with_ft_offset: bool = False,
    column_scale: bool = False,
) -> tuple[float, float]:
    """Compute D-optimal objective and condition number from a single SVD.

    Returns (d_optimal_value, condition_number). Avoids the cost of building
    the stacked regressor twice when both values are needed.
    """
    try:
        stacked = _compute_stacked_regressor(
            x, cache, model, data, body_name, subsample_factor, with_ft_offset
        )
        stacked = _maybe_column_scale(stacked, column_scale)
        sv = np.linalg.svd(stacked, compute_uv=False)
        sv_floored = np.maximum(sv, 1e-30)
        d_opt = -2.0 * np.sum(np.log(sv_floored))
        if sv[-1] < 1e-12:
            cond = float("inf")
        else:
            cond = float(sv[0] / sv[-1])
        return d_opt, cond
    except (np.linalg.LinAlgError, ValueError):
        return 1e12, 1e12


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
    with_ft_offset: bool = False,
    column_scale: bool = False,
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
        with_ft_offset=with_ft_offset,
    )
    cond = compute_condition_number(stacked, column_scale=column_scale)
    return cond, stacked
