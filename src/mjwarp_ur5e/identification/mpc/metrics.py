from __future__ import annotations

import mujoco
import numpy as np

from mjwarp_ur5e.model import get_named_object_id
from mjwarp_ur5e.trajectories.base import TrajectorySample

_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _gravity_directions_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str,
    subsample: int,
) -> np.ndarray:
    """Unit gravity vectors expressed in the body frame, one per sampled frame."""
    if subsample <= 0:
        raise ValueError("subsample must be positive")

    body_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id is None:
        raise ValueError(f"Unknown body name: {body_name}")

    directions = []
    for i in range(0, len(q_trajectory), subsample):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        g_body = rotation.T @ _GRAVITY_WORLD
        directions.append(g_body / np.linalg.norm(g_body))
    return np.asarray(directions, dtype=np.float64)


def _sweep_from_directions(directions: np.ndarray) -> float:
    if len(directions) < 2:
        return 0.0
    cos_angles = np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(cos_angles)))


def _spread_from_directions(directions: np.ndarray) -> float:
    if len(directions) < 2:
        return 0.0
    mean_dir = directions.mean(axis=0)
    mean_norm = np.linalg.norm(mean_dir)
    if mean_norm == 0.0:
        return float(np.pi)
    mean_dir /= mean_norm
    cos_angles = np.clip(directions @ mean_dir, -1.0, 1.0)
    return float(np.max(np.arccos(cos_angles)))


def gravity_sweep_angle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str = "payload_box_mount",
    subsample: int = 1,
) -> float:
    """Cumulative angular sweep of the gravity direction in the body frame [rad].

    Sums the angle between gravity directions at consecutive sampled frames.
    Larger values indicate better excitation of the gravity-dependent
    parameters (mass, first moment).
    """
    directions = _gravity_directions_body(model, data, q_trajectory, body_name, subsample)
    return _sweep_from_directions(directions)


def gravity_direction_spread(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str = "payload_box_mount",
    subsample: int = 1,
) -> float:
    """Maximum opening angle from the mean gravity direction in the body frame [rad]."""
    directions = _gravity_directions_body(model, data, q_trajectory, body_name, subsample)
    return _spread_from_directions(directions)


def acceleration_peak(trajectory: TrajectorySample) -> float:
    """Maximum absolute joint acceleration across all joints and timesteps [rad/s^2]."""
    return float(np.max(np.abs(trajectory.acceleration)))


def trajectory_excitation_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: TrajectorySample,
    body_name: str = "payload_box_mount",
) -> dict[str, float]:
    """Aggregate excitation-quality metrics for a trajectory."""
    directions = _gravity_directions_body(model, data, trajectory.position, body_name, 1)
    return {
        "gravity_sweep_angle": _sweep_from_directions(directions),
        "gravity_direction_spread": _spread_from_directions(directions),
        "acceleration_peak": acceleration_peak(trajectory),
        "velocity_peak": float(np.max(np.abs(trajectory.velocity))),
        "position_range": float(np.max(trajectory.position) - np.min(trajectory.position)),
    }
