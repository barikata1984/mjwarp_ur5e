from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mjwarp_ur5e.trajectories.base import TrajectorySample
from mjwarp_ur5e.trajectories.windowed_fourier import (
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)

if TYPE_CHECKING:
    from mjwarp_ur5e.identification.collision import CollisionConfig
    from mjwarp_ur5e.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig


@dataclass(frozen=True)
class JointLimits:
    """Joint position, velocity and acceleration limits for UR5e."""

    q_min: np.ndarray = field(default_factory=lambda: np.full(6, -2.0 * np.pi))
    q_max: np.ndarray = field(default_factory=lambda: np.full(6, 2.0 * np.pi))
    dq_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.pi, np.pi, np.pi, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi]
        )
    )
    ddq_max: np.ndarray = field(default_factory=lambda: np.full(6, 8.0))


class _TrajectoryCache:
    """Caches trajectory samples to avoid recomputation for the same x."""

    def __init__(
        self,
        num_joints: int,
        num_harmonics: int,
        base_freq: float,
        duration: float,
        fps: float,
        q0: np.ndarray,
    ) -> None:
        self.num_joints = num_joints
        self.num_harmonics = num_harmonics
        self.base_freq = base_freq
        self.duration = duration
        self.fps = fps
        self.q0 = np.asarray(q0, dtype=np.float64).copy()
        self._cache_key: bytes | None = None
        self._cache_value: TrajectorySample | None = None

    def get(self, x: np.ndarray) -> TrajectorySample:
        key = x.tobytes()
        if key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        sample = build_trajectory_from_params(
            x,
            self.num_joints,
            self.num_harmonics,
            self.base_freq,
            self.duration,
            self.fps,
            self.q0,
        )
        self._cache_key = key
        self._cache_value = sample
        return sample


def build_trajectory_from_params(
    x: np.ndarray,
    num_joints: int,
    num_harmonics: int,
    base_freq: float,
    duration: float,
    fps: float,
    q0: np.ndarray,
) -> TrajectorySample:
    """Reconstruct trajectory from flat optimisation variable x.

    x layout: first num_joints*num_harmonics values are sine coefficients (a),
    next num_joints*num_harmonics values are cosine coefficients (b).
    """
    n = num_joints * num_harmonics
    a = np.asarray(x[:n], dtype=np.float64).reshape(num_joints, num_harmonics)
    b = np.asarray(x[n : 2 * n], dtype=np.float64).reshape(num_joints, num_harmonics)
    config = WindowedFourierTrajectoryConfig(
        duration=duration,
        fps=fps,
        num_joints=num_joints,
        num_harmonics=num_harmonics,
        base_freq=base_freq,
        coefficients={"a": a.tolist(), "b": b.tolist()},
        q0=q0.tolist(),
    )
    return WindowedFourierTrajectory(config).sample()


def make_joint_position_constraint(
    cache: _TrajectoryCache,
    joint_limits: JointLimits,
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff all joint positions within limits."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        margin_lo = sample.position - joint_limits.q_min
        margin_hi = joint_limits.q_max - sample.position
        return float(np.min([margin_lo, margin_hi]))

    return constraint


def make_joint_velocity_constraint(
    cache: _TrajectoryCache,
    joint_limits: JointLimits,
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff all joint velocities within limits."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        margin = joint_limits.dq_max - np.abs(sample.velocity)
        return float(np.min(margin))

    return constraint


def make_joint_acceleration_constraint(
    cache: _TrajectoryCache,
    joint_limits: JointLimits,
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff all joint accelerations within limits."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        margin = joint_limits.ddq_max - np.abs(sample.acceleration)
        return float(np.min(margin))

    return constraint


def build_scipy_constraints(
    cache: _TrajectoryCache,
    joint_limits: JointLimits,
    workspace_config: WorkspaceConstraintConfig | None = None,
    collision_config: CollisionConfig | None = None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    payload_workspace_config: WorkspaceConstraintConfig | None = None,
    payload_body_name: str = "payload_box_mount",
    ee_velocity_config: EeVelocityConfig | None = None,
    site_name: str = "attachment_site",
) -> list[dict]:
    """Assemble all constraints in scipy.optimize format."""
    from mjwarp_ur5e.identification.collision import (
        CollisionChecker,
        make_collision_constraint,
    )
    from mjwarp_ur5e.identification.workspace import (
        make_ee_velocity_constraint,
        make_payload_workspace_constraint,
        make_workspace_constraint,
    )

    constraints: list[dict] = [
        {"type": "ineq", "fun": make_joint_position_constraint(cache, joint_limits)},
        {"type": "ineq", "fun": make_joint_velocity_constraint(cache, joint_limits)},
        {"type": "ineq", "fun": make_joint_acceleration_constraint(cache, joint_limits)},
    ]

    if workspace_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": make_workspace_constraint(cache, workspace_config, model, data),
            }
        )

    if payload_workspace_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": make_payload_workspace_constraint(
                    cache, payload_workspace_config, model, data, payload_body_name
                ),
            }
        )

    if collision_config is not None and model is not None and data is not None:
        checker = CollisionChecker(model, data, collision_config)
        constraints.append({"type": "ineq", "fun": make_collision_constraint(cache, checker)})

    if ee_velocity_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": make_ee_velocity_constraint(
                    cache, ee_velocity_config, model, data, site_name
                ),
            }
        )

    return constraints
