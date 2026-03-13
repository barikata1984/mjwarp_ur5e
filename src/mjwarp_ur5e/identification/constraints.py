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


def compute_fourier_velocity_bounds(
    num_joints: int,
    num_harmonics: int,
    base_freq: float,
    duration: float,
    dq_max: np.ndarray,
) -> np.ndarray:
    """Compute per-coefficient box bounds from velocity limits via triangle inequality.

    For the windowed Fourier trajectory v_j(t) = w'(t)*osc_j(t) + w(t)*osc_j'(t),
    the triangle inequality gives the sufficient condition:

        sum_k (|a_{j,k}| + |b_{j,k}|) * alpha_k  <=  dq_max_j

    where alpha_k = max|w'(t)| + 2*pi*base_freq*k  (velocity gain per harmonic).

    Allocating budget uniformly across harmonics yields per-coefficient box bounds:

        |a_{j,k}|, |b_{j,k}|  <=  dq_max_j / (2 * N_h * alpha_k)

    Returns the upper bound array with same layout as the flat decision vector x.
    """
    # max |w'(s)| for w(s) = 64 s^3 (1-s)^3, computed on fine grid
    s = np.linspace(0, 1, 10_000)
    dw_ds = 192.0 * s**2 - 768.0 * s**3 + 960.0 * s**4 - 384.0 * s**5
    dw_dt_max = float(np.max(np.abs(dw_ds))) / duration

    harmonics = np.arange(1, num_harmonics + 1, dtype=np.float64)
    omega = 2.0 * np.pi * base_freq * harmonics
    alpha = dw_dt_max + omega  # per-harmonic velocity gain (w_max = 1)

    n = num_joints * num_harmonics
    upper = np.empty(2 * n, dtype=np.float64)

    # x layout (C-order reshape): a[j, k] = x[j * num_harmonics + k]
    for j in range(num_joints):
        for k in range(num_harmonics):
            bound = float(dq_max[j]) / (2.0 * num_harmonics * alpha[k])
            idx = j * num_harmonics + k
            upper[idx] = bound
            upper[n + idx] = bound

    return upper


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
    enable_velocity_constraint: bool = True,
    enable_acceleration_constraint: bool = True,
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
        {
            "type": "ineq",
            "name": "joint_position",
            "fun": make_joint_position_constraint(cache, joint_limits),
        },
    ]
    if enable_velocity_constraint:
        constraints.append(
            {
                "type": "ineq",
                "name": "joint_velocity",
                "fun": make_joint_velocity_constraint(cache, joint_limits),
            }
        )
    if enable_acceleration_constraint:
        constraints.append(
            {
                "type": "ineq",
                "name": "joint_acceleration",
                "fun": make_joint_acceleration_constraint(cache, joint_limits),
            }
        )

    if workspace_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "name": "workspace",
                "fun": make_workspace_constraint(cache, workspace_config, model, data),
            }
        )

    if payload_workspace_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "name": "payload_workspace",
                "fun": make_payload_workspace_constraint(
                    cache, payload_workspace_config, model, data, payload_body_name
                ),
            }
        )

    if collision_config is not None and model is not None and data is not None:
        checker = CollisionChecker(model, data, collision_config)
        constraints.append(
            {
                "type": "ineq",
                "name": "collision",
                "fun": make_collision_constraint(cache, checker),
            }
        )

    if ee_velocity_config is not None and model is not None and data is not None:
        constraints.append(
            {
                "type": "ineq",
                "name": "ee_velocity",
                "fun": make_ee_velocity_constraint(
                    cache, ee_velocity_config, model, data, site_name
                ),
            }
        )

    return constraints
