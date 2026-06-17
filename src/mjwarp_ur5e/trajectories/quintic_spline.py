from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseTrajectory, BaseTrajectoryConfig, TrajectorySample


@dataclass(frozen=True, kw_only=True)
class QuinticSplineConfig(BaseTrajectoryConfig):
    num_joints: int
    num_segments: int
    q0: np.ndarray
    dq0: np.ndarray
    waypoints: np.ndarray
    dq_terminal: np.ndarray | None = None


@dataclass(frozen=True)
class _SegmentBoundaries:
    positions: np.ndarray  # (num_segments + 1, num_joints)
    velocities: np.ndarray  # (num_segments + 1, num_joints)
    accelerations: np.ndarray  # (num_segments + 1, num_joints)


class QuinticSplineTrajectory(BaseTrajectory):
    def __init__(self, config: QuinticSplineConfig) -> None:
        super().__init__(config)
        if config.num_joints <= 0:
            raise ValueError("num_joints must be positive")
        if config.num_segments <= 0:
            raise ValueError("num_segments must be positive")

        self.config = config
        self.q0 = self._validate_vector(config.q0, config.num_joints, "q0")
        self.dq0 = self._validate_vector(config.dq0, config.num_joints, "dq0")
        self.dq_terminal = (
            np.zeros(config.num_joints, dtype=np.float64)
            if config.dq_terminal is None
            else self._validate_vector(config.dq_terminal, config.num_joints, "dq_terminal")
        )

        waypoints = np.asarray(config.waypoints, dtype=np.float64)
        if waypoints.shape != (config.num_segments, config.num_joints):
            raise ValueError(
                f"waypoints must have shape ({config.num_segments}, {config.num_joints})"
            )
        self.waypoints = waypoints

        self.dt_seg = config.duration / config.num_segments
        self._boundaries = self._compute_boundaries()

    @staticmethod
    def _validate_vector(vec: np.ndarray, num_joints: int, name: str) -> np.ndarray:
        array = np.asarray(vec, dtype=np.float64)
        if array.shape != (num_joints,):
            raise ValueError(f"{name} must have shape ({num_joints},)")
        return array

    def _compute_boundaries(self) -> _SegmentBoundaries:
        n_seg = self.config.num_segments
        dt = self.dt_seg

        # wp[0] = q0, wp[k] = waypoints[k-1] for k = 1 ... n_seg
        wp = np.vstack([self.q0[None, :], self.waypoints])  # (n_seg + 1, num_joints)

        velocities = np.empty_like(wp)
        accelerations = np.empty_like(wp)

        velocities[0] = self.dq0
        velocities[-1] = self.dq_terminal

        # Interior nodes via central finite differences.
        for k in range(1, n_seg):
            velocities[k] = (wp[k + 1] - wp[k - 1]) / (2.0 * dt)
            accelerations[k] = (wp[k + 1] - 2.0 * wp[k] + wp[k - 1]) / dt**2

        # Start node: natural-spline acceleration consistent with dq0.
        accelerations[0] = 2.0 * (wp[1] - wp[0]) / dt**2 - 2.0 * self.dq0 / dt
        accelerations[-1] = np.zeros(self.config.num_joints, dtype=np.float64)

        return _SegmentBoundaries(positions=wp, velocities=velocities, accelerations=accelerations)

    def sample(self) -> TrajectorySample:
        t = self._time
        dt = self.dt_seg
        n_seg = self.config.num_segments

        # Locate the owning segment for each sample, clamped to the last segment.
        seg_idx = np.minimum(np.floor(t / dt).astype(int), n_seg - 1)
        tau = (t - seg_idx * dt) / dt

        b = self._boundaries
        p0 = b.positions[seg_idx]
        v0 = b.velocities[seg_idx]
        a0 = b.accelerations[seg_idx]
        p1 = b.positions[seg_idx + 1]
        v1 = b.velocities[seg_idx + 1]
        a1 = b.accelerations[seg_idx + 1]

        h = _hermite_basis(tau)
        hd = _hermite_basis_d1(tau)
        hdd = _hermite_basis_d2(tau)

        def combine(basis: tuple[np.ndarray, ...]) -> np.ndarray:
            h00, h10, h20, h01, h11, h21 = (col[:, None] for col in basis)
            return (
                h00 * p0
                + h10 * dt * v0
                + h20 * dt**2 * a0
                + h01 * p1
                + h11 * dt * v1
                + h21 * dt**2 * a1
            )

        position = combine(h)
        velocity = combine(hd) / dt
        acceleration = combine(hdd) / dt**2

        return TrajectorySample(
            time=self.time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )


def _hermite_basis(tau: np.ndarray) -> tuple[np.ndarray, ...]:
    t2 = tau**2
    t3 = tau**3
    t4 = tau**4
    t5 = tau**5
    h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    h10 = tau - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h20 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
    h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    h21 = 0.5 * t3 - t4 + 0.5 * t5
    return h00, h10, h20, h01, h11, h21


def _hermite_basis_d1(tau: np.ndarray) -> tuple[np.ndarray, ...]:
    t2 = tau**2
    t3 = tau**3
    t4 = tau**4
    h00 = -30.0 * t2 + 60.0 * t3 - 30.0 * t4
    h10 = 1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4
    h20 = tau - 4.5 * t2 + 6.0 * t3 - 2.5 * t4
    h01 = 30.0 * t2 - 60.0 * t3 + 30.0 * t4
    h11 = -12.0 * t2 + 28.0 * t3 - 15.0 * t4
    h21 = 1.5 * t2 - 4.0 * t3 + 2.5 * t4
    return h00, h10, h20, h01, h11, h21


def _hermite_basis_d2(tau: np.ndarray) -> tuple[np.ndarray, ...]:
    t2 = tau**2
    t3 = tau**3
    h00 = -60.0 * tau + 180.0 * t2 - 120.0 * t3
    h10 = -36.0 * tau + 96.0 * t2 - 60.0 * t3
    h20 = 1.0 - 9.0 * tau + 18.0 * t2 - 10.0 * t3
    h01 = 60.0 * tau - 180.0 * t2 + 120.0 * t3
    h11 = -24.0 * tau + 84.0 * t2 - 60.0 * t3
    h21 = 3.0 * tau - 12.0 * t2 + 10.0 * t3
    return h00, h10, h20, h01, h11, h21


def build_quintic_from_decision_vars(
    x: np.ndarray,
    q0: np.ndarray,
    dq0: np.ndarray,
    num_segments: int,
    num_joints: int,
    duration: float,
    fps: float,
    dq_terminal: np.ndarray | None = None,
) -> TrajectorySample:
    """最適化変数から TrajectorySample を直接生成する。

    x を (num_segments, num_joints) に reshape して waypoints として扱う。
    q0 は固定で決定変数に含まない。
    """
    x = np.asarray(x, dtype=np.float64)
    expected_len = num_segments * num_joints
    if x.size != expected_len:
        raise ValueError(f"x must have length {expected_len}, got {x.size}")
    waypoints = x.reshape(num_segments, num_joints)

    config = QuinticSplineConfig(
        duration=duration,
        fps=fps,
        num_joints=num_joints,
        num_segments=num_segments,
        q0=np.asarray(q0, dtype=np.float64),
        dq0=np.asarray(dq0, dtype=np.float64),
        waypoints=waypoints,
        dq_terminal=dq_terminal,
    )
    return QuinticSplineTrajectory(config).sample()
