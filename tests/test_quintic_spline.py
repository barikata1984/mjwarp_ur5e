from __future__ import annotations

import numpy as np
import pytest

from mjwarp_ur5e.trajectories.base import TrajectorySample
from mjwarp_ur5e.trajectories.quintic_spline import (
    QuinticSplineConfig,
    QuinticSplineTrajectory,
    build_quintic_from_decision_vars,
)

NUM_JOINTS = 6
NUM_SEGMENTS = 4
DURATION = 2.0
FPS = 200.0


def _make_trajectory(
    q0: np.ndarray,
    dq0: np.ndarray,
    waypoints: np.ndarray,
    dq_terminal: np.ndarray | None = None,
) -> QuinticSplineTrajectory:
    config = QuinticSplineConfig(
        duration=DURATION,
        fps=FPS,
        num_joints=NUM_JOINTS,
        num_segments=NUM_SEGMENTS,
        q0=q0,
        dq0=dq0,
        waypoints=waypoints,
        dq_terminal=dq_terminal,
    )
    return QuinticSplineTrajectory(config)


def test_static_trajectory_is_zero():
    q0 = np.full(NUM_JOINTS, 0.3)
    dq0 = np.zeros(NUM_JOINTS)
    waypoints = np.tile(q0, (NUM_SEGMENTS, 1))

    sample = _make_trajectory(q0, dq0, waypoints).sample()

    expected = np.broadcast_to(q0, sample.position.shape)
    np.testing.assert_allclose(sample.position, expected, atol=1e-12)
    np.testing.assert_allclose(sample.velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(sample.acceleration, 0.0, atol=1e-12)


def test_initial_boundary_conditions():
    rng = np.random.default_rng(0)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))

    sample = _make_trajectory(q0, dq0, waypoints).sample()

    np.testing.assert_allclose(sample.position[0], q0, atol=1e-10)
    np.testing.assert_allclose(sample.velocity[0], dq0, atol=1e-10)


def test_terminal_velocity():
    rng = np.random.default_rng(1)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))
    dq_terminal = rng.standard_normal(NUM_JOINTS)

    sample = _make_trajectory(q0, dq0, waypoints, dq_terminal).sample()

    np.testing.assert_allclose(sample.velocity[-1], dq_terminal, atol=1e-10)


def test_terminal_velocity_defaults_to_zero():
    rng = np.random.default_rng(2)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))

    sample = _make_trajectory(q0, dq0, waypoints).sample()

    np.testing.assert_allclose(sample.velocity[-1], 0.0, atol=1e-10)


def test_c2_continuity_at_segment_boundaries():
    rng = np.random.default_rng(3)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))
    traj = _make_trajectory(q0, dq0, waypoints)

    dt = DURATION / NUM_SEGMENTS
    for k in range(1, NUM_SEGMENTS):
        t_boundary = k * dt
        # End of segment k-1 (tau=1) must match start of segment k (tau=0).
        left = _eval_at(traj, t_boundary, segment=k - 1)
        right = _eval_at(traj, t_boundary, segment=k)
        np.testing.assert_allclose(left[0], right[0], atol=1e-10)
        np.testing.assert_allclose(left[1], right[1], atol=1e-10)
        np.testing.assert_allclose(left[2], right[2], atol=1e-10)


def _eval_at(
    traj: QuinticSplineTrajectory, t: float, segment: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one instant on a specified segment via the Hermite boundary values."""
    from mjwarp_ur5e.trajectories import quintic_spline as qs

    dt = traj.dt_seg
    tau = np.array([(t - segment * dt) / dt], dtype=np.float64)
    b = traj._boundaries
    p0, v0, a0 = b.positions[segment], b.velocities[segment], b.accelerations[segment]
    p1, v1, a1 = b.positions[segment + 1], b.velocities[segment + 1], b.accelerations[segment + 1]

    def combine(basis: tuple[np.ndarray, ...], scale: float) -> np.ndarray:
        h00, h10, h20, h01, h11, h21 = (c[:, None] for c in basis)
        out = (
            h00 * p0
            + h10 * dt * v0
            + h20 * dt**2 * a0
            + h01 * p1
            + h11 * dt * v1
            + h21 * dt**2 * a1
        )
        return (out / scale)[0]

    position = combine(qs._hermite_basis(tau), 1.0)
    velocity = combine(qs._hermite_basis_d1(tau), dt)
    acceleration = combine(qs._hermite_basis_d2(tau), dt**2)
    return position, velocity, acceleration


def test_sample_shapes():
    rng = np.random.default_rng(4)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))

    sample = _make_trajectory(q0, dq0, waypoints).sample()

    num_steps = sample.time.shape[0]
    assert sample.position.shape == (num_steps, NUM_JOINTS)
    assert sample.velocity.shape == (num_steps, NUM_JOINTS)
    assert sample.acceleration.shape == (num_steps, NUM_JOINTS)


def test_build_from_decision_vars_roundtrip():
    rng = np.random.default_rng(5)
    q0 = rng.standard_normal(NUM_JOINTS)
    dq0 = rng.standard_normal(NUM_JOINTS)
    waypoints = rng.standard_normal((NUM_SEGMENTS, NUM_JOINTS))
    x = waypoints.reshape(-1)

    sample = build_quintic_from_decision_vars(
        x=x,
        q0=q0,
        dq0=dq0,
        num_segments=NUM_SEGMENTS,
        num_joints=NUM_JOINTS,
        duration=DURATION,
        fps=FPS,
    )

    assert isinstance(sample, TrajectorySample)
    direct = _make_trajectory(q0, dq0, waypoints).sample()
    np.testing.assert_allclose(sample.position, direct.position)


def test_build_from_decision_vars_wrong_length():
    with pytest.raises(ValueError):
        build_quintic_from_decision_vars(
            x=np.zeros(NUM_SEGMENTS * NUM_JOINTS + 1),
            q0=np.zeros(NUM_JOINTS),
            dq0=np.zeros(NUM_JOINTS),
            num_segments=NUM_SEGMENTS,
            num_joints=NUM_JOINTS,
            duration=DURATION,
            fps=FPS,
        )
