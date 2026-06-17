from __future__ import annotations

import numpy as np
import pytest

from mjwarp_ur5e.identification.mpc.metrics import (
    acceleration_peak,
    gravity_direction_spread,
    gravity_sweep_angle,
    trajectory_excitation_summary,
)
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.trajectories.base import TrajectorySample

BODY_NAME = "payload_box_mount"


@pytest.fixture(scope="module")
def loaded():
    return load_and_reset("assets/ur5e/mjcf/scene_with_box.xml")


def _static_trajectory(model, num: int = 5) -> np.ndarray:
    q = np.linspace(-0.5, 0.5, model.nq)
    return np.tile(q, (num, 1))


def test_static_trajectory_has_zero_sweep_and_spread(loaded) -> None:
    q_traj = _static_trajectory(loaded.model)

    assert gravity_sweep_angle(loaded.model, loaded.data, q_traj, BODY_NAME) == 0.0
    assert gravity_direction_spread(loaded.model, loaded.data, q_traj, BODY_NAME) == 0.0


def test_nonzero_trajectory_has_positive_sweep(loaded) -> None:
    rng = np.random.default_rng(0)
    q_traj = rng.uniform(-1.0, 1.0, size=(5, loaded.model.nq))

    sweep = gravity_sweep_angle(loaded.model, loaded.data, q_traj, BODY_NAME)
    spread = gravity_direction_spread(loaded.model, loaded.data, q_traj, BODY_NAME)

    assert sweep > 0.0
    assert spread > 0.0


def test_subsample_coarsens_sweep(loaded) -> None:
    rng = np.random.default_rng(1)
    q_traj = rng.uniform(-1.0, 1.0, size=(10, loaded.model.nq))

    full = gravity_sweep_angle(loaded.model, loaded.data, q_traj, BODY_NAME, subsample=1)
    coarse = gravity_sweep_angle(loaded.model, loaded.data, q_traj, BODY_NAME, subsample=2)

    assert coarse <= full


def test_acceleration_peak() -> None:
    n = 4
    accel = np.zeros((n, 6))
    accel[2, 3] = -7.5
    accel[1, 0] = 3.0
    trajectory = TrajectorySample(
        time=np.arange(n, dtype=float),
        position=np.zeros((n, 6)),
        velocity=np.zeros((n, 6)),
        acceleration=accel,
    )

    assert acceleration_peak(trajectory) == 7.5


def test_summary_has_expected_keys(loaded) -> None:
    n = 5
    rng = np.random.default_rng(2)
    position = rng.uniform(-1.0, 1.0, size=(n, loaded.model.nq))
    trajectory = TrajectorySample(
        time=np.linspace(0.0, 1.0, n),
        position=position,
        velocity=rng.uniform(-2.0, 2.0, size=(n, loaded.model.nv)),
        acceleration=rng.uniform(-3.0, 3.0, size=(n, loaded.model.nv)),
    )

    summary = trajectory_excitation_summary(loaded.model, loaded.data, trajectory, BODY_NAME)

    assert set(summary) == {
        "gravity_sweep_angle",
        "gravity_direction_spread",
        "acceleration_peak",
        "velocity_peak",
        "position_range",
    }
    assert all(isinstance(v, float) for v in summary.values())
