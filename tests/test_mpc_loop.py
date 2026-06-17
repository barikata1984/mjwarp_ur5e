"""Tests for the MPC excitation loop."""

from __future__ import annotations

import numpy as np

from mjwarp_ur5e.identification.mpc.config import MPCConfig, PlannerConfig
from mjwarp_ur5e.identification.mpc.loop import MPCLoop, _slice_trajectory
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.trajectories.base import TrajectorySample


def _short_config() -> MPCConfig:
    return MPCConfig(
        max_mpc_steps=2,
        planner=PlannerConfig(n_restarts=2, max_iter_per_start=20),
    )


def test_mpc_loop_runs():
    loaded = load_and_reset("assets/ur5e/mjcf/scene_with_box.xml")
    mpc = MPCLoop(_short_config(), loaded.model, loaded.data)
    result = mpc.run()

    assert result is not None
    assert len(result.steps) <= 2
    assert result.final_estimation.phi.shape == (10,)
    assert np.isfinite(result.final_condition_number)
    assert result.total_samples > 0


def test_state_continuity():
    loaded = load_and_reset("assets/ur5e/mjcf/scene_with_box.xml")
    mpc = MPCLoop(_short_config(), loaded.model, loaded.data)
    result = mpc.run()

    if len(result.steps) >= 2:
        np.testing.assert_allclose(result.steps[1].q_start, result.steps[0].q_end, atol=1e-9)


def test_slice_trajectory():
    fps = 100.0
    duration = 3.0
    n = int(round(duration * fps)) + 1
    time = np.linspace(0.0, duration, n)
    position = np.zeros((n, 6))
    traj = TrajectorySample(
        time=time,
        position=position,
        velocity=position.copy(),
        acceleration=position.copy(),
    )

    sliced = _slice_trajectory(traj, 0.0, 1.5)
    assert sliced.time[0] == 0.0
    assert abs(sliced.time[-1] - 1.5) < 1e-6
    assert sliced.position.shape[0] == sliced.time.shape[0]
