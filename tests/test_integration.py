"""Integration tests exercising the full pipeline end-to-end."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mjwarp_ur5e.identification import (  # noqa: E402
    BatchLeastSquares,
    BatchLSConfig,
    ExcitationOptimizer,
    JointLimits,
    OptimizationResult,
    OptimizerConfig,
    PlaybackConfig,
    TrajectoryPlayback,
    _TrajectoryCache,
    body_inertial_parameters_from_model,
    compute_stacked_body_regressor,
    load_optimization_result,
    make_joint_acceleration_constraint,
    make_joint_position_constraint,
    make_joint_velocity_constraint,
    result_to_trajectory,
    save_optimization_result,
)
from mjwarp_ur5e.model import load_model, reset_to_home  # noqa: E402
from mjwarp_ur5e.trajectories import (  # noqa: E402
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)

NUM_JOINTS = 6
Q0 = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
BODY_NAME = "payload_box_mount"


def _load_scene():
    loaded = load_model("assets/ur5e/mjcf/scene_with_box.xml")
    reset_to_home(loaded.model, loaded.data)
    return loaded


def test_optimization_and_validation_roundtrip() -> None:
    """Run a minimal optimization, save/load result, validate it."""
    loaded = _load_scene()
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=2,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0,
        subsample_factor=10,
        n_monte_carlo=1,
        max_iter_per_start=3,
        seed=42,
        body_name=BODY_NAME,
    )
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)
    result = opt.optimize()

    assert isinstance(result, OptimizationResult)
    assert np.isfinite(result.condition_number)
    assert result.condition_number > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.json"
        save_optimization_result(result, path)
        loaded_result = load_optimization_result(path)

    np.testing.assert_allclose(loaded_result.x_opt, result.x_opt)
    assert loaded_result.condition_number == pytest.approx(result.condition_number)
    np.testing.assert_allclose(loaded_result.a_opt, result.a_opt)
    np.testing.assert_allclose(loaded_result.b_opt, result.b_opt)
    np.testing.assert_allclose(loaded_result.q0, result.q0)

    sample = result_to_trajectory(loaded_result)
    expected_steps = int(cfg.duration * cfg.fps) + 1
    assert sample.position.shape == (expected_steps, NUM_JOINTS)
    assert sample.velocity.shape == (expected_steps, NUM_JOINTS)
    assert sample.acceleration.shape == (expected_steps, NUM_JOINTS)


def test_playback_and_estimation_pipeline() -> None:
    """Playback a trajectory and run estimation."""
    loaded = _load_scene()
    duration = 1.0
    fps = 50.0

    rng = np.random.default_rng(42)
    coefficients = {
        "a": rng.uniform(-0.05, 0.05, size=(6, 2)).tolist(),
        "b": rng.uniform(-0.05, 0.05, size=(6, 2)).tolist(),
    }
    traj_config = WindowedFourierTrajectoryConfig(
        duration=duration,
        fps=fps,
        num_joints=NUM_JOINTS,
        num_harmonics=2,
        base_freq=0.2,
        coefficients=coefficients,
        q0=Q0,
    )
    traj = WindowedFourierTrajectory(traj_config).sample()

    playback_cfg = PlaybackConfig(
        body_name=BODY_NAME,
        site_name="attachment_site",
    )
    playback = TrajectoryPlayback(loaded.model, loaded.data, playback_cfg)
    buffer = playback.execute(traj)

    expected_steps = int(duration * fps) + 1
    assert len(buffer) == expected_steps

    arrays = buffer.to_arrays()
    assert arrays["q"].shape == (expected_steps, NUM_JOINTS)
    assert arrays["wrench"].shape == (expected_steps, 6)

    regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        arrays["q"],
        arrays["dq"],
        arrays["ddq"],
        BODY_NAME,
        subsample_factor=5,
    )
    wrench_stacked = arrays["wrench"][::5].reshape(-1)

    estimator = BatchLeastSquares(BatchLSConfig())
    result = estimator.estimate(regressor, wrench_stacked)

    true_params = body_inertial_parameters_from_model(loaded.model, BODY_NAME)
    assert result.mass == pytest.approx(true_params.mass, rel=0.1)


def test_constraints_accept_small_reject_large() -> None:
    """Small-amplitude trajectory passes all constraints,
    large fails."""
    joint_limits = JointLimits()

    num_harmonics = 2
    cache_small = _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=num_harmonics,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0,
    )

    rng = np.random.default_rng(42)
    n = NUM_JOINTS * num_harmonics
    x_small = rng.uniform(-0.01, 0.01, size=2 * n)

    pos_fn = make_joint_position_constraint(cache_small, joint_limits)
    vel_fn = make_joint_velocity_constraint(cache_small, joint_limits)
    acc_fn = make_joint_acceleration_constraint(cache_small, joint_limits)

    assert pos_fn(x_small) >= 0
    assert vel_fn(x_small) >= 0
    assert acc_fn(x_small) >= 0

    cache_large = _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=num_harmonics,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0,
    )
    x_large = rng.uniform(-100.0, 100.0, size=2 * n)

    pos_fn_l = make_joint_position_constraint(cache_large, joint_limits)
    vel_fn_l = make_joint_velocity_constraint(cache_large, joint_limits)
    acc_fn_l = make_joint_acceleration_constraint(cache_large, joint_limits)

    margins = [
        pos_fn_l(x_large),
        vel_fn_l(x_large),
        acc_fn_l(x_large),
    ]
    assert any(m < 0 for m in margins), (
        "At least one constraint should be violated for large amplitude"
    )


def test_json_io_preserves_all_fields() -> None:
    """Save and load optimization result, verify all fields match."""
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=3,
        base_freq=0.15,
        duration=5.0,
        fps=100.0,
        q0=Q0,
    )
    rng = np.random.default_rng(99)
    n = NUM_JOINTS * 3
    x_opt = rng.uniform(-0.1, 0.1, size=2 * n)
    a_opt = x_opt[:n].reshape(NUM_JOINTS, 3)
    b_opt = x_opt[n:].reshape(NUM_JOINTS, 3)

    original = OptimizationResult(
        x_opt=x_opt,
        condition_number=123.456,
        a_opt=a_opt,
        b_opt=b_opt,
        q0=Q0.copy(),
        config=cfg,
        n_evaluations=500,
        wall_time=12.34,
        n_restarts=20,
        best_start_index=7,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_result.json"
        save_optimization_result(original, path)
        loaded = load_optimization_result(path)

    np.testing.assert_allclose(loaded.x_opt, original.x_opt)
    np.testing.assert_allclose(loaded.a_opt, original.a_opt)
    np.testing.assert_allclose(loaded.b_opt, original.b_opt)
    np.testing.assert_allclose(loaded.q0, original.q0)
    assert loaded.condition_number == pytest.approx(original.condition_number)
    assert loaded.n_evaluations == original.n_evaluations
    assert loaded.wall_time == pytest.approx(original.wall_time)
    assert loaded.n_restarts == original.n_restarts
    assert loaded.best_start_index == original.best_start_index
    assert loaded.config.num_joints == original.config.num_joints
    assert loaded.config.num_harmonics == original.config.num_harmonics
    assert loaded.config.base_freq == pytest.approx(original.config.base_freq)
    assert loaded.config.duration == pytest.approx(original.config.duration)
    assert loaded.config.fps == pytest.approx(original.config.fps)
    assert loaded.config.body_name == original.config.body_name
    assert loaded.config.site_name == original.config.site_name
