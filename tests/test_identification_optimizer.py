import tempfile
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mjwarp_ur5e.identification.constraints import (  # noqa: E402
    _TrajectoryCache,
)
from mjwarp_ur5e.identification.io import (  # noqa: E402
    load_optimization_result,
    result_to_trajectory,
    save_optimization_result,
)
from mjwarp_ur5e.identification.objective import (  # noqa: E402
    condition_number_objective,
    d_optimal_objective,
)
from mjwarp_ur5e.identification.optimizer import (  # noqa: E402
    ExcitationOptimizer,
    OptimizationResult,
    OptimizerConfig,
)
from mjwarp_ur5e.model import load_model, reset_to_home  # noqa: E402

NUM_JOINTS = 6
NUM_HARMONICS = 2
BASE_FREQ = 0.2
DURATION = 4.0
FPS = 50.0
Q0 = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _load_scene():
    loaded = load_model("assets/ur5e/mjcf/scene_with_box.xml")
    reset_to_home(loaded.model, loaded.data)
    return loaded


def _make_cache() -> _TrajectoryCache:
    return _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
    )


def _make_x(scale: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = NUM_JOINTS * NUM_HARMONICS
    return rng.uniform(-scale, scale, size=2 * n)


# --- Objective tests ---


def test_objective_returns_finite_for_random_coefficients() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    x = _make_x(scale=0.05)

    cond = condition_number_objective(
        x,
        cache,
        loaded.model,
        loaded.data,
        body_name="payload_box_mount",
        subsample_factor=5,
    )
    assert np.isfinite(cond)
    assert cond > 0


def test_objective_returns_large_value_for_zero_coefficients() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    x = np.zeros(2 * NUM_JOINTS * NUM_HARMONICS)

    cond = condition_number_objective(
        x,
        cache,
        loaded.model,
        loaded.data,
        body_name="payload_box_mount",
        subsample_factor=5,
    )
    # Zero coefficients produce a degenerate (static) trajectory:
    # condition number should be inf or very large
    assert cond == float("inf") or cond > 1e6


# --- D-optimal objective tests ---


def test_d_optimal_returns_finite_for_random_coefficients() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    x = _make_x(scale=0.05)

    val = d_optimal_objective(
        x,
        cache,
        loaded.model,
        loaded.data,
        body_name="payload_box_mount",
        subsample_factor=5,
    )
    assert np.isfinite(val)


def test_d_optimal_returns_large_value_for_zero_coefficients() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    x = np.zeros(2 * NUM_JOINTS * NUM_HARMONICS)

    val = d_optimal_objective(
        x,
        cache,
        loaded.model,
        loaded.data,
        body_name="payload_box_mount",
        subsample_factor=5,
    )
    # Degenerate trajectory → near-zero singular values → large positive D-optimal value
    assert val > 100


def test_d_optimal_decreases_with_better_trajectory() -> None:
    """A trajectory with better excitation should have a lower (more negative) D-optimal value."""
    loaded = _load_scene()
    cache = _make_cache()

    x_small = _make_x(scale=0.01, seed=0)
    x_larger = _make_x(scale=0.05, seed=0)

    val_small = d_optimal_objective(
        x_small, cache, loaded.model, loaded.data, "payload_box_mount", 5
    )
    val_larger = d_optimal_objective(
        x_larger, cache, loaded.model, loaded.data, "payload_box_mount", 5
    )
    # Larger amplitude → better excitation → larger singular values → more negative D-optimal
    assert val_larger < val_small


# --- OptimizerConfig tests ---


def test_optimizer_config_defaults() -> None:
    cfg = OptimizerConfig()
    assert cfg.num_joints == 6
    assert cfg.num_harmonics == 5
    assert cfg.base_freq == 0.1
    assert cfg.duration == 10.0
    assert cfg.fps == 100.0
    assert cfg.q0 is not None
    assert cfg.q0.shape == (6,)
    assert cfg.subsample_factor == 10
    assert cfg.n_monte_carlo == 20
    assert cfg.max_iter_per_start == 200
    assert cfg.objective_type == "d_optimal"
    assert cfg.optimizer_method == "SLSQP"
    assert cfg.ftol == 1e-6
    assert cfg.seed == 42
    assert cfg.joint_limits is not None
    assert cfg.body_name == "payload_box_mount"
    assert cfg.site_name == "attachment_site"


# --- _generate_random_x0 tests ---


def test_generate_random_x0_shape_and_scaling() -> None:
    loaded = _load_scene()
    cfg = OptimizerConfig(num_joints=6, num_harmonics=3)
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)

    rng = np.random.default_rng(0)
    x0 = opt._generate_random_x0(rng)

    assert x0.shape == (opt._get_x_size(),)
    assert x0.shape == (2 * 6 * 3,)

    # Higher harmonics should have smaller amplitude on average
    nj = cfg.num_joints
    nh = cfg.num_harmonics
    for k in range(nh):
        expected_scale = 0.3 / (k + 1)
        a_block = x0[k * nj : (k + 1) * nj]
        b_block = x0[nj * nh + k * nj : nj * nh + (k + 1) * nj]
        assert np.all(np.abs(a_block) <= expected_scale + 1e-15)
        assert np.all(np.abs(b_block) <= expected_scale + 1e-15)


# --- Smoke test for optimize() ---


def test_optimize_smoke() -> None:
    loaded = _load_scene()
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
        subsample_factor=10,
        n_monte_carlo=2,
        max_iter_per_start=5,
        seed=123,
    )
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)
    result = opt.optimize()

    assert isinstance(result, OptimizationResult)
    assert result.x_opt.shape == (2 * NUM_JOINTS * NUM_HARMONICS,)
    assert np.isfinite(result.condition_number)
    assert result.condition_number > 0
    assert result.a_opt.shape == (NUM_JOINTS, NUM_HARMONICS)
    assert result.b_opt.shape == (NUM_JOINTS, NUM_HARMONICS)
    assert result.q0.shape == (NUM_JOINTS,)
    assert result.n_evaluations > 0
    assert result.wall_time > 0
    assert result.n_restarts == 2
    assert 0 <= result.best_start_index < 2


def test_optimize_smoke_d_optimal() -> None:
    loaded = _load_scene()
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
        subsample_factor=10,
        n_monte_carlo=2,
        max_iter_per_start=5,
        objective_type="d_optimal",
        seed=123,
    )
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)
    result = opt.optimize()

    assert isinstance(result, OptimizationResult)
    assert result.x_opt.shape == (2 * NUM_JOINTS * NUM_HARMONICS,)
    # condition_number is always reported regardless of objective_type
    assert np.isfinite(result.condition_number)
    assert result.condition_number > 0
    assert result.n_restarts == 2


# --- JSON round-trip tests ---


def test_save_load_json_roundtrip() -> None:
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
    )
    rng = np.random.default_rng(99)
    n = NUM_JOINTS * NUM_HARMONICS
    x_opt = rng.uniform(-0.1, 0.1, size=2 * n)
    a_opt = x_opt[:n].reshape(NUM_JOINTS, NUM_HARMONICS)
    b_opt = x_opt[n:].reshape(NUM_JOINTS, NUM_HARMONICS)

    original = OptimizationResult(
        x_opt=x_opt,
        condition_number=42.5,
        a_opt=a_opt,
        b_opt=b_opt,
        q0=Q0.copy(),
        config=cfg,
        n_evaluations=100,
        wall_time=3.14,
        n_restarts=10,
        best_start_index=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.json"
        save_optimization_result(original, path)
        loaded = load_optimization_result(path)

    np.testing.assert_allclose(loaded.x_opt, original.x_opt)
    assert loaded.condition_number == pytest.approx(original.condition_number)
    np.testing.assert_allclose(loaded.a_opt, original.a_opt)
    np.testing.assert_allclose(loaded.b_opt, original.b_opt)
    np.testing.assert_allclose(loaded.q0, original.q0)
    assert loaded.n_evaluations == original.n_evaluations
    assert loaded.wall_time == pytest.approx(original.wall_time)
    assert loaded.n_restarts == original.n_restarts
    assert loaded.best_start_index == original.best_start_index
    assert loaded.config.num_joints == original.config.num_joints
    assert loaded.config.num_harmonics == original.config.num_harmonics
    assert loaded.config.base_freq == original.config.base_freq


# --- result_to_trajectory tests ---


def test_result_to_trajectory_reconstructs_valid_trajectory() -> None:
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
    )
    rng = np.random.default_rng(7)
    n = NUM_JOINTS * NUM_HARMONICS
    x_opt = rng.uniform(-0.05, 0.05, size=2 * n)

    result = OptimizationResult(
        x_opt=x_opt,
        condition_number=10.0,
        a_opt=x_opt[:n].reshape(NUM_JOINTS, NUM_HARMONICS),
        b_opt=x_opt[n:].reshape(NUM_JOINTS, NUM_HARMONICS),
        q0=Q0.copy(),
        config=cfg,
        n_evaluations=50,
        wall_time=1.0,
        n_restarts=5,
        best_start_index=0,
    )

    sample = result_to_trajectory(result)
    expected_steps = int(DURATION * FPS) + 1
    assert sample.position.shape == (expected_steps, NUM_JOINTS)
    assert sample.velocity.shape == (expected_steps, NUM_JOINTS)
    assert sample.acceleration.shape == (expected_steps, NUM_JOINTS)
    # Boundary conditions: windowed trajectory returns to q0
    np.testing.assert_allclose(sample.position[0], Q0, atol=1e-10)
    np.testing.assert_allclose(sample.position[-1], Q0, atol=1e-10)


# --- validate_trajectory tests ---


def test_validate_trajectory_returns_expected_keys() -> None:
    loaded = _load_scene()
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0,
        subsample_factor=10,
        n_monte_carlo=1,
        max_iter_per_start=2,
        seed=0,
    )
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)

    # Create a small-amplitude result for validation
    rng = np.random.default_rng(42)
    n = NUM_JOINTS * NUM_HARMONICS
    x_opt = rng.uniform(-0.02, 0.02, size=2 * n)

    result = OptimizationResult(
        x_opt=x_opt,
        condition_number=50.0,
        a_opt=x_opt[:n].reshape(NUM_JOINTS, NUM_HARMONICS),
        b_opt=x_opt[n:].reshape(NUM_JOINTS, NUM_HARMONICS),
        q0=Q0.copy(),
        config=cfg,
        n_evaluations=10,
        wall_time=0.5,
        n_restarts=1,
        best_start_index=0,
    )

    validation = opt.validate_trajectory(result)
    assert "condition_number" in validation
    assert "all_constraints_satisfied" in validation
    assert "constraint_margins" in validation
    assert isinstance(validation["condition_number"], float)
    assert isinstance(validation["all_constraints_satisfied"], bool)
    assert isinstance(validation["constraint_margins"], list)
    assert len(validation["constraint_margins"]) >= 3
