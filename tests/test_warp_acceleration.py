"""Tests for Warp-accelerated trajectory and regressor utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mjwarp_ur5e.trajectories.fourier_warp import HAS_WARP

pytestmark = pytest.mark.skipif(not HAS_WARP, reason="warp not available")


# ---------------------------------------------------------------------------
# Fourier trajectory
# ---------------------------------------------------------------------------


class TestFourierTrajectoryWarp:
    """Compare Warp Fourier trajectory against NumPy reference."""

    @pytest.fixture()
    def coefficients(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        num_joints = 6
        num_harmonics = 5
        a = rng.uniform(-0.3, 0.3, size=(num_joints, num_harmonics))
        b = rng.uniform(-0.3, 0.3, size=(num_joints, num_harmonics))
        q0 = rng.uniform(-1.0, 1.0, size=num_joints)
        return {"a": a, "b": b, "q0": q0}

    @pytest.fixture()
    def time_array(self) -> np.ndarray:
        return np.linspace(0.0, 10.0, 1001, dtype=np.float64)

    def test_matches_numpy(
        self,
        coefficients: dict[str, np.ndarray],
        time_array: np.ndarray,
    ) -> None:
        from mjwarp_ur5e.trajectories.fourier_warp import (
            _fourier_trajectory_numpy,
            fourier_trajectory_warp,
        )

        a, b, q0 = coefficients["a"], coefficients["b"], coefficients["q0"]
        base_freq = 0.1

        q_np, dq_np, ddq_np = _fourier_trajectory_numpy(time_array, q0, a, b, base_freq)
        q_wp, dq_wp, ddq_wp = fourier_trajectory_warp(time_array, q0, a, b, base_freq)

        np.testing.assert_allclose(q_wp, q_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dq_wp, dq_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ddq_wp, ddq_np, rtol=1e-4, atol=1e-4)

    def test_output_shapes(
        self,
        coefficients: dict[str, np.ndarray],
        time_array: np.ndarray,
    ) -> None:
        from mjwarp_ur5e.trajectories.fourier_warp import fourier_trajectory_warp

        a, b, q0 = coefficients["a"], coefficients["b"], coefficients["q0"]
        q, dq, ddq = fourier_trajectory_warp(time_array, q0, a, b, 0.1)

        n = time_array.shape[0]
        nj = a.shape[0]
        assert q.shape == (n, nj)
        assert dq.shape == (n, nj)
        assert ddq.shape == (n, nj)

    def test_zero_coefficients(self) -> None:
        from mjwarp_ur5e.trajectories.fourier_warp import fourier_trajectory_warp

        nj, nh = 6, 3
        q0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a = np.zeros((nj, nh))
        b = np.zeros((nj, nh))
        t = np.linspace(0.0, 5.0, 100)

        q, dq, ddq = fourier_trajectory_warp(t, q0, a, b, 0.1)

        np.testing.assert_allclose(q, np.tile(q0, (100, 1)), atol=1e-6)
        np.testing.assert_allclose(dq, 0.0, atol=1e-6)
        np.testing.assert_allclose(ddq, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Windowed Fourier trajectory
# ---------------------------------------------------------------------------


class TestWindowedFourierTrajectoryWarp:
    """Compare Warp windowed Fourier against NumPy WindowedFourierTrajectory."""

    def test_matches_reference(self) -> None:
        from mjwarp_ur5e.trajectories import (
            WindowedFourierTrajectory,
            WindowedFourierTrajectoryConfig,
        )
        from mjwarp_ur5e.trajectories.fourier_warp import (
            windowed_fourier_trajectory_warp,
        )

        rng = np.random.default_rng(42)
        nj, nh = 6, 5
        base_freq = 0.1
        duration = 10.0
        fps = 100.0

        a = rng.uniform(-0.2, 0.2, size=(nj, nh))
        b = rng.uniform(-0.2, 0.2, size=(nj, nh))
        q0 = np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])

        config = WindowedFourierTrajectoryConfig(
            duration=duration,
            fps=fps,
            num_joints=nj,
            num_harmonics=nh,
            base_freq=base_freq,
            coefficients={"a": a.tolist(), "b": b.tolist()},
            q0=q0.tolist(),
        )
        ref = WindowedFourierTrajectory(config).sample()

        time = np.linspace(0.0, duration, int(round(duration * fps)) + 1)
        q_wp, dq_wp, ddq_wp = windowed_fourier_trajectory_warp(time, q0, a, b, base_freq, duration)

        np.testing.assert_allclose(q_wp, ref.position, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dq_wp, ref.velocity, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ddq_wp, ref.acceleration, rtol=1e-4, atol=1e-4)

    def test_boundary_values(self) -> None:
        from mjwarp_ur5e.trajectories.fourier_warp import (
            windowed_fourier_trajectory_warp,
        )

        nj, nh = 3, 2
        q0 = np.array([1.0, 2.0, 3.0])
        a = np.ones((nj, nh)) * 0.5
        b = np.ones((nj, nh)) * 0.3
        duration = 5.0
        t = np.array([0.0, duration])

        q, dq, ddq = windowed_fourier_trajectory_warp(t, q0, a, b, 0.1, duration)

        # Window is zero at boundaries -> position = q0
        np.testing.assert_allclose(q[0], q0, atol=1e-6)
        np.testing.assert_allclose(q[-1], q0, atol=1e-6)


# ---------------------------------------------------------------------------
# Batch skew-symmetric
# ---------------------------------------------------------------------------


class TestBatchSkewSymmetricWarp:
    def test_matches_numpy(self) -> None:
        from mjwarp_ur5e.identification.regressor_warp import (
            _batch_skew_numpy,
            batch_skew_symmetric_warp,
        )

        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((50, 3))

        ref = _batch_skew_numpy(vectors)
        result = batch_skew_symmetric_warp(vectors)

        np.testing.assert_allclose(result, ref, atol=1e-5)

    def test_antisymmetric(self) -> None:
        from mjwarp_ur5e.identification.regressor_warp import (
            batch_skew_symmetric_warp,
        )

        vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = batch_skew_symmetric_warp(vectors)

        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i], -result[i].T, atol=1e-6)


# ---------------------------------------------------------------------------
# Batch condition number
# ---------------------------------------------------------------------------


class TestBatchConditionNumberWarp:
    def test_known_values(self) -> None:
        from mjwarp_ur5e.identification.regressor_warp import (
            batch_condition_number_warp,
        )

        identity = np.eye(3, dtype=np.float64)
        scaled = 2.0 * np.eye(3, dtype=np.float64)
        matrices = [identity, scaled]

        conds = batch_condition_number_warp(matrices)

        np.testing.assert_allclose(conds[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(conds[1], 1.0, atol=1e-10)

    def test_singular_matrix(self) -> None:
        from mjwarp_ur5e.identification.regressor_warp import (
            batch_condition_number_warp,
        )

        singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        conds = batch_condition_number_warp([singular])
        assert conds[0] == float("inf")

    def test_stacked_array_input(self) -> None:
        from mjwarp_ur5e.identification.regressor_warp import (
            batch_condition_number_warp,
        )

        rng = np.random.default_rng(99)
        stacked = rng.standard_normal((5, 4, 4))
        conds = batch_condition_number_warp(stacked)
        assert conds.shape == (5,)
        assert np.all(conds > 0)
