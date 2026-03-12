"""Tests for identification estimator algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from mjwarp_ur5e.identification.estimators import (
    BatchLeastSquares,
    BatchLSConfig,
    BatchTLSConfig,
    BatchTotalLeastSquares,
    EstimationResult,
    RecursiveTotalLeastSquares,
    RTLSConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_problem(rng: np.random.Generator):
    """Generate synthetic A (60x10), phi_true (10,), y = A @ phi_true."""
    A = rng.standard_normal((60, 10))
    phi_true = rng.standard_normal(10)
    phi_true[0] = abs(phi_true[0]) + 0.5  # positive mass
    y = A @ phi_true
    return A, y, phi_true


# ---------------------------------------------------------------------------
# EstimationResult tests
# ---------------------------------------------------------------------------


class TestEstimationResult:
    def test_properties(self) -> None:
        phi = np.array([2.0, 0.4, 0.6, 0.2, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        result = EstimationResult(phi=phi, condition_number=5.0, residual_norm=1e-6, n_samples=10)

        assert result.mass == pytest.approx(2.0)
        np.testing.assert_allclose(result.center_of_mass, [0.2, 0.3, 0.1])
        assert result.inertia_matrix.shape == (3, 3)
        # Symmetry
        np.testing.assert_allclose(result.inertia_matrix, result.inertia_matrix.T)

    def test_inertia_at_com(self) -> None:
        # For a point mass at origin (h=0), I_com should equal I_origin
        phi = np.array([3.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        result = EstimationResult(phi=phi)
        np.testing.assert_allclose(result.inertia_at_com, result.inertia_matrix)

    def test_inertia_at_com_parallel_axis(self) -> None:
        m = 2.0
        cx, cy, cz = 0.1, 0.2, 0.3
        hx, hy, hz = m * cx, m * cy, m * cz
        # Identity inertia at origin
        I_origin = np.eye(3)
        phi = np.array(
            [
                m,
                hx,
                hy,
                hz,
                I_origin[0, 0],
                I_origin[1, 1],
                I_origin[2, 2],
                I_origin[0, 1],
                I_origin[0, 2],
                I_origin[1, 2],
            ]
        )
        result = EstimationResult(phi=phi)
        c = np.array([cx, cy, cz])
        expected = I_origin - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        np.testing.assert_allclose(result.inertia_at_com, expected, atol=1e-12)

    def test_to_dict_roundtrip(self) -> None:
        phi = np.array([1.0, 0.1, 0.2, 0.3, 0.04, 0.05, 0.06, 0.0, 0.0, 0.0])
        result = EstimationResult(phi=phi, condition_number=10.0, residual_norm=1e-3, n_samples=20)
        d = result.to_dict()
        assert d["mass"] == pytest.approx(1.0)
        assert len(d["phi"]) == 10
        assert len(d["center_of_mass"]) == 3
        assert len(d["inertia_matrix"]) == 3
        assert len(d["inertia_at_com"]) == 3

        # Reconstruct from dict
        result2 = EstimationResult(
            phi=np.array(d["phi"]),
            condition_number=d["condition_number"],
            residual_norm=d["residual_norm"],
            n_samples=d["n_samples"],
        )
        np.testing.assert_allclose(result2.phi, result.phi)

    def test_str(self) -> None:
        phi = np.ones(10)
        result = EstimationResult(phi=phi, condition_number=5.0, residual_norm=0.001, n_samples=50)
        s = str(result)
        assert "mass" in s
        assert "CoM" in s
        assert "cond" in s

    def test_zero_mass_safety(self) -> None:
        phi = np.zeros(10)
        result = EstimationResult(phi=phi)
        np.testing.assert_array_equal(result.center_of_mass, [0, 0, 0])
        # inertia_at_com should return inertia_matrix when mass is 0
        np.testing.assert_array_equal(result.inertia_at_com, result.inertia_matrix)


# ---------------------------------------------------------------------------
# BatchLS tests
# ---------------------------------------------------------------------------


class TestBatchLeastSquares:
    def test_recovers_true_params_noise_free(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem
        estimator = BatchLeastSquares()
        result = estimator.estimate(A, y)
        np.testing.assert_allclose(result.phi, phi_true, atol=1e-10)
        assert result.residual_norm < 1e-10
        assert result.condition_number > 0
        assert np.isfinite(result.condition_number)
        assert result.n_samples == 60

    def test_with_regularization(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem
        config = BatchLSConfig(regularization=0.01)
        estimator = BatchLeastSquares(config)
        result = estimator.estimate(A, y)
        # With regularization, won't exactly recover but should be close
        np.testing.assert_allclose(result.phi, phi_true, atol=0.5)
        assert result.n_samples == 60

    def test_regularization_shrinks_norm(self, synthetic_problem) -> None:
        A, y, _ = synthetic_problem
        est_plain = BatchLeastSquares().estimate(A, y)
        est_reg = BatchLeastSquares(BatchLSConfig(regularization=1.0)).estimate(A, y)
        assert np.linalg.norm(est_reg.phi) <= np.linalg.norm(est_plain.phi)


# ---------------------------------------------------------------------------
# BatchTLS tests
# ---------------------------------------------------------------------------


class TestBatchTotalLeastSquares:
    def test_recovers_true_params_noise_free(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem
        estimator = BatchTotalLeastSquares()
        result = estimator.estimate(A, y)
        np.testing.assert_allclose(result.phi, phi_true, atol=1e-8)
        assert result.n_samples == 60

    def test_with_truncation(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem
        # Use rank=10 (full rank for 10 params) to keep accuracy
        config = BatchTLSConfig(truncation_rank=10)
        estimator = BatchTotalLeastSquares(config)
        result = estimator.estimate(A, y)
        np.testing.assert_allclose(result.phi, phi_true, atol=1e-6)

    def test_truncation_reduces_rank(self, rng) -> None:
        """Truncation with lower rank still produces a result."""
        A = rng.standard_normal((60, 10))
        phi_true = rng.standard_normal(10)
        y = A @ phi_true
        config = BatchTLSConfig(truncation_rank=8)
        estimator = BatchTotalLeastSquares(config)
        result = estimator.estimate(A, y)
        # Should produce some estimate (may not be very accurate)
        assert result.phi.shape == (10,)
        assert np.all(np.isfinite(result.phi))

    def test_with_regularization(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem
        config = BatchTLSConfig(regularization=0.001)
        estimator = BatchTotalLeastSquares(config)
        result = estimator.estimate(A, y)
        np.testing.assert_allclose(result.phi, phi_true, atol=0.5)


# ---------------------------------------------------------------------------
# RTLS tests
# ---------------------------------------------------------------------------


class TestRecursiveTotalLeastSquares:
    def test_converges_to_batch(self, synthetic_problem) -> None:
        A, y, phi_true = synthetic_problem

        # Batch TLS reference
        batch = BatchTotalLeastSquares().estimate(A, y)

        # RTLS: initialize with first 30 rows, update with rest
        rtls = RecursiveTotalLeastSquares()
        rtls.initialize(A[:30], y[:30])
        for i in range(30, 60):
            rtls.update(A[i : i + 1], y[i : i + 1])

        result = rtls.get_current_estimate()
        np.testing.assert_allclose(result.phi, batch.phi, atol=1e-6)

    def test_reset(self, synthetic_problem) -> None:
        A, y, _ = synthetic_problem
        rtls = RecursiveTotalLeastSquares()
        rtls.initialize(A, y)
        assert rtls._initialized
        rtls.reset()
        assert not rtls._initialized
        np.testing.assert_array_equal(rtls._phi, np.zeros(10))

    def test_with_forgetting_factor(self, rng) -> None:
        n = 10
        # Two different parameter vectors
        phi1 = rng.standard_normal(n)
        phi2 = rng.standard_normal(n)

        A1 = rng.standard_normal((60, n))
        y1 = A1 @ phi1

        A2 = rng.standard_normal((60, n))
        y2 = A2 @ phi2

        config = RTLSConfig(forgetting_factor=0.95)
        rtls = RecursiveTotalLeastSquares(config)
        rtls.initialize(A1, y1)

        # Feed in data from phi2
        for i in range(60):
            rtls.update(A2[i : i + 1], y2[i : i + 1])

        result = rtls.get_current_estimate()
        # Should be closer to phi2 than phi1 due to forgetting
        err1 = np.linalg.norm(result.phi - phi1)
        err2 = np.linalg.norm(result.phi - phi2)
        assert err2 < err1

    def test_windowed(self, rng) -> None:
        n = 10
        phi_true = rng.standard_normal(n)
        A = rng.standard_normal((100, n))
        y = A @ phi_true

        config = RTLSConfig(window_size=50)
        rtls = RecursiveTotalLeastSquares(config)
        rtls.initialize(A[:20], y[:20])

        for i in range(20, 100):
            rtls.update(A[i : i + 1], y[i : i + 1])

        result = rtls.get_current_estimate()
        np.testing.assert_allclose(result.phi, phi_true, atol=1e-6)

    def test_update_before_init_raises(self) -> None:
        rtls = RecursiveTotalLeastSquares()
        with pytest.raises(RuntimeError):
            rtls.update(np.zeros((1, 10)), np.zeros(1))

    def test_get_current_estimate_has_metrics(self, synthetic_problem) -> None:
        A, y, _ = synthetic_problem
        rtls = RecursiveTotalLeastSquares()
        rtls.initialize(A, y)
        result = rtls.get_current_estimate()
        assert np.isfinite(result.condition_number)
        assert np.isfinite(result.residual_norm)
        assert result.n_samples == 60
