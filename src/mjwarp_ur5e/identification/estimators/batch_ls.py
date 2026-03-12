"""Batch Least Squares estimator for inertial parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import EstimationResult


@dataclass
class BatchLSConfig:
    """Configuration for Batch Least Squares estimator."""

    regularization: float = 0.0


class BatchLeastSquares:
    """Batch Ordinary Least Squares (OLS) estimator.

    Solves y = A @ phi via pseudoinverse or Tikhonov regularization.
    """

    def __init__(self, config: BatchLSConfig | None = None) -> None:
        self.config = config or BatchLSConfig()

    def estimate(self, A: np.ndarray, y: np.ndarray) -> EstimationResult:
        """Solve A @ phi = y for phi.

        Args:
            A: Regressor matrix (m, 10).
            y: Observation vector (m,).

        Returns:
            EstimationResult with estimated parameters.
        """
        A = np.asarray(A, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if A.ndim != 2 or A.shape[1] != 10:
            raise ValueError(f"A must be (m, 10), got {A.shape}")
        if y.shape[0] != A.shape[0]:
            raise ValueError(f"y length ({y.shape[0]}) != A rows ({A.shape[0]})")

        lam = self.config.regularization
        if lam > 0:
            AtA = A.T @ A
            Aty = A.T @ y
            phi = np.linalg.solve(AtA + lam * np.eye(10), Aty)
        else:
            phi, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        from mjwarp_ur5e.identification.regressor import compute_condition_number

        return EstimationResult(
            phi=phi,
            condition_number=compute_condition_number(A),
            residual_norm=float(np.linalg.norm(y - A @ phi)),
            n_samples=A.shape[0],
        )
