"""Batch Total Least Squares estimator for inertial parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import EstimationResult


@dataclass
class BatchTLSConfig:
    """Configuration for Batch Total Least Squares estimator."""

    n_params: int = 10
    regularization: float = 0.0
    truncation_rank: int | None = None


class BatchTotalLeastSquares:
    """Batch Total Least Squares (TLS) estimator.

    Accounts for errors in both A and y by solving the
    errors-in-variables problem via SVD of the augmented matrix [A | y].
    """

    def __init__(self, config: BatchTLSConfig | None = None) -> None:
        self.config = config or BatchTLSConfig()

    def estimate(self, A: np.ndarray, y: np.ndarray) -> EstimationResult:
        """Solve the TLS problem for A @ phi ~ y.

        Args:
            A: Regressor matrix (m, n_params).
            y: Observation vector (m,).

        Returns:
            EstimationResult with estimated parameters.
        """
        A = np.asarray(A, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_params = self.config.n_params

        if A.ndim != 2 or A.shape[1] != n_params:
            raise ValueError(f"A must be (m, {n_params}), got {A.shape}")
        if y.shape[0] != A.shape[0]:
            raise ValueError(f"y length ({y.shape[0]}) != A rows ({A.shape[0]})")

        # Optional regularization: append sqrt(lam)*I rows
        if self.config.regularization > 0:
            sqrt_lam = np.sqrt(self.config.regularization)
            A = np.vstack([A, sqrt_lam * np.eye(n_params)])
            y = np.concatenate([y, np.zeros(n_params)])

        # Form augmented matrix [A | y]
        C = np.hstack([A, y.reshape(-1, 1)])

        # SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=True)

        # Optional truncation
        rank = self.config.truncation_rank
        if rank is not None and rank < min(C.shape):
            # Truncated TLS: zero out small singular values,
            # reconstruct, then solve standard TLS
            S_trunc = S.copy()
            S_trunc[rank:] = 0.0
            n_sv = len(S)
            C_trunc = U[:, :n_sv] @ np.diag(S_trunc) @ Vt[:n_sv, :]
            A_t = C_trunc[:, :-1]
            y_t = C_trunc[:, -1]
            C2 = np.hstack([A_t, y_t.reshape(-1, 1)])
            _, S2, Vt2 = np.linalg.svd(C2, full_matrices=True)
            v = Vt2[-1, :]
            sv_for_residual = S2
        else:
            v = Vt[-1, :]
            sv_for_residual = S

        if abs(v[-1]) < 1e-15:
            raise ValueError(
                "TLS problem is ill-posed: last component of "
                "smallest right singular vector is near zero."
            )

        phi = -v[:-1] / v[-1]

        cond = _condition_number(A)
        residual = float(sv_for_residual[-1])
        n_samples = A.shape[0]

        return EstimationResult(
            phi=phi,
            condition_number=cond,
            residual_norm=residual,
            n_samples=n_samples,
        )


def _condition_number(A: np.ndarray) -> float:
    from mjwarp_ur5e.identification.regressor import compute_condition_number

    return compute_condition_number(A)
