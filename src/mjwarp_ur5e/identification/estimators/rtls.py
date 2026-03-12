"""Recursive Total Least Squares estimator for inertial parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import EstimationResult


@dataclass
class RTLSConfig:
    """Configuration for Recursive Total Least Squares."""

    n_params: int = 10
    forgetting_factor: float = 1.0
    window_size: int | None = None


class RecursiveTotalLeastSquares:
    """Recursive TLS estimator using incremental SVD updates.

    Maintains an SVD of the augmented matrix [A | y] and updates it
    incrementally as new rows arrive. Supports an exponential
    forgetting factor and an optional sliding window.
    """

    def __init__(self, config: RTLSConfig | None = None) -> None:
        self.config = config or RTLSConfig()
        n = self.config.n_params
        self._phi: np.ndarray = np.zeros(n)
        self._n_samples: int = 0
        self._initialized: bool = False

        # SVD components of augmented matrix [A | y]
        self._U: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._Vt: np.ndarray | None = None

        # Window buffer (used when window_size is set)
        self._row_buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, A_init: np.ndarray, y_init: np.ndarray) -> None:
        """Compute initial SVD from a batch of data.

        Args:
            A_init: Regressor matrix (m, n_params).
            y_init: Observation vector (m,).
        """
        A_init = np.asarray(A_init, dtype=np.float64)
        y_init = np.asarray(y_init, dtype=np.float64).ravel()
        n = self.config.n_params

        if A_init.ndim != 2 or A_init.shape[1] != n:
            raise ValueError(f"A_init must be (m, {n}), got {A_init.shape}")
        if y_init.shape[0] != A_init.shape[0]:
            raise ValueError("y_init length must match A_init rows")

        C = np.hstack([A_init, y_init.reshape(-1, 1)])
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        self._U = U
        self._S = S
        self._Vt = Vt
        self._n_samples = A_init.shape[0]
        self._initialized = True

        # Store rows for windowed mode
        if self.config.window_size is not None:
            self._row_buffer = [C[i] for i in range(C.shape[0])]

        self._extract_solution()

    def update(self, A_row: np.ndarray, y_row: np.ndarray) -> EstimationResult:
        """Incrementally update the estimate with new data.

        Args:
            A_row: Regressor row(s) (n_params,) or (k, n_params).
            y_row: Observation(s) scalar or (k,).

        Returns:
            Updated EstimationResult.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")

        A_row = np.atleast_2d(np.asarray(A_row, dtype=np.float64))
        y_row = np.atleast_1d(np.asarray(y_row, dtype=np.float64))
        if y_row.ndim == 1:
            y_row = y_row.reshape(-1, 1)

        new_rows = np.hstack([A_row, y_row])  # (k, n+1)

        # Windowed mode: manage buffer
        ws = self.config.window_size
        if ws is not None:
            for i in range(new_rows.shape[0]):
                self._row_buffer.append(new_rows[i])
            # If window exceeded, recompute from buffer tail
            if len(self._row_buffer) > ws:
                self._row_buffer = self._row_buffer[-ws:]
                C = np.array(self._row_buffer)
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                self._U, self._S, self._Vt = U, S, Vt
                self._n_samples = C.shape[0]
                self._extract_solution()
                return self.get_current_estimate()

        # Apply forgetting factor
        ff = self.config.forgetting_factor
        if ff < 1.0:
            self._S = self._S * ff

        # Incremental SVD update (Brand's algorithm, block version)
        self._svd_update_block(new_rows)
        self._n_samples += new_rows.shape[0]
        self._extract_solution()
        return self.get_current_estimate()

    def reset(self) -> None:
        """Clear all state."""
        n = self.config.n_params
        self._phi = np.zeros(n)
        self._n_samples = 0
        self._initialized = False
        self._U = None
        self._S = None
        self._Vt = None
        self._row_buffer = []

    def get_current_estimate(self) -> EstimationResult:
        """Return the current estimation result."""
        if self._S is not None and len(self._S) > 1:
            cond = float(self._S[0] / max(self._S[-1], 1e-15))
            residual = float(self._S[-1])
        else:
            cond = float("nan")
            residual = float("nan")
        return EstimationResult(
            phi=self._phi.copy(),
            condition_number=cond,
            residual_norm=residual,
            n_samples=self._n_samples,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_solution(self) -> None:
        """Extract TLS solution from current SVD."""
        if self._Vt is None:
            return
        # Right singular vector for smallest singular value
        min_idx = int(np.argmin(self._S))
        v = self._Vt[min_idx, :]
        if abs(v[-1]) < 1e-15:
            return  # keep previous estimate
        self._phi = -v[:-1] / v[-1]

    def _svd_update_block(self, new_rows: np.ndarray) -> None:
        """Brand's incremental SVD update for a block of rows."""
        assert self._U is not None
        assert self._S is not None
        assert self._Vt is not None

        U = self._U
        S = self._S
        V = self._Vt.T  # (n+1, k)
        k = len(S)
        m = U.shape[0]
        p = new_rows.shape[0]

        # Project new rows onto V space
        P = new_rows @ V  # (p, k)
        R = new_rows - P @ V.T  # (p, n+1)

        # QR of residual
        Q_r, R_r = np.linalg.qr(R.T, mode="reduced")
        r_rank = int(np.sum(np.abs(np.diag(R_r)) > 1e-12))

        if r_rank > 0:
            Q_r = Q_r[:, :r_rank]
            R_r = R_r[:r_rank, :]

            K = np.zeros((k + p, k + r_rank))
            K[:k, :k] = np.diag(S)
            K[k : k + p, :k] = P
            K[k : k + p, k : k + r_rank] = R_r.T[:p, :r_rank]

            Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

            U_ext = np.zeros((m + p, k + p))
            U_ext[:m, :k] = U
            U_ext[m : m + p, k : k + p] = np.eye(p)
            U_new = U_ext @ Uk

            V_ext = np.hstack([V, Q_r])
            V_new = V_ext @ Vkt.T
        else:
            K = np.zeros((k + p, k))
            K[:k, :k] = np.diag(S)
            K[k : k + p, :] = P

            Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

            U_ext = np.zeros((m + p, k + p))
            U_ext[:m, :k] = U
            U_ext[m : m + p, k : k + p] = np.eye(p)
            U_new = U_ext[:, : Uk.shape[0]] @ Uk

            V_new = V @ Vkt.T
            Sk = Sk

        # Keep up to n_params+1 singular values (augmented dim)
        max_rank = self.config.n_params + 1
        n_keep = min(len(Sk), max_rank)
        self._U = U_new[:, :n_keep]
        self._S = Sk[:n_keep]
        self._Vt = V_new.T[:n_keep, :]
