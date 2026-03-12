"""Estimator algorithms for inertial parameter identification.

Provides batch and recursive estimators:
- BatchLeastSquares: OLS / Tikhonov-regularized LS
- BatchTotalLeastSquares: TLS via SVD of [A | y]
- RecursiveTotalLeastSquares: Online TLS with incremental SVD
"""

from .batch_ls import BatchLeastSquares, BatchLSConfig
from .batch_tls import BatchTLSConfig, BatchTotalLeastSquares
from .rtls import RecursiveTotalLeastSquares, RTLSConfig
from .types import EstimationResult

__all__ = [
    "BatchLeastSquares",
    "BatchLSConfig",
    "BatchTotalLeastSquares",
    "BatchTLSConfig",
    "EstimationResult",
    "RecursiveTotalLeastSquares",
    "RTLSConfig",
]
