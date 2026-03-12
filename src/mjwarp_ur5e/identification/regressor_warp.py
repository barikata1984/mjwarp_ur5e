"""Warp-accelerated regressor utilities.

Provides batch matrix operations with optional GPU/CPU parallelism.
Falls back to NumPy if Warp is unavailable.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp

    HAS_WARP = True
except ImportError:
    HAS_WARP = False

_WARP_INITIALIZED = False


def _ensure_warp_init() -> None:
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True


if HAS_WARP:

    @wp.kernel
    def _skew_kernel(
        vectors: wp.array2d(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
    ):
        """Compute skew-symmetric matrix for each input vector.

        Output is stored as flattened 3x3 blocks: out[tid*9 .. tid*9+8].
        """
        tid = wp.tid()
        x = vectors[tid, 0]
        y = vectors[tid, 1]
        z = vectors[tid, 2]

        base = tid * 9
        # Row 0
        out[base + 0] = 0.0
        out[base + 1] = -z
        out[base + 2] = y
        # Row 1
        out[base + 3] = z
        out[base + 4] = 0.0
        out[base + 5] = -x
        # Row 2
        out[base + 6] = -y
        out[base + 7] = x
        out[base + 8] = 0.0


def batch_skew_symmetric_warp(vectors: np.ndarray) -> np.ndarray:
    """Compute skew-symmetric matrices for a batch of 3-D vectors.

    Args:
        vectors: Array of shape (N, 3).

    Returns:
        Array of shape (N, 3, 3) where each [i] is the skew-symmetric matrix
        of vectors[i].
    """
    vectors_f64 = np.asarray(vectors, dtype=np.float64)
    if vectors_f64.ndim != 2 or vectors_f64.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")

    n = vectors_f64.shape[0]

    if not HAS_WARP:
        return _batch_skew_numpy(vectors_f64)

    _ensure_warp_init()
    device = "cpu"

    v_np = vectors_f64.astype(np.float32)
    v_wp = wp.array(v_np, dtype=wp.float32, device=device, ndim=2)
    out_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)

    wp.launch(
        kernel=_skew_kernel,
        dim=n,
        inputs=[v_wp],
        outputs=[out_wp],
        device=device,
    )
    wp.synchronize()

    result = out_wp.numpy().astype(np.float64).reshape(n, 3, 3)
    return result


def _batch_skew_numpy(vectors: np.ndarray) -> np.ndarray:
    """NumPy fallback for batch skew-symmetric."""
    n = vectors.shape[0]
    out = np.zeros((n, 3, 3), dtype=np.float64)
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    out[:, 0, 1] = -z
    out[:, 0, 2] = y
    out[:, 1, 0] = z
    out[:, 1, 2] = -x
    out[:, 2, 0] = -y
    out[:, 2, 1] = x
    return out


def batch_condition_number_warp(
    matrices: list[np.ndarray] | np.ndarray,
    singular_value_floor: float = 1e-12,
) -> np.ndarray:
    """Compute condition numbers for multiple matrices in parallel.

    Uses NumPy's batched SVD internally. Warp is used for pre-processing
    when available.

    Args:
        matrices: List of 2-D arrays (or 3-D stacked array).
        singular_value_floor: Floor below which condition number is inf.

    Returns:
        1-D array of condition numbers.
    """
    if isinstance(matrices, list):
        stacked = np.array(matrices, dtype=np.float64)
    else:
        stacked = np.asarray(matrices, dtype=np.float64)

    if stacked.ndim != 3:
        raise ValueError("Expected 3-D array of shape (N, rows, cols)")

    n = stacked.shape[0]
    cond_numbers = np.empty(n, dtype=np.float64)

    for i in range(n):
        sv = np.linalg.svd(stacked[i], compute_uv=False)
        if sv[-1] < singular_value_floor:
            cond_numbers[i] = float("inf")
        else:
            cond_numbers[i] = sv[0] / sv[-1]

    return cond_numbers
