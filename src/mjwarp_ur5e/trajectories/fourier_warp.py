"""Warp-accelerated Fourier trajectory sampling.

Provides GPU/CPU-parallel evaluation of Fourier series trajectories.
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

# Maximum dimensions supported by static Warp kernel arrays
_MAX_JOINTS = 12
_MAX_HARMONICS = 12


def _ensure_warp_init() -> None:
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True


if HAS_WARP:

    @wp.kernel
    def _fourier_kernel(
        time: wp.array(dtype=wp.float32),
        q0: wp.array(dtype=wp.float32),
        a_flat: wp.array(dtype=wp.float32),
        b_flat: wp.array(dtype=wp.float32),
        base_freq: wp.float32,
        num_joints: wp.int32,
        num_harmonics: wp.int32,
        q_out: wp.array2d(dtype=wp.float32),
        dq_out: wp.array2d(dtype=wp.float32),
        ddq_out: wp.array2d(dtype=wp.float32),
    ):
        tid = wp.tid()
        t = time[tid]
        two_pi = 6.28318530717958647692

        for j in range(num_joints):
            q_val = q0[j]
            dq_val = wp.float32(0.0)
            ddq_val = wp.float32(0.0)

            for k in range(num_harmonics):
                omega = two_pi * base_freq * wp.float32(k + 1)
                phase = omega * t
                s = wp.sin(phase)
                c = wp.cos(phase)

                idx = j * num_harmonics + k
                a_jk = a_flat[idx]
                b_jk = b_flat[idx]

                q_val = q_val + a_jk * s + b_jk * c
                dq_val = dq_val + a_jk * omega * c - b_jk * omega * s
                ddq_val = ddq_val - a_jk * omega * omega * s - b_jk * omega * omega * c

            q_out[tid, j] = q_val
            dq_out[tid, j] = dq_val
            ddq_out[tid, j] = ddq_val


def fourier_trajectory_warp(
    time: np.ndarray,
    q0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    base_freq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Fourier trajectory samples using Warp (parallel across timesteps).

    Args:
        time: 1-D array of timesteps, shape (N,).
        q0: Home joint positions, shape (num_joints,).
        a: Sine coefficients, shape (num_joints, num_harmonics).
        b: Cosine coefficients, shape (num_joints, num_harmonics).
        base_freq: Fundamental frequency in Hz.

    Returns:
        (q, dq, ddq) each with shape (N, num_joints).
    """
    if not HAS_WARP:
        return _fourier_trajectory_numpy(time, q0, a, b, base_freq)

    _ensure_warp_init()

    time_f64 = np.asarray(time, dtype=np.float64).ravel()
    q0_f64 = np.asarray(q0, dtype=np.float64).ravel()
    a_f64 = np.asarray(a, dtype=np.float64)
    b_f64 = np.asarray(b, dtype=np.float64)

    num_steps = time_f64.shape[0]
    num_joints = a_f64.shape[0]
    num_harmonics = a_f64.shape[1]

    if num_joints > _MAX_JOINTS or num_harmonics > _MAX_HARMONICS:
        return _fourier_trajectory_numpy(time, q0, a, b, base_freq)

    device = "cpu"

    time_wp = wp.array(time_f64.astype(np.float32), dtype=wp.float32, device=device)
    q0_wp = wp.array(q0_f64.astype(np.float32), dtype=wp.float32, device=device)
    a_wp = wp.array(a_f64.astype(np.float32).ravel(), dtype=wp.float32, device=device)
    b_wp = wp.array(b_f64.astype(np.float32).ravel(), dtype=wp.float32, device=device)

    q_out = wp.zeros((num_steps, num_joints), dtype=wp.float32, device=device)
    dq_out = wp.zeros((num_steps, num_joints), dtype=wp.float32, device=device)
    ddq_out = wp.zeros((num_steps, num_joints), dtype=wp.float32, device=device)

    wp.launch(
        kernel=_fourier_kernel,
        dim=num_steps,
        inputs=[
            time_wp,
            q0_wp,
            a_wp,
            b_wp,
            float(base_freq),
            int(num_joints),
            int(num_harmonics),
        ],
        outputs=[q_out, dq_out, ddq_out],
        device=device,
    )
    wp.synchronize()

    q = q_out.numpy().astype(np.float64)
    dq = dq_out.numpy().astype(np.float64)
    ddq = ddq_out.numpy().astype(np.float64)

    return q, dq, ddq


def _fourier_trajectory_numpy(
    time: np.ndarray,
    q0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    base_freq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy fallback matching FourierTrajectory.sample() logic."""
    time = np.asarray(time, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    num_harmonics = a.shape[1]
    harmonics = np.arange(1, num_harmonics + 1, dtype=np.float64)
    omega = 2.0 * np.pi * base_freq * harmonics
    phase = np.outer(time, omega)

    sin_p = np.sin(phase)
    cos_p = np.cos(phase)

    q = q0 + sin_p @ a.T + cos_p @ b.T
    dq = (cos_p * omega) @ a.T - (sin_p * omega) @ b.T
    ddq = -(sin_p * omega**2) @ a.T - (cos_p * omega**2) @ b.T

    return q, dq, ddq


def windowed_fourier_trajectory_warp(
    time: np.ndarray,
    q0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    base_freq: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute windowed Fourier trajectory using Warp for the Fourier part.

    The window function is applied in NumPy (simple elementwise ops).

    Args:
        time: 1-D array of timesteps, shape (N,).
        q0: Home joint positions, shape (num_joints,).
        a: Sine coefficients, shape (num_joints, num_harmonics).
        b: Cosine coefficients, shape (num_joints, num_harmonics).
        base_freq: Fundamental frequency in Hz.
        duration: Total trajectory duration in seconds.

    Returns:
        (q, dq, ddq) each with shape (N, num_joints).
    """
    num_joints = np.asarray(a).shape[0]
    zero_q0 = np.zeros(num_joints, dtype=np.float64)

    osc_q, osc_dq, osc_ddq = fourier_trajectory_warp(time, zero_q0, a, b, base_freq)

    time_f64 = np.asarray(time, dtype=np.float64)
    q0_f64 = np.asarray(q0, dtype=np.float64)

    s = time_f64 / duration
    window = 64.0 * (s**3 - 3.0 * s**4 + 3.0 * s**5 - s**6)
    dw_ds = 192.0 * s**2 - 768.0 * s**3 + 960.0 * s**4 - 384.0 * s**5
    ddw_ds2 = 384.0 * s - 2304.0 * s**2 + 3840.0 * s**3 - 1920.0 * s**4

    dw_dt = dw_ds / duration
    ddw_dt2 = ddw_ds2 / (duration**2)

    q = q0_f64 + window[:, None] * osc_q
    dq = dw_dt[:, None] * osc_q + window[:, None] * osc_dq
    ddq = ddw_dt2[:, None] * osc_q + 2.0 * dw_dt[:, None] * osc_dq + window[:, None] * osc_ddq

    return q, dq, ddq
