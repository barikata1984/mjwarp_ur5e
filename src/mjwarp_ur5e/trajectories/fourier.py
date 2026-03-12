from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseTrajectory, BaseTrajectoryConfig, TrajectorySample


@dataclass(frozen=True, kw_only=True)
class FourierTrajectoryConfig(BaseTrajectoryConfig):
    num_joints: int
    num_harmonics: int
    base_freq: float
    coefficients: dict[str, list[list[float]] | np.ndarray] | None = None
    q0: list[float] | np.ndarray | None = None


@dataclass(frozen=True)
class FourierCoefficients:
    a: np.ndarray
    b: np.ndarray


class FourierTrajectory(BaseTrajectory):
    def __init__(self, config: FourierTrajectoryConfig) -> None:
        super().__init__(config)
        if config.num_joints <= 0:
            raise ValueError("num_joints must be positive")
        if config.num_harmonics <= 0:
            raise ValueError("num_harmonics must be positive")
        if config.base_freq <= 0.0:
            raise ValueError("base_freq must be positive")

        self.config = config
        self.coefficients = self._parse_coefficients(config)
        self.q0 = self._parse_q0(config.q0, config.num_joints)

    @staticmethod
    def _parse_q0(
        q0: list[float] | np.ndarray | None,
        num_joints: int,
    ) -> np.ndarray:
        if q0 is None:
            return np.zeros(num_joints, dtype=np.float64)
        q0_array = np.asarray(q0, dtype=np.float64)
        if q0_array.shape != (num_joints,):
            raise ValueError(f"q0 must have shape ({num_joints},)")
        return q0_array

    @staticmethod
    def _parse_coefficients(config: FourierTrajectoryConfig) -> FourierCoefficients:
        shape = (config.num_joints, config.num_harmonics)
        if config.coefficients is None:
            return FourierCoefficients(
                a=np.zeros(shape, dtype=np.float64),
                b=np.zeros(shape, dtype=np.float64),
            )

        if "a" not in config.coefficients or "b" not in config.coefficients:
            raise ValueError("coefficients must contain 'a' and 'b'")

        a = np.asarray(config.coefficients["a"], dtype=np.float64)
        b = np.asarray(config.coefficients["b"], dtype=np.float64)
        if a.shape != shape or b.shape != shape:
            raise ValueError(f"Fourier coefficients must have shape {shape}")
        return FourierCoefficients(a=a, b=b)

    def sample(self) -> TrajectorySample:
        harmonics = np.arange(1, self.config.num_harmonics + 1, dtype=np.float64)
        angular_frequency = 2.0 * np.pi * self.config.base_freq * harmonics
        phase = np.outer(self._time, angular_frequency)

        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)

        position = self.q0 + sin_phase @ self.coefficients.a.T + cos_phase @ self.coefficients.b.T
        velocity = (cos_phase * angular_frequency) @ self.coefficients.a.T - (
            sin_phase * angular_frequency
        ) @ self.coefficients.b.T
        acceleration = (
            -(sin_phase * angular_frequency**2) @ self.coefficients.a.T
            - (cos_phase * angular_frequency**2) @ self.coefficients.b.T
        )

        return TrajectorySample(
            time=self.time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )

    def as_serializable_coefficients(self) -> dict[str, list[list[float]]]:
        return {
            "a": self.coefficients.a.tolist(),
            "b": self.coefficients.b.tolist(),
        }
