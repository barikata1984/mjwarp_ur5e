from __future__ import annotations

from dataclasses import dataclass

from .base import TrajectorySample
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig


@dataclass(frozen=True, kw_only=True)
class WindowedFourierTrajectoryConfig(FourierTrajectoryConfig):
    pass


class WindowedFourierTrajectory:
    def __init__(self, config: WindowedFourierTrajectoryConfig) -> None:
        self.config = config
        self._fourier = FourierTrajectory(
            FourierTrajectoryConfig(
                duration=config.duration,
                fps=config.fps,
                num_joints=config.num_joints,
                num_harmonics=config.num_harmonics,
                base_freq=config.base_freq,
                coefficients=config.coefficients,
                q0=[0.0] * config.num_joints,
            )
        )
        self._window = WindowTrajectory(
            WindowTrajectoryConfig(
                duration=config.duration,
                fps=config.fps,
                num_joints=config.num_joints,
            )
        )
        self._base = FourierTrajectory._parse_q0(config.q0, config.num_joints)

    @property
    def time(self):
        return self._fourier.time

    def sample(self) -> TrajectorySample:
        oscillation = self._fourier.sample()
        window, d_window_dt, dd_window_dt2 = self._window.scalar_profile()

        position = self._base + window[:, None] * oscillation.position
        velocity = (
            d_window_dt[:, None] * oscillation.position + window[:, None] * oscillation.velocity
        )
        acceleration = (
            dd_window_dt2[:, None] * oscillation.position
            + 2.0 * d_window_dt[:, None] * oscillation.velocity
            + window[:, None] * oscillation.acceleration
        )
        return TrajectorySample(
            time=self.time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )

    def as_serializable_coefficients(self) -> dict[str, list[list[float]]]:
        return self._fourier.as_serializable_coefficients()
