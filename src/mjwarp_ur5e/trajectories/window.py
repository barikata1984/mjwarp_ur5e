from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseTrajectory, BaseTrajectoryConfig, TrajectorySample


@dataclass(frozen=True, kw_only=True)
class WindowTrajectoryConfig(BaseTrajectoryConfig):
    num_joints: int


class WindowTrajectory(BaseTrajectory):
    def __init__(self, config: WindowTrajectoryConfig) -> None:
        super().__init__(config)
        if config.num_joints <= 0:
            raise ValueError("num_joints must be positive")
        self.config = config

    def scalar_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        duration = self.config.duration
        normalized_time = self._time / duration

        s = normalized_time
        window = 64.0 * (s**3 - 3.0 * s**4 + 3.0 * s**5 - s**6)
        d_window_ds = 192.0 * s**2 - 768.0 * s**3 + 960.0 * s**4 - 384.0 * s**5
        dd_window_ds2 = 384.0 * s - 2304.0 * s**2 + 3840.0 * s**3 - 1920.0 * s**4

        d_window_dt = d_window_ds / duration
        dd_window_dt2 = dd_window_ds2 / (duration**2)
        return window, d_window_dt, dd_window_dt2

    def sample(self) -> TrajectorySample:
        window, d_window_dt, dd_window_dt2 = self.scalar_profile()
        position = np.repeat(window[:, None], self.config.num_joints, axis=1)
        velocity = np.repeat(d_window_dt[:, None], self.config.num_joints, axis=1)
        acceleration = np.repeat(dd_window_dt2[:, None], self.config.num_joints, axis=1)
        return TrajectorySample(
            time=self.time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )
