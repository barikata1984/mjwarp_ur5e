from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class BaseTrajectoryConfig:
    duration: float
    fps: float


@dataclass(frozen=True)
class TrajectorySample:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class BaseTrajectory:
    def __init__(self, config: BaseTrajectoryConfig) -> None:
        self.config = config
        self._time = self._build_time_array(config.duration, config.fps)

    @property
    def time(self) -> np.ndarray:
        view = self._time.view()
        view.flags.writeable = False
        return view

    @staticmethod
    def _build_time_array(duration: float, fps: float) -> np.ndarray:
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if fps <= 0.0:
            raise ValueError("fps must be positive")

        num_steps = int(round(duration * fps)) + 1
        if num_steps < 2:
            raise ValueError("duration and fps must produce at least two samples")

        return np.linspace(0.0, duration, num_steps, dtype=np.float64)

    def sample(self) -> TrajectorySample:
        raise NotImplementedError
