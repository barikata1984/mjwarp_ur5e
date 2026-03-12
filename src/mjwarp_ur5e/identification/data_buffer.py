"""Time-series data collection for trajectory playback measurements."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .regressor import (
    body_inertial_parameters_from_model,
    compute_wrench_from_parameters,
    sample_body_regressor,
)
from .sampling import set_model_state


@dataclass(frozen=True)
class SensorSample:
    """Single timestep measurement from trajectory playback."""

    timestamp: float
    q: np.ndarray  # (6,)
    dq: np.ndarray  # (6,)
    ddq: np.ndarray  # (6,)
    ee_position: np.ndarray  # (3,)
    ee_rotation: np.ndarray  # (3, 3)
    wrench: np.ndarray | None  # (6,) or None


class DataBuffer:
    """Accumulates SensorSample instances for batch processing."""

    def __init__(self) -> None:
        self._samples: list[SensorSample] = []

    def append(self, sample: SensorSample) -> None:
        """Add a sample to the buffer."""
        self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert stored samples to a dict of stacked arrays.

        Returns dict with keys: timestamp, q, dq, ddq,
        ee_position, ee_rotation, wrench.
        """
        if len(self._samples) == 0:
            return {
                "timestamp": np.empty((0,), dtype=np.float64),
                "q": np.empty((0, 6), dtype=np.float64),
                "dq": np.empty((0, 6), dtype=np.float64),
                "ddq": np.empty((0, 6), dtype=np.float64),
                "ee_position": np.empty((0, 3), dtype=np.float64),
                "ee_rotation": np.empty((0, 3, 3), dtype=np.float64),
                "wrench": np.empty((0, 6), dtype=np.float64),
            }

        timestamps = np.array([s.timestamp for s in self._samples], dtype=np.float64)
        q = np.array([s.q for s in self._samples], dtype=np.float64)
        dq = np.array([s.dq for s in self._samples], dtype=np.float64)
        ddq = np.array([s.ddq for s in self._samples], dtype=np.float64)
        ee_position = np.array([s.ee_position for s in self._samples], dtype=np.float64)
        ee_rotation = np.array([s.ee_rotation for s in self._samples], dtype=np.float64)

        # Handle wrench: use zeros if any sample has None
        wrench_list: list[np.ndarray] = []
        for s in self._samples:
            if s.wrench is not None:
                wrench_list.append(s.wrench)
            else:
                wrench_list.append(np.zeros(6, dtype=np.float64))
        wrench = np.array(wrench_list, dtype=np.float64)

        return {
            "timestamp": timestamps,
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "ee_position": ee_position,
            "ee_rotation": ee_rotation,
            "wrench": wrench,
        }

    def build_regressor_data(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build stacked regressor and wrench vectors.

        For each stored sample, sets the model state, computes
        the body regressor, and computes the wrench from the
        model's inertial parameters.

        Returns (A_stacked, y_stacked) where:
            A_stacked: (N*6, 10) regressor matrix
            y_stacked: (N*6,) wrench vector
        """
        params = body_inertial_parameters_from_model(model, body_name)

        regressor_rows: list[np.ndarray] = []
        wrench_rows: list[np.ndarray] = []

        for sample in self._samples:
            set_model_state(model, data, sample.q, sample.dq, sample.ddq)
            reg_sample = sample_body_regressor(model, data, body_name)
            regressor_rows.append(reg_sample.regressor)
            wrench = compute_wrench_from_parameters(reg_sample.regressor, params)
            wrench_rows.append(wrench)

        a_stacked = np.vstack(regressor_rows)
        y_stacked = np.concatenate(wrench_rows)
        return a_stacked, y_stacked
