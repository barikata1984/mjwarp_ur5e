from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyKinematics:
    body_name: str
    rotation_body_to_world: np.ndarray
    angular_velocity_body: np.ndarray
    linear_velocity_body: np.ndarray
    angular_acceleration_body: np.ndarray
    linear_acceleration_body: np.ndarray
    gravity_body: np.ndarray

    @property
    def spatial_velocity_body(self) -> np.ndarray:
        return np.concatenate((self.angular_velocity_body, self.linear_velocity_body))

    @property
    def spatial_acceleration_body(self) -> np.ndarray:
        linear = self.linear_acceleration_body - self.gravity_body
        return np.concatenate((self.angular_acceleration_body, linear))


@dataclass(frozen=True)
class InertialParameters:
    mass: float
    first_moments: np.ndarray
    inertia_matrix: np.ndarray

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mass,
                self.first_moments[0],
                self.first_moments[1],
                self.first_moments[2],
                self.inertia_matrix[0, 0],
                self.inertia_matrix[1, 1],
                self.inertia_matrix[2, 2],
                self.inertia_matrix[0, 1],
                self.inertia_matrix[0, 2],
                self.inertia_matrix[1, 2],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RegressorSample:
    body_name: str
    regressor: np.ndarray
    kinematics: BodyKinematics
