"""Estimation result types for inertial parameter identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EstimationResult:
    """Result of an inertial parameter estimation.

    The parameter vector phi follows the convention:
        [m, h_x, h_y, h_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]

    where h = m * c is the first moment of mass (mass * center of mass),
    and I is the rotational inertia about the body frame origin.
    """

    phi: np.ndarray  # (10,) estimated parameter vector
    condition_number: float = float("nan")
    residual_norm: float = float("nan")
    n_samples: int = 0

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=np.float64).ravel()
        if self.phi.shape != (10,):
            raise ValueError(f"phi must have 10 elements, got {self.phi.shape[0]}")

    @property
    def mass(self) -> float:
        return float(self.phi[0])

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass (3,) in body frame. Returns h/m."""
        m = self.phi[0]
        if abs(m) < 1e-15:
            return np.zeros(3)
        return self.phi[1:4] / m

    @property
    def inertia_matrix(self) -> np.ndarray:
        """Rotational inertia (3,3) about the body frame origin."""
        Ixx, Iyy, Izz = self.phi[4], self.phi[5], self.phi[6]
        Ixy, Ixz, Iyz = self.phi[7], self.phi[8], self.phi[9]
        return np.array(
            [
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz],
            ]
        )

    @property
    def inertia_at_com(self) -> np.ndarray:
        """Rotational inertia (3,3) about the center of mass.

        Uses the parallel axis theorem:
            I_com = I_origin - m * (c^T c * I3 - c c^T)
        """
        m = self.phi[0]
        if abs(m) < 1e-15:
            return self.inertia_matrix
        c = self.center_of_mass
        # Parallel axis theorem (reverse): I_com = I_o - m*(|c|^2 I - c c^T)
        return self.inertia_matrix - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "phi": self.phi.tolist(),
            "condition_number": self.condition_number,
            "residual_norm": self.residual_norm,
            "n_samples": self.n_samples,
            "mass": self.mass,
            "center_of_mass": self.center_of_mass.tolist(),
            "inertia_matrix": self.inertia_matrix.tolist(),
            "inertia_at_com": self.inertia_at_com.tolist(),
        }

    def __str__(self) -> str:
        c = self.center_of_mass
        I_com = self.inertia_at_com
        lines = [
            "EstimationResult:",
            f"  mass          = {self.mass:.6f}",
            f"  CoM           = [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]",
            f"  I_com diag    = [{I_com[0, 0]:.6e}, {I_com[1, 1]:.6e}, {I_com[2, 2]:.6e}]",
            f"  cond. number  = {self.condition_number:.2f}",
            f"  residual norm = {self.residual_norm:.6e}",
            f"  n_samples     = {self.n_samples}",
        ]
        return "\n".join(lines)
