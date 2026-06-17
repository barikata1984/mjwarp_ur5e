from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mjwarp_ur5e.identification.constraints import JointLimits

_UR5E_HOME = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


@dataclass
class HorizonConfig:
    """Planning horizon parameters."""

    duration: float = 3.0  # T_h [s]
    num_segments: int = 4  # quintic spline segments
    fps: float = 100.0  # trajectory sampling rate
    subsample_factor: int = 10  # regressor subsampling for cost eval


@dataclass
class PlannerConfig:
    """Single-horizon optimization parameters."""

    n_restarts: int = 8  # multi-start count
    max_iter_per_start: int = 100  # SLSQP iterations per start
    ftol: float = 1e-6
    method: str = "SLSQP"
    seed: int = 42
    waypoint_perturbation: float = 0.3  # random init scale [rad]


@dataclass
class MPCConfig:
    """Top-level MPC loop configuration."""

    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    # MPC loop
    replan_period: float = 1.5  # T_replan [s]: execute before replanning
    max_mpc_steps: int = 10
    convergence_threshold: float = 0.01  # relative cond improvement

    # Robot
    num_joints: int = 6
    q0: np.ndarray | None = None  # initial position (None -> _UR5E_HOME)
    joint_limits: JointLimits = field(default_factory=JointLimits)

    # MuJoCo identifiers
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"
    model_path: str = "assets/ur5e/mjcf/scene_with_box.xml"

    # Execution
    use_pd_control: bool = False
    noise_std_wrench: float = 0.0

    def __post_init__(self) -> None:
        if self.q0 is None:
            self.q0 = _UR5E_HOME.copy()
        if self.replan_period > self.horizon.duration:
            raise ValueError(
                f"replan_period ({self.replan_period}) must be "
                f"<= horizon.duration ({self.horizon.duration})"
            )
