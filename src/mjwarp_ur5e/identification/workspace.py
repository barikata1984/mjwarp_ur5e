from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from mjwarp_ur5e.identification.constraints import _TrajectoryCache
from mjwarp_ur5e.model import get_named_object_id


@dataclass(frozen=True)
class WorkspaceConstraintConfig:
    max_displacement: float = 0.5
    box_lower: np.ndarray | None = None
    box_upper: np.ndarray | None = None
    safety_margin: float = 0.01


def _evaluate_workspace_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    site_name: str = "attachment_site",
) -> np.ndarray:
    """Return EE positions (n_steps, 3) for each timestep."""
    site_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id is None:
        raise ValueError(f"Unknown site: {site_name}")

    n_steps = q_trajectory.shape[0]
    positions = np.zeros((n_steps, 3), dtype=np.float64)

    for i in range(n_steps):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        positions[i] = data.site_xpos[site_id].copy()

    return positions


def evaluate_workspace_displacement(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    site_name: str = "attachment_site",
) -> np.ndarray:
    """Compute EE displacement from initial position for each timestep."""
    positions = _evaluate_workspace_positions(model, data, q_trajectory, site_name)
    initial_pos = positions[0]
    return np.linalg.norm(positions - initial_pos, axis=1)


def make_workspace_constraint(
    cache: _TrajectoryCache,
    workspace_config: WorkspaceConstraintConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "attachment_site",
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff EE stays within max displacement."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        distances = evaluate_workspace_displacement(model, data, sample.position, site_name)
        return float(
            workspace_config.max_displacement - np.max(distances) - workspace_config.safety_margin
        )

    return constraint


def make_box_workspace_constraint(
    cache: _TrajectoryCache,
    workspace_config: WorkspaceConstraintConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "attachment_site",
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff EE stays within box bounds."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        positions = _evaluate_workspace_positions(model, data, sample.position, site_name)
        margins = []
        if workspace_config.box_lower is not None:
            lower = np.asarray(workspace_config.box_lower)
            margins.append(float(np.min(positions - lower)) - workspace_config.safety_margin)
        if workspace_config.box_upper is not None:
            upper = np.asarray(workspace_config.box_upper)
            margins.append(float(np.min(upper - positions)) - workspace_config.safety_margin)
        if not margins:
            return 0.0
        return float(np.min(margins))

    return constraint
