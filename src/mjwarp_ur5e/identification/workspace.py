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


def _box_vertices(half_extents: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return 8 vertices of an axis-aligned box in local frame (8, 3)."""
    hx, hy, hz = half_extents
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half_extents + offset


def _evaluate_payload_vertices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str,
) -> np.ndarray:
    """Return payload geom vertex positions (n_steps, 8, 3) in world frame."""
    body_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id is None:
        raise ValueError(f"Unknown body: {body_name}")

    # Find the first box geom attached to this body
    geom_id: int | None = None
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == body_id and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
            geom_id = gi
            break

    if geom_id is None:
        raise ValueError(f"No box geom found on body: {body_name}")

    half_extents = model.geom_size[geom_id].copy()
    geom_offset = model.geom_pos[geom_id].copy()
    local_verts = _box_vertices(half_extents, geom_offset)  # (8, 3)

    n_steps = q_trajectory.shape[0]
    world_verts = np.zeros((n_steps, 8, 3), dtype=np.float64)

    for i in range(n_steps):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)
        world_verts[i] = (body_rot @ local_verts.T).T + body_pos

    return world_verts


def make_payload_workspace_constraint(
    cache: _TrajectoryCache,
    workspace_config: WorkspaceConstraintConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = "payload_box_mount",
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff all payload geom vertices stay within box bounds."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        # (n_steps, 8, 3)
        verts = _evaluate_payload_vertices(model, data, sample.position, body_name)
        # Flatten to (n_steps*8, 3) for bounds check
        pts = verts.reshape(-1, 3)
        margins: list[float] = []
        if workspace_config.box_lower is not None:
            lower = np.asarray(workspace_config.box_lower)
            margins.append(float(np.min(pts - lower)) - workspace_config.safety_margin)
        if workspace_config.box_upper is not None:
            upper = np.asarray(workspace_config.box_upper)
            margins.append(float(np.min(upper - pts)) - workspace_config.safety_margin)
        if not margins:
            return 0.0
        return float(np.min(margins))

    return constraint
