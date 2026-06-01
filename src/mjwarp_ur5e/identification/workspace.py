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


@dataclass(frozen=True)
class EeVelocityConfig:
    """End-effector linear velocity limit."""

    max_linear_velocity: float = 0.25  # m/s


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
    # Hoist static array conversions out of the hot loop
    lower = (
        np.asarray(workspace_config.box_lower, dtype=np.float64)
        if workspace_config.box_lower is not None
        else None
    )
    upper = (
        np.asarray(workspace_config.box_upper, dtype=np.float64)
        if workspace_config.box_upper is not None
        else None
    )
    margin = workspace_config.safety_margin

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        positions = _evaluate_workspace_positions(model, data, sample.position, site_name)
        return _compute_box_margin(positions, lower, upper, margin)

    return constraint


def _box_vertices(half_extents: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return 8 vertices of an axis-aligned box in local frame (8, 3)."""
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


def _box_surface_points(half_extents: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return 26 surface sample points of a box: 8 vertices + 12 edge midpoints + 6 face centers."""
    coords = np.array([-1.0, 0.0, 1.0])
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    # Remove the interior point (0, 0, 0)
    mask = np.any(grid != 0.0, axis=1)
    surface = grid[mask]  # (26, 3)
    return surface * half_extents + offset


def find_payload_constraint_geom(
    model: mujoco.MjModel,
    body_name: str,
    preferred_geom: str = "payload_box_red",
) -> int:
    """Return the geom id of the box used for payload workspace/collision constraints.

    Searches the subtree rooted at `body_name` (the body and all its descendants),
    so the geom may live on a nested child body. If a geom named `preferred_geom`
    exists in the subtree it is chosen; otherwise the first box geom is used.
    """
    root_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root_id is None:
        raise ValueError(f"Unknown body: {body_name}")

    def in_subtree(bid: int) -> bool:
        while bid != 0:
            if bid == root_id:
                return True
            bid = int(model.body_parentid[bid])
        return root_id == 0

    first_box: int | None = None
    for gi in range(model.ngeom):
        if model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if not in_subtree(int(model.geom_bodyid[gi])):
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) == preferred_geom:
            return gi
        if first_box is None:
            first_box = gi

    if first_box is None:
        raise ValueError(f"No box geom found in subtree of body: {body_name}")
    return first_box


def _evaluate_payload_surface_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str,
) -> np.ndarray:
    """Return payload geom surface sample positions (n_steps, 26, 3) in world frame.

    Samples 26 points on the box surface: 8 vertices, 12 edge midpoints, 6 face centers.
    The box geom is resolved from the subtree of `body_name` (preferring the red
    payload block) and transformed by its own parent body's world frame.
    """
    geom_id = find_payload_constraint_geom(model, body_name)
    geom_body_id = int(model.geom_bodyid[geom_id])

    half_extents = model.geom_size[geom_id].copy()
    geom_offset = model.geom_pos[geom_id].copy()
    local_points = _box_surface_points(half_extents, geom_offset)  # (26, 3)
    n_points = local_points.shape[0]

    n_steps = q_trajectory.shape[0]
    world_points = np.zeros((n_steps, n_points, 3), dtype=np.float64)

    for i in range(n_steps):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        body_pos = data.xpos[geom_body_id]
        body_rot = data.xmat[geom_body_id].reshape(3, 3)
        world_points[i] = (body_rot @ local_points.T).T + body_pos

    return world_points


def _compute_box_margin(
    points: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    safety_margin: float,
) -> float:
    """Compute minimum margin of points (N, 3) against optional box bounds."""
    margins: list[float] = []
    if lower is not None:
        margins.append(float(np.min(points - lower)) - safety_margin)
    if upper is not None:
        margins.append(float(np.min(upper - points)) - safety_margin)
    if not margins:
        return 0.0
    return float(np.min(margins))


def make_payload_workspace_constraint(
    cache: _TrajectoryCache,
    workspace_config: WorkspaceConstraintConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = "payload_box_mount",
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff all payload surface sample points stay within box bounds."""
    lower = (
        np.asarray(workspace_config.box_lower, dtype=np.float64)
        if workspace_config.box_lower is not None
        else None
    )
    upper = (
        np.asarray(workspace_config.box_upper, dtype=np.float64)
        if workspace_config.box_upper is not None
        else None
    )
    margin = workspace_config.safety_margin

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        surface_pts = _evaluate_payload_surface_points(model, data, sample.position, body_name)
        pts = surface_pts.reshape(-1, 3)
        return _compute_box_margin(pts, lower, upper, margin)

    return constraint


def _evaluate_ee_linear_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    dq_trajectory: np.ndarray,
    site_name: str = "attachment_site",
) -> np.ndarray:
    """Return EE linear speed (n_steps,) for each timestep."""
    site_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id is None:
        raise ValueError(f"Unknown site: {site_name}")

    nv = model.nv
    n_steps = q_trajectory.shape[0]
    speeds = np.zeros(n_steps, dtype=np.float64)
    jacp = np.zeros((3, nv), dtype=np.float64)

    for i in range(n_steps):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        jacp[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        linear_vel = jacp @ dq_trajectory[i]
        speeds[i] = np.linalg.norm(linear_vel)

    return speeds


def make_ee_velocity_constraint(
    cache: _TrajectoryCache,
    ee_velocity_config: EeVelocityConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "attachment_site",
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff EE linear speed stays within limit."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        speeds = _evaluate_ee_linear_velocity(
            model, data, sample.position, sample.velocity, site_name
        )
        return float(ee_velocity_config.max_linear_velocity - np.max(speeds))

    return constraint
