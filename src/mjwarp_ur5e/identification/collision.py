from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjwarp_ur5e.identification.constraints import _TrajectoryCache
from mjwarp_ur5e.identification.workspace import _box_vertices
from mjwarp_ur5e.model import get_named_object_id

# UR5e link body names (shoulder through wrist_3)
_UR5E_LINK_NAMES: list[str] = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]

# Non-adjacent link pairs for self-collision checking
_SELF_COLLISION_PAIRS: list[tuple[int, int]] = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 5),
]

# Default self-collision radii for UR5e (tuned for sphere-sphere check)
_UR5E_SELF_COLLISION_RADII: list[float] = [0.06, 0.05, 0.045, 0.035, 0.035, 0.03]


@dataclass
class _CapsuleLocal:
    """Capsule geometry in parent body local frame."""

    p1: np.ndarray  # (3,) endpoint 1
    p2: np.ndarray  # (3,) endpoint 2
    radius: float


def _extract_capsule_geom(model: mujoco.MjModel, geom_id: int) -> _CapsuleLocal:
    """Extract capsule/cylinder geom as _CapsuleLocal in body frame."""
    r = float(model.geom_size[geom_id, 0])
    half_len = float(model.geom_size[geom_id, 1])
    pos = model.geom_pos[geom_id].copy()
    quat = model.geom_quat[geom_id].copy()
    rot = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, quat)
    z_axis = rot.reshape(3, 3)[:, 2]
    return _CapsuleLocal(
        p1=pos + z_axis * half_len,
        p2=pos - z_axis * half_len,
        radius=r,
    )


def _extract_link_capsules(
    model: mujoco.MjModel, link_body_ids: list[int]
) -> list[list[_CapsuleLocal]]:
    """Extract capsule/cylinder geoms for each link body in body-local frame."""
    result: list[list[_CapsuleLocal]] = []
    for bid in link_body_ids:
        caps: list[_CapsuleLocal] = []
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] != bid:
                continue
            gtype = model.geom_type[gi]
            if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                caps.append(_extract_capsule_geom(model, gi))
        result.append(caps)
    return result


def _find_payload_box_geom(model: mujoco.MjModel, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (half_extents, geom_offset) of the first box geom on a body."""
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == body_id and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
            return model.geom_size[gi].copy(), model.geom_pos[gi].copy()
    raise ValueError(f"No box geom found on body {body_id}")


def _segment_aabb_distance(p1: np.ndarray, p2: np.ndarray, half_extents: np.ndarray) -> float:
    """Min distance from line segment [p1,p2] to AABB [-h,h] via alternating projection."""
    d = p2 - p1
    dd = np.dot(d, d)
    if dd < 1e-12:
        closest = np.clip(p1, -half_extents, half_extents)
        return float(np.linalg.norm(p1 - closest))
    t = 0.5
    for _ in range(8):
        pt = p1 + t * d
        closest_on_box = np.clip(pt, -half_extents, half_extents)
        t = float(np.clip(np.dot(closest_on_box - p1, d) / dd, 0.0, 1.0))
    pt = p1 + t * d
    closest_on_box = np.clip(pt, -half_extents, half_extents)
    return float(np.linalg.norm(pt - closest_on_box))


@dataclass(frozen=True)
class CollisionConfig:
    ground_z_min: float = 0.01
    self_collision_min_dist: float = 0.02
    self_collision_radii: list[float] = field(
        default_factory=lambda: list(_UR5E_SELF_COLLISION_RADII)
    )
    payload_half_extents: list[float] | None = None
    payload_offset: list[float] | None = None
    safety_margin: float = 0.05


class CollisionChecker:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: CollisionConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or CollisionConfig()

        self._link_body_ids: list[int] = []
        for name in _UR5E_LINK_NAMES:
            bid = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid is None:
                raise ValueError(f"Link body not found: {name}")
            self._link_body_ids.append(bid)

        payload_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_box_mount")
        self._payload_body_id = payload_id

        # Self-collision radii (sphere-sphere, tuned defaults for UR5e)
        self._self_radii = np.array(self.config.self_collision_radii, dtype=np.float64)

        # Extract capsule geoms per link for accurate box-capsule collision
        self._link_capsules = _extract_link_capsules(model, self._link_body_ids)

        # Payload box geometry: from config or auto-extract from model
        if self.config.payload_half_extents is not None and self.config.payload_offset is not None:
            self._payload_half_extents = np.array(
                self.config.payload_half_extents, dtype=np.float64
            )
            self._payload_offset = np.array(self.config.payload_offset, dtype=np.float64)
        elif payload_id is not None:
            he, off = _find_payload_box_geom(model, payload_id)
            self._payload_half_extents = he
            self._payload_offset = off
        else:
            self._payload_half_extents = np.zeros(3)
            self._payload_offset = np.zeros(3)

        self._payload_local_verts = _box_vertices(
            self._payload_half_extents, self._payload_offset
        )  # (8, 3)

    def _run_kinematics(self, q: np.ndarray) -> None:
        """Set qpos and run forward kinematics (once)."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _get_link_positions(self) -> np.ndarray:
        """Return link body positions (n_links, 3). Call _run_kinematics first."""
        positions = np.zeros((len(self._link_body_ids), 3), dtype=np.float64)
        for i, bid in enumerate(self._link_body_ids):
            positions[i] = self.data.xpos[bid].copy()
        return positions

    def _check_ground_clearance(self, positions: np.ndarray) -> float:
        """Min z of all link positions minus ground_z_min."""
        return float(np.min(positions[:, 2]) - self.config.ground_z_min)

    def _check_self_collision(self, positions: np.ndarray) -> float:
        """Min distance between non-adjacent link pairs minus threshold."""
        min_clearance = float("inf")
        for i, j in _SELF_COLLISION_PAIRS:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            clearance = dist - self._self_radii[i] - self._self_radii[j]
            min_clearance = min(min_clearance, clearance)
        return min_clearance - self.config.self_collision_min_dist

    @staticmethod
    def _box_sphere_clearance(
        box_center: np.ndarray,
        box_rot: np.ndarray,
        half_extents: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ) -> float:
        """Minimum distance from oriented box surface to sphere surface."""
        local = box_rot.T @ (sphere_center - box_center)
        closest = np.clip(local, -half_extents, half_extents)
        return float(np.linalg.norm(local - closest)) - sphere_radius

    @staticmethod
    def _box_capsule_clearance(
        box_center: np.ndarray,
        box_rot: np.ndarray,
        half_extents: np.ndarray,
        cap_p1_world: np.ndarray,
        cap_p2_world: np.ndarray,
        cap_radius: float,
    ) -> float:
        """Min distance from oriented box surface to capsule surface.

        Transforms capsule endpoints into box-local frame, computes
        segment-to-AABB distance, then subtracts capsule radius.
        """
        p1_local = box_rot.T @ (cap_p1_world - box_center)
        p2_local = box_rot.T @ (cap_p2_world - box_center)
        seg_dist = _segment_aabb_distance(p1_local, p2_local, half_extents)
        return seg_dist - cap_radius

    def _get_payload_box_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (box_center_world, box_rot_world). Call _run_kinematics first."""
        bid = self._payload_body_id
        body_pos = self.data.xpos[bid]
        body_rot = self.data.xmat[bid].reshape(3, 3)
        box_center = body_pos + body_rot @ self._payload_offset
        return box_center, body_rot

    def _check_payload_collision(
        self,
        box_center: np.ndarray,
        box_rot: np.ndarray,
    ) -> float:
        """Min surface distance from payload box to non-adjacent link capsules."""
        min_clearance = float("inf")
        # Check links 0..3 (shoulder through wrist_1, non-adjacent to payload)
        for i in range(len(self._link_body_ids) - 2):
            bid = self._link_body_ids[i]
            body_pos = self.data.xpos[bid]
            body_rot = self.data.xmat[bid].reshape(3, 3)
            for cap in self._link_capsules[i]:
                # Transform capsule endpoints to world frame
                p1_world = body_pos + body_rot @ cap.p1
                p2_world = body_pos + body_rot @ cap.p2
                clearance = self._box_capsule_clearance(
                    box_center,
                    box_rot,
                    self._payload_half_extents,
                    p1_world,
                    p2_world,
                    cap.radius,
                )
                min_clearance = min(min_clearance, clearance)
        return min_clearance - self.config.safety_margin

    def _check_payload_ground_clearance(self) -> float:
        """Min z of payload box vertices minus ground_z_min. Call _run_kinematics first."""
        bid = self._payload_body_id
        body_pos = self.data.xpos[bid]
        body_rot = self.data.xmat[bid].reshape(3, 3)
        world_verts = (body_rot @ self._payload_local_verts.T).T + body_pos
        return float(np.min(world_verts[:, 2]) - self.config.ground_z_min)

    def compute_min_clearance(self, q_trajectory: np.ndarray) -> float:
        """Return minimum clearance over full trajectory."""
        min_clearance = float("inf")
        for i in range(q_trajectory.shape[0]):
            c = self.check_single_config(q_trajectory[i])
            min_clearance = min(min_clearance, c)
        return min_clearance

    def check_single_config(self, q: np.ndarray) -> float:
        """Return minimum clearance for a single configuration."""
        self._run_kinematics(q)
        link_pos = self._get_link_positions()
        clearances = [
            self._check_ground_clearance(link_pos),
            self._check_self_collision(link_pos),
        ]
        if self._payload_body_id is not None:
            box_center, box_rot = self._get_payload_box_world()
            clearances.append(self._check_payload_collision(box_center, box_rot))
            clearances.append(self._check_payload_ground_clearance())
        return float(np.min(clearances))


def make_collision_constraint(
    cache: _TrajectoryCache,
    collision_checker: CollisionChecker,
) -> Callable[[np.ndarray], float]:
    """Return f(x)->float >= 0 iff trajectory is collision-free."""

    def constraint(x: np.ndarray) -> float:
        sample = cache.get(x)
        return collision_checker.compute_min_clearance(sample.position)

    return constraint
