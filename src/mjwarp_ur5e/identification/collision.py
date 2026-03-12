from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mjwarp_ur5e.identification.constraints import _TrajectoryCache
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


@dataclass(frozen=True)
class CollisionConfig:
    ground_z_min: float = 0.01
    self_collision_min_dist: float = 0.02
    payload_half_extents: list[float] = field(default_factory=lambda: [0.05, 0.075, 0.10])
    payload_offset: list[float] = field(default_factory=lambda: [0, 0, 0.10])
    link_radii: list[float] = field(default_factory=lambda: [0.06, 0.05, 0.045, 0.035, 0.035, 0.03])
    safety_margin: float = 0.005


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

        # Pre-compute radii array once
        self._radii = np.array(self.config.link_radii, dtype=np.float64)

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

    def _get_payload_position(self) -> np.ndarray:
        """Get payload body position (3,). Call _run_kinematics first."""
        if self._payload_body_id is not None:
            return self.data.xpos[self._payload_body_id].copy()
        # Fall back: use wrist_3_link position
        wrist_id = self._link_body_ids[-1]
        return self.data.xpos[wrist_id].copy()

    def _check_ground_clearance(self, positions: np.ndarray) -> float:
        """Min z of all link positions minus ground_z_min."""
        return float(np.min(positions[:, 2]) - self.config.ground_z_min)

    def _check_self_collision(self, positions: np.ndarray) -> float:
        """Min distance between non-adjacent link pairs minus threshold."""
        min_clearance = float("inf")
        for i, j in _SELF_COLLISION_PAIRS:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            clearance = dist - self._radii[i] - self._radii[j]
            min_clearance = min(min_clearance, clearance)
        return min_clearance - self.config.self_collision_min_dist

    def _check_payload_collision(
        self,
        link_positions: np.ndarray,
        payload_pos: np.ndarray,
    ) -> float:
        """Min distance from payload to non-adjacent links minus safety."""
        min_clearance = float("inf")
        for i in range(len(self._link_body_ids) - 2):
            dist = float(np.linalg.norm(link_positions[i] - payload_pos))
            clearance = dist - self._radii[i]
            min_clearance = min(min_clearance, clearance)
        return min_clearance - self.config.safety_margin

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
            payload_pos = self._get_payload_position()
            clearances.append(self._check_payload_collision(link_pos, payload_pos))
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
