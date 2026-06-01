from __future__ import annotations

import mujoco
import numpy as np

from mjwarp_ur5e.model import get_named_object_id

from .sampling import sample_body_kinematics, set_model_state, trajectory_subsample_indices
from .types import BodyKinematics, InertialParameters, RegressorSample


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _quat_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _spatial_inertia_matrix_from_vector(parameter_vector: np.ndarray) -> np.ndarray:
    mass, hx, hy, hz, ixx, iyy, izz, ixy, ixz, iyz = np.asarray(
        parameter_vector,
        dtype=np.float64,
    )
    first_moments = np.array([hx, hy, hz], dtype=np.float64)
    inertia = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=np.float64,
    )
    spatial_inertia = np.zeros((6, 6), dtype=np.float64)
    spatial_inertia[:3, :3] = inertia
    spatial_inertia[:3, 3:] = _skew(first_moments)
    spatial_inertia[3:, :3] = -_skew(first_moments)
    spatial_inertia[3:, 3:] = mass * np.eye(3, dtype=np.float64)
    return spatial_inertia


def _force_cross_operator(spatial_velocity: np.ndarray) -> np.ndarray:
    angular = np.asarray(spatial_velocity[:3], dtype=np.float64)
    linear = np.asarray(spatial_velocity[3:], dtype=np.float64)
    operator = np.zeros((6, 6), dtype=np.float64)
    operator[:3, :3] = _skew(angular)
    operator[:3, 3:] = _skew(linear)
    operator[3:, 3:] = _skew(angular)
    return operator


def rigid_body_wrench_regressor(kinematics: BodyKinematics) -> np.ndarray:
    spatial_velocity = kinematics.spatial_velocity_body
    spatial_acceleration = kinematics.spatial_acceleration_body

    basis_vectors = np.eye(10, dtype=np.float64)
    cross_force = _force_cross_operator(spatial_velocity)
    regressor = np.zeros((6, 10), dtype=np.float64)

    for column_index, basis_vector in enumerate(basis_vectors):
        spatial_inertia = _spatial_inertia_matrix_from_vector(basis_vector)
        spatial_momentum = spatial_inertia @ spatial_velocity
        regressor[:, column_index] = (
            spatial_inertia @ spatial_acceleration + cross_force @ spatial_momentum
        )

    return regressor


def sample_body_regressor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    site_name: str | None = None,
) -> RegressorSample:
    kinematics = sample_body_kinematics(model, data, body_name, site_name)
    regressor = rigid_body_wrench_regressor(kinematics)
    return RegressorSample(body_name=body_name, regressor=regressor, kinematics=kinematics)


def _single_body_inertia_about_own_origin(
    model: mujoco.MjModel,
    body_id: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mass, com, inertia) of one body about its own frame origin."""
    mass = float(model.body_mass[body_id])
    com = np.array(model.body_ipos[body_id], dtype=np.float64)

    inertia_diag = np.array(model.body_inertia[body_id], dtype=np.float64)
    inertial_rotation = _quat_to_rotation_matrix(
        np.array(model.body_iquat[body_id], dtype=np.float64)
    )
    inertia_com = inertial_rotation @ np.diag(inertia_diag) @ inertial_rotation.T

    parallel_axis = mass * ((com @ com) * np.eye(3, dtype=np.float64) - np.outer(com, com))
    inertia_origin = inertia_com + parallel_axis
    return mass, com, inertia_origin


def body_inertial_parameters_from_model(
    model: mujoco.MjModel,
    body_name: str,
) -> InertialParameters:
    """Combined inertial parameters of a body and all its descendants.

    The assembly is rigidly attached (no joints between the named body and its
    children), so all child masses are aggregated into the named body's frame,
    expressed about that frame's origin. For a body with no children this reduces
    to the single-body case.
    """
    root_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root_id is None:
        raise ValueError(f"Unknown body name: {body_name}")

    # Collect the subtree rooted at root_id (root + all descendants).
    subtree = [root_id]
    for bid in range(model.nbody):
        parent = bid
        while parent != 0:
            parent = int(model.body_parentid[parent])
            if parent == root_id:
                subtree.append(bid)
                break

    total_mass = 0.0
    total_first_moments = np.zeros(3, dtype=np.float64)
    total_inertia = np.zeros((3, 3), dtype=np.float64)

    for bid in subtree:
        mass, com_local, inertia_local_origin = _single_body_inertia_about_own_origin(model, bid)
        # Pose of body `bid` relative to the root frame.
        rel_rot, rel_pos = _relative_pose(model, root_id, bid)

        # COM and inertia of this body expressed in the root frame, about root origin.
        com_root = rel_pos + rel_rot @ com_local
        # Inertia about the body's own origin, rotated into root orientation.
        inertia_about_body_origin_root = rel_rot @ inertia_local_origin @ rel_rot.T
        # Shift reference point from body origin (at rel_pos in root frame) to root origin.
        # I_root_origin = I_body_origin_in_root + m * (shift parallel-axis from body origin to root origin)
        # Using first-moment form to stay valid for off-origin COM:
        #   I about root origin = I about body origin (in root frame)
        #     + m[(d·d)E - d⊗d] evaluated with d measured between the two reference
        #       points, but the cleanest route is via the COM. Recompute about COM then shift.
        # Inertia about this body's COM (frame-invariant under translation):
        d_body = rel_rot @ com_local  # body origin -> COM, in root frame
        inertia_about_com = inertia_about_body_origin_root - mass * (
            (d_body @ d_body) * np.eye(3) - np.outer(d_body, d_body)
        )
        # Shift from COM to root origin:
        inertia_about_root = inertia_about_com + mass * (
            (com_root @ com_root) * np.eye(3) - np.outer(com_root, com_root)
        )

        total_mass += mass
        total_first_moments += mass * com_root
        total_inertia += inertia_about_root

    return InertialParameters(
        mass=total_mass,
        first_moments=total_first_moments,
        inertia_matrix=total_inertia,
    )


def _relative_pose(
    model: mujoco.MjModel,
    root_id: int,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose (rotation, position) of `body_id` expressed in `root_id`'s frame.

    Walks the kinematic tree from body_id up to root_id, composing the fixed
    body_pos/body_quat transforms. Assumes body_id is in the subtree of root_id.
    """
    rot = np.eye(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    bid = body_id
    while bid != root_id:
        local_rot = _quat_to_rotation_matrix(np.array(model.body_quat[bid], dtype=np.float64))
        local_pos = np.array(model.body_pos[bid], dtype=np.float64)
        # Compose: parent_T_body = local; accumulate child->root.
        rot = local_rot @ rot
        pos = local_pos + local_rot @ pos
        bid = int(model.body_parentid[bid])
    return rot, pos


def compute_wrench_from_parameters(
    regressor: np.ndarray,
    parameters: InertialParameters | np.ndarray,
) -> np.ndarray:
    if isinstance(parameters, InertialParameters):
        parameter_vector = parameters.to_vector()
    else:
        parameter_vector = np.asarray(parameters, dtype=np.float64)

    if parameter_vector.shape != (10,):
        raise ValueError("parameter_vector must have shape (10,)")
    return regressor @ parameter_vector


def compute_stacked_body_regressor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    body_name: str,
    subsample_factor: int = 1,
    with_ft_offset: bool = False,
    site_name: str | None = None,
) -> np.ndarray:
    q_array = np.asarray(q, dtype=np.float64)
    dq_array = np.asarray(dq, dtype=np.float64)
    ddq_array = np.asarray(ddq, dtype=np.float64)

    if q_array.ndim != 2 or q_array.shape[1] != model.nq:
        raise ValueError(f"q must have shape (N, {model.nq})")
    if dq_array.shape != (q_array.shape[0], model.nv):
        raise ValueError(f"dq must have shape ({q_array.shape[0]}, {model.nv})")
    if ddq_array.shape != (q_array.shape[0], model.nv):
        raise ValueError(f"ddq must have shape ({q_array.shape[0]}, {model.nv})")

    rows: list[np.ndarray] = []
    for index in trajectory_subsample_indices(q_array.shape[0], subsample_factor):
        set_model_state(model, data, q_array[index], dq_array[index], ddq_array[index])
        rows.append(sample_body_regressor(model, data, body_name, site_name).regressor)

    stacked = np.vstack(rows)

    if with_ft_offset:
        n_samples = len(rows)
        identity_block = np.tile(np.eye(6, dtype=np.float64), (n_samples, 1))
        stacked = np.hstack([identity_block, stacked])

    return stacked


def compute_condition_number(
    regressor: np.ndarray,
    singular_value_floor: float = 1e-12,
    column_scale: bool = False,
) -> float:
    matrix = np.asarray(regressor, dtype=np.float64)
    if column_scale:
        norms = np.linalg.norm(matrix, axis=0)
        norms = np.maximum(norms, 1e-30)
        matrix = matrix / norms
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] < singular_value_floor:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])
