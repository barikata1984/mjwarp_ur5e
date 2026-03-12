from __future__ import annotations

import mujoco
import numpy as np

from mjwarp_ur5e.model import get_named_object_id

from .types import BodyKinematics


def set_model_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    qacc: np.ndarray | None = None,
) -> None:
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.shape != (model.nq,):
        raise ValueError(f"qpos must have shape ({model.nq},)")

    data.qpos[:] = qpos_array

    if qvel is None:
        data.qvel[:] = 0.0
    else:
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qvel_array.shape != (model.nv,):
            raise ValueError(f"qvel must have shape ({model.nv},)")
        data.qvel[:] = qvel_array

    if qacc is None:
        data.qacc[:] = 0.0
    else:
        qacc_array = np.asarray(qacc, dtype=np.float64)
        if qacc_array.shape != (model.nv,):
            raise ValueError(f"qacc must have shape ({model.nv},)")
        data.qacc[:] = qacc_array

    # Compute position and velocity kinematics without solving forward dynamics.
    # mj_forward would overwrite qacc with values from the equations of motion
    # and leave cacc (body Cartesian accelerations) unpopulated, causing
    # sample_body_kinematics to return zero angular acceleration.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_fwdVelocity(model, data)


def _body_acceleration_from_qacc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the qacc-dependent body acceleration via the body Jacobian.

    MuJoCo's ``mj_objectAcceleration`` only returns the velocity-dependent
    (centripetal / Coriolis) part of the body acceleration because ``data.cacc``
    is not populated by the kinematic-only pipeline used in ``set_model_state``.
    This helper computes the missing term ``J @ qacc`` in the world frame and
    returns it rotated into the body frame.

    Returns (angular_acceleration_body, linear_acceleration_body) from qacc.
    """
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    alpha_world = jacr @ data.qacc
    a_world = jacp @ data.qacc

    rotation = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return rotation.T @ alpha_world, rotation.T @ a_world


def sample_body_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> BodyKinematics:
    body_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id is None:
        raise ValueError(f"Unknown body name: {body_name}")

    rotation = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)

    velocity_body = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity_body,
        1,
    )

    # mj_objectAcceleration returns only the velocity-dependent part
    # (centripetal / Coriolis) because data.cacc is not populated by the
    # kinematic-only pipeline.  We add the qacc-dependent part via J @ qacc.
    accel_vel_body = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectAcceleration(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        accel_vel_body,
        1,
    )
    alpha_qacc_body, a_qacc_body = _body_acceleration_from_qacc(model, data, body_id)

    gravity_world = np.array(model.opt.gravity, dtype=np.float64)
    gravity_body = rotation.T @ gravity_world

    return BodyKinematics(
        body_name=body_name,
        rotation_body_to_world=rotation,
        angular_velocity_body=velocity_body[:3],
        linear_velocity_body=velocity_body[3:],
        angular_acceleration_body=accel_vel_body[:3] + alpha_qacc_body,
        linear_acceleration_body=accel_vel_body[3:] + a_qacc_body,
        gravity_body=gravity_body,
    )


def trajectory_subsample_indices(num_samples: int, subsample_factor: int) -> range:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if subsample_factor <= 0:
        raise ValueError("subsample_factor must be positive")
    return range(0, num_samples, subsample_factor)
