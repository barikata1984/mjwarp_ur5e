from __future__ import annotations

from dataclasses import dataclass

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

    mujoco.mj_forward(model, data)


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
    acceleration_body = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity_body,
        1,
    )
    mujoco.mj_objectAcceleration(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        acceleration_body,
        1,
    )

    gravity_world = np.array(model.opt.gravity, dtype=np.float64)
    gravity_body = rotation.T @ gravity_world

    return BodyKinematics(
        body_name=body_name,
        rotation_body_to_world=rotation,
        angular_velocity_body=velocity_body[:3],
        linear_velocity_body=velocity_body[3:],
        angular_acceleration_body=acceleration_body[:3],
        linear_acceleration_body=acceleration_body[3:],
        gravity_body=gravity_body,
    )


def trajectory_subsample_indices(num_samples: int, subsample_factor: int) -> range:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if subsample_factor <= 0:
        raise ValueError("subsample_factor must be positive")
    return range(0, num_samples, subsample_factor)