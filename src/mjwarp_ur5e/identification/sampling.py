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


def _sample_site_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    site_name: str,
) -> BodyKinematics:
    """Kinematics about a force/torque sensor site, consistent with cfrc_int.

    Reading mj_objectVelocity/Acceleration on the SITE (rather than the body)
    returns quantities in the site frame about the site origin -- the same frame
    and reference point the FT sensor uses. cacc is populated via mj_inverse
    (which preserves the commanded qacc), so mj_objectAcceleration returns the
    full proper acceleration and no manual gravity term is needed.
    """
    site_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id is None:
        raise ValueError(f"Unknown site name: {site_name}")

    # Populate data.cacc while preserving the commanded qacc. mj_forward would
    # overwrite qacc with the forward-dynamics solution; mj_inverse keeps it and
    # fills cacc via mj_rnePostConstraint, so mj_objectAcceleration returns the
    # full proper acceleration (gravity included).
    mujoco.mj_inverse(model, data)

    rotation = np.array(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 1)
    acceleration = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, acceleration, 1)

    angular_velocity = velocity[:3]
    linear_velocity = velocity[3:]
    # mj_objectAcceleration returns the classical acceleration of the site origin;
    # the Newton-Euler regressor expects the spatial acceleration, which differs by
    # the transport term -omega x v.
    linear_acceleration = acceleration[3:] - np.cross(angular_velocity, linear_velocity)

    return BodyKinematics(
        body_name=body_name,
        rotation_body_to_world=rotation,
        angular_velocity_body=angular_velocity,
        linear_velocity_body=linear_velocity,
        angular_acceleration_body=acceleration[:3],
        linear_acceleration_body=linear_acceleration,
        # cacc already includes gravity (proper acceleration), so no manual term.
        gravity_body=np.zeros(3, dtype=np.float64),
    )


def sample_body_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    site_name: str | None = None,
) -> BodyKinematics:
    """Body kinematics about a force/torque sensor site.

    The regressor is always evaluated about the FT sensor site so it matches the
    sensor's interaction force (cfrc_int). ``site_name`` defaults to ``ft_sensor``;
    a model without that site raises ValueError (FT-less models are unsupported).
    """
    return _sample_site_kinematics(model, data, body_name, site_name or "ft_sensor")


def trajectory_subsample_indices(num_samples: int, subsample_factor: int) -> range:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if subsample_factor <= 0:
        raise ValueError("subsample_factor must be positive")
    return range(0, num_samples, subsample_factor)
