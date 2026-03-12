"""Trajectory playback and measurement on MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjwarp_ur5e.frames import get_site_frame
from mjwarp_ur5e.trajectories.base import TrajectorySample

from .data_buffer import DataBuffer, SensorSample
from .regressor import (
    body_inertial_parameters_from_model,
    compute_wrench_from_parameters,
    sample_body_regressor,
)
from .sampling import set_model_state

_DEFAULT_KP = 1000.0
_DEFAULT_KD = 100.0


@dataclass
class PlaybackConfig:
    """Configuration for trajectory playback."""

    dt: float = 0.002
    use_pd_control: bool = False
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    noise_std_q: float = 0.0
    noise_std_dq: float = 0.0
    noise_std_wrench: float = 0.0
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"


class TrajectoryPlayback:
    """Execute a trajectory on a MuJoCo model and collect data."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PlaybackConfig,
    ) -> None:
        self._model = model
        self._data = data
        self._config = config

    def execute(
        self,
        trajectory: TrajectorySample,
        rng: np.random.Generator | None = None,
    ) -> DataBuffer:
        """Play back a trajectory and collect sensor data.

        Args:
            trajectory: Desired trajectory with time, position,
                velocity, and acceleration arrays.
            rng: Random generator for measurement noise.

        Returns:
            DataBuffer with one SensorSample per timestep.
        """
        cfg = self._config
        model = self._model
        data = self._data
        n_joints = trajectory.position.shape[1]

        kp = cfg.kp if cfg.kp is not None else np.full(n_joints, _DEFAULT_KP)
        kd = cfg.kd if cfg.kd is not None else np.full(n_joints, _DEFAULT_KD)

        params = body_inertial_parameters_from_model(model, cfg.body_name)

        buffer = DataBuffer()
        n_steps = len(trajectory.time)

        for i in range(n_steps):
            t = float(trajectory.time[i])
            q_des = trajectory.position[i]
            dq_des = trajectory.velocity[i]
            ddq_des = trajectory.acceleration[i]

            if cfg.use_pd_control:
                # PD control mode: compute torque and step
                q_err = q_des - data.qpos[:n_joints]
                dq_err = dq_des - data.qvel[:n_joints]
                tau = kp * q_err + kd * dq_err
                data.ctrl[:n_joints] = tau
                mujoco.mj_step(model, data)

                q_meas = np.array(data.qpos[:n_joints], dtype=np.float64)
                dq_meas = np.array(data.qvel[:n_joints], dtype=np.float64)
                ddq_meas = ddq_des.copy()
            else:
                # Open-loop mode: set state directly
                set_model_state(model, data, q_des, dq_des, ddq_des)
                q_meas = q_des.copy()
                dq_meas = dq_des.copy()
                ddq_meas = ddq_des.copy()

            # Get EE pose
            frame = get_site_frame(model, data, cfg.site_name)
            if frame is not None:
                ee_pos = np.array(frame.position, dtype=np.float64)
                ee_rot = np.array(frame.rotation, dtype=np.float64)
            else:
                ee_pos = np.zeros(3, dtype=np.float64)
                ee_rot = np.eye(3, dtype=np.float64)

            # Compute wrench via regressor
            reg_sample = sample_body_regressor(model, data, cfg.body_name)
            wrench = compute_wrench_from_parameters(reg_sample.regressor, params)

            # Add measurement noise
            if rng is not None:
                if cfg.noise_std_q > 0:
                    q_meas = q_meas + rng.normal(0, cfg.noise_std_q, n_joints)
                if cfg.noise_std_dq > 0:
                    dq_meas = dq_meas + rng.normal(0, cfg.noise_std_dq, n_joints)
                if cfg.noise_std_wrench > 0:
                    wrench = wrench + rng.normal(0, cfg.noise_std_wrench, 6)

            sample = SensorSample(
                timestamp=t,
                q=q_meas,
                dq=dq_meas,
                ddq=ddq_meas,
                ee_position=ee_pos,
                ee_rotation=ee_rot,
                wrench=wrench,
            )
            buffer.append(sample)

        return buffer

    def compute_tracking_error(
        self,
        trajectory: TrajectorySample,
        buffer: DataBuffer,
    ) -> dict[str, float]:
        """Compare desired vs actual positions/velocities.

        Returns dict with max_position_error,
        max_velocity_error, rms_position_error,
        rms_velocity_error.
        """
        arrays = buffer.to_arrays()
        q_actual = arrays["q"]
        dq_actual = arrays["dq"]

        n = min(len(trajectory.time), len(buffer))
        q_des = trajectory.position[:n]
        dq_des = trajectory.velocity[:n]
        q_act = q_actual[:n]
        dq_act = dq_actual[:n]

        pos_err = np.abs(q_des - q_act)
        vel_err = np.abs(dq_des - dq_act)

        return {
            "max_position_error": float(np.max(pos_err)),
            "max_velocity_error": float(np.max(vel_err)),
            "rms_position_error": float(np.sqrt(np.mean(pos_err**2))),
            "rms_velocity_error": float(np.sqrt(np.mean(vel_err**2))),
        }
