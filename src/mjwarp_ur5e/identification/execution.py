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
    # Seconds to hold the initial target before recording so the arm settles
    # into its gravity-loaded equilibrium (PD-servo mode only).
    settle_time: float = 1.0
    # Names of the MuJoCo force/torque sensors at the tool0 site. When both are
    # present in the model, the recorded wrench is read from these sensors
    # (interaction force/torque in the site frame) instead of being computed
    # analytically from the rigid-body regressor.
    force_sensor_name: str = "ft_force"
    torque_sensor_name: str = "ft_torque"


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

        params = body_inertial_parameters_from_model(model, cfg.body_name)

        # Resolve the tool0 force/torque sensors. When both exist, the wrench is
        # read directly from the simulator's interaction force/torque (matching a
        # physical FT sensor) rather than computed from the regressor.
        force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, cfg.force_sensor_name)
        torque_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, cfg.torque_sensor_name)
        use_ft_sensor = force_sid >= 0 and torque_sid >= 0
        if use_ft_sensor:
            force_adr = int(model.sensor_adr[force_sid])
            torque_adr = int(model.sensor_adr[torque_sid])

        buffer = DataBuffer()
        n_steps = len(trajectory.time)

        # Number of physics substeps per trajectory step so that one trajectory
        # interval (its dt) is fully integrated by the simulator. The MJCF
        # actuators are position-velocity servos, so ctrl receives target angles
        # and the built-in servo provides the tracking torque.
        if n_steps > 1:
            traj_dt = float(trajectory.time[1] - trajectory.time[0])
            n_substeps = max(1, round(traj_dt / model.opt.timestep))
        else:
            n_substeps = 1

        # Settle the arm into its gravity-loaded equilibrium before recording:
        # hold the initial target on the servos and integrate so the transient
        # from the static (qacc=0) reset decays. Without this, the first frames
        # carry a large acceleration spike unrelated to the desired trajectory.
        if cfg.use_pd_control and cfg.settle_time > 0.0:
            q_start = trajectory.position[0]
            data.ctrl[:n_joints] = q_start
            n_settle = max(1, round(cfg.settle_time / model.opt.timestep))
            for _ in range(n_settle):
                mujoco.mj_step(model, data)

        for i in range(n_steps):
            t = float(trajectory.time[i])
            q_des = trajectory.position[i]
            dq_des = trajectory.velocity[i]
            ddq_des = trajectory.acceleration[i]

            if cfg.use_pd_control:
                # Servo-tracking mode: feed the target angle to the built-in
                # position-velocity servos and integrate the full trajectory dt.
                data.ctrl[:n_joints] = q_des
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                q_meas = np.array(data.qpos[:n_joints], dtype=np.float64)
                dq_meas = np.array(data.qvel[:n_joints], dtype=np.float64)
                ddq_meas = np.array(data.qacc[:n_joints], dtype=np.float64)
            else:
                # Open-loop mode: set state directly. mj_forward is needed so the
                # FT sensor (an interaction force) gets populated.
                set_model_state(model, data, q_des, dq_des, ddq_des)
                if use_ft_sensor:
                    mujoco.mj_forward(model, data)
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

            if use_ft_sensor:
                # MuJoCo FT sensor: force/torque in the site (tool0) frame,
                # ordered [Fx, Fy, Fz, Mx, My, Mz]. The MuJoCo convention reports
                # the constraint force the parent exerts on the child (the
                # supporting reaction). Negate it so the recorded wrench is the
                # load the payload exerts on the flange, matching a physical FT
                # sensor (child -> parent).
                force = np.array(data.sensordata[force_adr : force_adr + 3], dtype=np.float64)
                torque = np.array(data.sensordata[torque_adr : torque_adr + 3], dtype=np.float64)
                wrench = -np.concatenate((force, torque))
            else:
                # Fallback: analytic rigid-body regressor wrench ([torque; force]).
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
