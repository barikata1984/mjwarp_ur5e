"""Replay NPZ joint trajectory on the UR5e+FT300s+2F-85 grasping scene.

Outputs:
  1. 4-view grid video (overview, front, top, side) at 30 fps
  2. Simulated FT wrench via mj_inverse (saved to NPZ)
  3. Comparison plot: sim FT vs recorded wrench_actual
"""

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

CUBE_BODY_XML = """\
                          <body name="cube" pos="0 0 0.145">
                            <inertial mass="0.3375" pos="0 0 0"
                              diaginertia="1.40625e-04 1.40625e-04 1.40625e-04"/>
                            <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
                              rgba="0.85 0.32 0.1 1" friction="1.0 0.005 0.001"/>
                          </body>
"""

PINCH_SITE_TAG = (
    '<site name="pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>'
)

CAMERAS = ["overview", "front", "top", "side"]
LABELS = ["Overview", "Front", "Top", "Side"]
VIDEO_FPS = 30
VIEW_W, VIEW_H = 480, 360


def _load_grasping_model(scene_xml: str) -> mujoco.MjModel:
    mjcf_dir = Path(scene_xml).parent
    robot_path = mjcf_dir / "ur5e_with_ft300s_and_gripper.xml"
    robot_xml = robot_path.read_text()
    robot_xml = robot_xml.replace(PINCH_SITE_TAG, PINCH_SITE_TAG + "\n" + CUBE_BODY_XML)

    tmp_robot = mjcf_dir / "_tmp_robot_with_cube.xml"
    tmp_robot.write_text(robot_xml)

    scene_text = (
        Path(scene_xml)
        .read_text()
        .replace(
            'file="ur5e_with_ft300s_and_gripper.xml"',
            f'file="{tmp_robot.name}"',
        )
    )
    tmp_scene = mjcf_dir / "_tmp_scene_replay.xml"
    tmp_scene.write_text(scene_text)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_scene))
    finally:
        tmp_robot.unlink(missing_ok=True)
        tmp_scene.unlink(missing_ok=True)
    return model


def _close_gripper(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Close the gripper around the cube and return the settled gripper qpos."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ur_home = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    for i, c in enumerate(ur_home):
        data.ctrl[i] = c
    fingers_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    data.ctrl[fingers_id] = 255

    saved_grav = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    for _ in range(3000):
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = saved_grav

    return data.qpos[6:].copy()


def _compute_ft(
    model: mujoco.MjModel,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Compute FT wrench for each timestep via mj_inverse."""
    data = mujoco.MjData(model)
    fsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
    tsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
    fadr = int(model.sensor_adr[fsid])
    tadr = int(model.sensor_adr[tsid])

    n = len(q)
    wrench = np.zeros((n, 6))
    nv_gripper = len(gripper_qpos)

    for i in range(n):
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = q[i]
        data.qpos[6 : 6 + nv_gripper] = gripper_qpos
        data.qvel[:6] = dq[i]
        data.qacc[:6] = ddq[i]
        mujoco.mj_inverse(model, data)
        f = data.sensordata[fadr : fadr + 3]
        t = data.sensordata[tadr : tadr + 3]
        wrench[i] = -np.concatenate((f, t))

    return wrench


def _render_grid(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    time_s: float,
) -> np.ndarray:
    """Render 4 camera views and compose into a 2x2 grid."""
    views = []
    for cam_name in CAMERAS:
        renderer.update_scene(data, camera=cam_name)
        views.append(renderer.render().copy())

    top_row = np.concatenate([views[0], views[1]], axis=1)
    bot_row = np.concatenate([views[2], views[3]], axis=1)
    grid = np.concatenate([top_row, bot_row], axis=0)
    return grid


def _numerical_ddq(dq: np.ndarray, dt: np.ndarray) -> np.ndarray:
    ddq = np.diff(dq, axis=0) / dt[:, None]
    return np.vstack([ddq, ddq[-1:]])


def main() -> None:
    npz_dir = sorted(Path("data").glob("replay_recording_*"))[-1]
    npz_path = npz_dir / "recording.npz"
    scene_xml = "assets/ur5e/mjcf/scene_with_ft300s_and_gripper.xml"

    print(f"Loading trajectory data from {npz_path}...")
    d = np.load(str(npz_path), allow_pickle=True)
    time_raw = d["time"]
    time_arr = time_raw - time_raw[0]
    q = d["joint_position"]
    dq = d["joint_velocity"]
    wrench_actual = d["wrench"]

    dt = np.diff(time_arr)
    ddq = _numerical_ddq(dq, dt)

    dt_median = float(np.median(dt))
    print(f"Trajectory: {len(time_arr)} samples, {time_arr[-1]:.2f}s, dt={dt_median * 1000:.1f}ms")

    print("Loading model and closing gripper...")
    model = _load_grasping_model(scene_xml)
    data = mujoco.MjData(model)
    gripper_qpos = _close_gripper(model, data)

    print("Computing sim FT via mj_inverse...")
    wrench_sim = _compute_ft(model, q, dq, ddq, gripper_qpos)
    wrench_sim_tared = wrench_sim - wrench_sim[0]
    wrench_actual_tared = wrench_actual - wrench_actual[0]

    out_dir = Path("results/replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "replay_ft.npz",
        time=time_arr,
        wrench_sim=wrench_sim_tared,
        wrench_actual=wrench_actual_tared,
        note="sim: grasping scene (UR5e+FT300s+2F85+5cm cube), child->parent, frame0-tared",
    )
    print(f"Saved FT data to {out_dir / 'replay_ft.npz'}")

    print("Rendering 4-view video...")
    renderer = mujoco.Renderer(model, height=VIEW_H, width=VIEW_W)

    dt_video = 1.0 / VIDEO_FPS
    frame_indices = np.searchsorted(time_arr, np.arange(0, time_arr[-1], dt_video))
    frame_indices = np.clip(frame_indices, 0, len(time_arr) - 1)

    video_path = out_dir / "replay_4view.mp4"
    grid_w = VIEW_W * 2
    grid_h = VIEW_H * 2
    grid_w = (grid_w + 15) // 16 * 16
    grid_h = (grid_h + 15) // 16 * 16

    writer = imageio.get_writer(str(video_path), fps=VIDEO_FPS, macro_block_size=1, quality=8)

    for frame_no, idx in enumerate(frame_indices):
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = q[idx]
        data.qpos[6 : 6 + len(gripper_qpos)] = gripper_qpos
        data.qvel[:6] = dq[idx]
        mujoco.mj_forward(model, data)

        grid = _render_grid(renderer, data, time_arr[idx])
        writer.append_data(grid)

        if frame_no % 30 == 0:
            print(f"  frame {frame_no}/{len(frame_indices)}, t={time_arr[idx]:.2f}s")

    writer.close()
    renderer.close()
    print(f"Saved video to {video_path}")

    print("Generating FT comparison plot...")
    force_labels = ["Fx", "Fy", "Fz"]
    torque_labels = ["Mx", "My", "Mz"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle(
        "FT Comparison: Sim (grasping scene) vs Real",
        fontsize=14,
        fontweight="bold",
    )

    for col in range(3):
        ax_f = axes[0, col]
        ax_f.plot(time_arr, wrench_actual_tared[:, col], label="Real", alpha=0.8, linewidth=0.8)
        ax_f.plot(time_arr, wrench_sim_tared[:, col], label="Sim", alpha=0.8, linewidth=0.8)
        ax_f.set_title(force_labels[col], fontsize=12)
        ax_f.legend(fontsize=8)
        ax_f.grid(True, alpha=0.3)

        ax_t = axes[1, col]
        ax_t.plot(time_arr, wrench_actual_tared[:, col + 3], label="Real", alpha=0.8, linewidth=0.8)
        ax_t.plot(time_arr, wrench_sim_tared[:, col + 3], label="Sim", alpha=0.8, linewidth=0.8)
        ax_t.set_title(torque_labels[col], fontsize=12)
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlabel("Time [s]")

    f_all = np.concatenate([wrench_actual_tared[:, :3].ravel(), wrench_sim_tared[:, :3].ravel()])
    f_margin = (np.nanmax(f_all) - np.nanmin(f_all)) * 0.05
    f_ylim = (np.nanmin(f_all) - f_margin, np.nanmax(f_all) + f_margin)
    for ax in axes[0]:
        ax.set_ylim(f_ylim)
    axes[0, 0].set_ylabel("Force [N]")

    t_all = np.concatenate([wrench_actual_tared[:, 3:].ravel(), wrench_sim_tared[:, 3:].ravel()])
    t_margin = (np.nanmax(t_all) - np.nanmin(t_all)) * 0.05
    t_ylim = (np.nanmin(t_all) - t_margin, np.nanmax(t_all) + t_margin)
    for ax in axes[1]:
        ax.set_ylim(t_ylim)
    axes[1, 0].set_ylabel("Torque [Nm]")

    fig.tight_layout()

    plot_path = out_dir / "ft_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FT comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
