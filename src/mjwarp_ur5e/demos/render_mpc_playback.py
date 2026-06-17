"""Render MPC identification trajectory as a 4-view grid video.

Loads the joint trajectory saved by run_mpc_identification.py (.npz)
and replays it through MuJoCo, capturing 4 camera views per frame.

Usage:
    python -m mjwarp_ur5e.demos.render_mpc_playback
    python -m mjwarp_ur5e.demos.render_mpc_playback \
        --trajectory results/mpc_identification_result.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from mjwarp_ur5e.cli import load_config
from mjwarp_ur5e.cli.configs import ModelConfig
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.rendering import add_ee_frame_overlay


@dataclass(slots=True)
class RenderMPCConfig(ModelConfig):
    """Config for MPC playback rendering."""

    trajectory: str = "results/mpc_identification_result.npz"
    output: str = "results/mpc_playback.mp4"
    width: int = 640
    height: int = 480
    fps_video: int = 30
    playback_speed: float = 1.0
    cameras: tuple[str, ...] = ("", "view_x", "view_y", "view_z")
    show_ee_frame: bool = True
    axis_length: float = 0.15


def _render_view(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | None,
    show_ee_frame: bool,
    axis_length: float,
) -> np.ndarray:
    if camera:
        renderer.update_scene(data, camera=camera)
    else:
        renderer.update_scene(data)
    if show_ee_frame:
        add_ee_frame_overlay(model, data, renderer.scene, axis_length)
    return renderer.render().copy()


def main() -> None:
    config = load_config(RenderMPCConfig)

    traj_path = Path(config.trajectory)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    data_file = np.load(str(traj_path))
    q_all = data_file["q"]
    traj_fps = float(data_file["fps"])
    n_frames_traj = len(q_all)
    duration = n_frames_traj / traj_fps

    print(f"Trajectory: {n_frames_traj} frames, {duration:.1f}s @ {traj_fps:.0f} fps")

    loaded = load_and_reset(config.model)
    model, data = loaded.model, loaded.data
    n_joints = q_all.shape[1]

    tile_w = min(config.width, int(model.vis.global_.offwidth))
    tile_h = min(config.height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)

    cameras: list[str | None] = [(c if c else None) for c in config.cameras[:4]]
    while len(cameras) < 4:
        cameras.append(None)
    cam_labels = [c or "default" for c in cameras]
    print(f"Cameras: {cam_labels}, tile={tile_w}x{tile_h}")

    video_duration = duration / config.playback_speed
    n_video_frames = max(1, int(video_duration * config.fps_video))
    traj_indices = np.linspace(0, n_frames_traj - 1, n_video_frames, dtype=int)

    print(
        f"Rendering {n_video_frames} frames at {config.fps_video} fps "
        f"({video_duration:.1f}s video, {config.playback_speed}x speed)"
    )

    frames: list[np.ndarray] = []
    for frame_idx, traj_idx in enumerate(traj_indices):
        data.qpos[:n_joints] = q_all[traj_idx]
        mujoco.mj_forward(model, data)

        tiles = [
            _render_view(renderer, model, data, cam, config.show_ee_frame, config.axis_length)
            for cam in cameras
        ]
        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        frames.append(np.concatenate([top, bottom], axis=0))

        if (frame_idx + 1) % 100 == 0 or frame_idx == n_video_frames - 1:
            t = traj_idx / traj_fps
            print(f"  frame {frame_idx + 1}/{n_video_frames} (t={t:.2f}s)")

    renderer.close()

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.stack(frames)
    iio.imwrite(str(output_path), video_array, fps=config.fps_video, macro_block_size=1)
    print(f"\nVideo saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
