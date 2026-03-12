"""Render an optimized excitation trajectory as a video on MuJoCo."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import tyro

from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.model import load_model, reset_to_home
from mjwarp_ur5e.rendering import add_ee_frame_overlay


@dataclasses.dataclass
class RenderPlaybackConfig:
    """Configuration for rendering excitation trajectory playback."""

    result_json: str = "debug/excitation_result.json"
    model: str = ""
    output: str = "debug/excitation_playback.mp4"
    width: int = 960
    height: int = 544
    camera: str | None = None
    fps_video: int = 30
    show_ee_frame: bool = True
    axis_length: float = 0.15
    playback_speed: float = 1.0


def main() -> None:
    config = tyro.cli(RenderPlaybackConfig)

    # Load optimization result
    print(f"Loading result from {config.result_json}")
    result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(result)
    traj_fps = result.config.fps
    duration = result.config.duration
    n_steps = len(trajectory.time)
    print(f"Trajectory: {n_steps} steps, {duration:.1f}s, {traj_fps:.0f} fps")

    # Load model
    model_path = config.model if config.model else "assets/ur5e/mjcf/scene_with_box.xml"
    loaded = load_model(model_path)
    reset_to_home(loaded.model, loaded.data)
    model, data = loaded.model, loaded.data

    # Setup renderer
    width = min(config.width, int(model.vis.global_.offwidth))
    height = min(config.height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Compute frame sampling: trajectory at traj_fps, video at fps_video
    video_fps = config.fps_video
    speed = config.playback_speed
    # Sample trajectory indices for each video frame
    video_duration = duration / speed
    n_video_frames = int(video_duration * video_fps)
    # Map video frame index -> trajectory step index
    traj_indices = np.linspace(0, n_steps - 1, n_video_frames, dtype=int)

    print(
        f"Rendering {n_video_frames} frames at {video_fps} fps "
        f"({video_duration:.1f}s video, {speed}x speed)"
    )

    frames: list[np.ndarray] = []
    n_joints = trajectory.position.shape[1]

    for frame_idx, traj_idx in enumerate(traj_indices):
        # Set joint positions from trajectory
        q = trajectory.position[traj_idx]
        data.qpos[:n_joints] = q
        mujoco.mj_forward(model, data)

        # Render
        if config.camera is None:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=config.camera)

        if config.show_ee_frame:
            add_ee_frame_overlay(model, data, renderer.scene, config.axis_length)

        image = renderer.render()
        frames.append(image.copy())

        if (frame_idx + 1) % 100 == 0 or frame_idx == n_video_frames - 1:
            t_traj = trajectory.time[traj_idx]
            print(f"  frame {frame_idx + 1}/{n_video_frames} (t={t_traj:.2f}s)")

    renderer.close()

    # Write video
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.stack(frames)
    iio.imwrite(
        str(output_path),
        video_array,
        fps=video_fps,
        macro_block_size=1,
    )
    print(f"\nVideo saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
