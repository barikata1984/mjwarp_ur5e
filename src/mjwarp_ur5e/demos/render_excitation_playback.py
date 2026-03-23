"""Render an optimized excitation trajectory as a video on MuJoCo."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from mjwarp_ur5e.cli import RenderPlaybackConfig, load_config
from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.rendering import add_ee_frame_overlay


def _render_single_view(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | None,
    show_ee_frame: bool,
    axis_length: float,
) -> np.ndarray:
    """Render a single camera view and return the image array."""
    if camera:
        renderer.update_scene(data, camera=camera)
    else:
        renderer.update_scene(data)

    if show_ee_frame:
        add_ee_frame_overlay(model, data, renderer.scene, axis_length)

    return renderer.render().copy()


def main() -> None:
    config = load_config(RenderPlaybackConfig)

    print(f"Loading result from {config.result_json}")
    result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(result)
    traj_fps = result.config.fps
    duration = result.config.duration
    n_steps = len(trajectory.time)
    print(f"Trajectory: {n_steps} steps, {duration:.1f}s, {traj_fps:.0f} fps")

    loaded = load_and_reset(config.model or None)
    model, data = loaded.model, loaded.data

    tile_w = min(config.width, int(model.vis.global_.offwidth))
    tile_h = min(config.height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)

    if config.multi_camera:
        cameras: list[str | None] = [(c if c else None) for c in config.grid_cameras]
        while len(cameras) < 4:
            cameras.append(None)
        cameras = cameras[:4]
        cam_labels = [c or "default" for c in cameras]
        print(
            f"Multi-camera mode: {cam_labels}, "
            f"tile={tile_w}x{tile_h}, grid={tile_w * 2}x{tile_h * 2}"
        )

    frame_dirs: dict[str, Path] = {}
    if config.save_frames and config.multi_camera:
        base_dir = Path(config.frames_dir)
        for cam in cameras:
            label = cam or "default"
            cam_dir = base_dir / label
            cam_dir.mkdir(parents=True, exist_ok=True)
            frame_dirs[label] = cam_dir
        print(f"Saving per-camera frames to {base_dir}/")
    elif config.save_frames:
        base_dir = Path(config.frames_dir)
        label = config.camera or "default"
        cam_dir = base_dir / label
        cam_dir.mkdir(parents=True, exist_ok=True)
        frame_dirs[label] = cam_dir
        print(f"Saving frames to {cam_dir}/")

    video_fps = config.fps_video
    speed = config.playback_speed
    video_duration = duration / speed
    n_video_frames = int(video_duration * video_fps)
    traj_indices = np.linspace(0, n_steps - 1, n_video_frames, dtype=int)

    print(
        f"Rendering {n_video_frames} frames at {video_fps} fps "
        f"({video_duration:.1f}s video, {speed}x speed)"
    )

    frames: list[np.ndarray] = []
    n_joints = trajectory.position.shape[1]

    for frame_idx, traj_idx in enumerate(traj_indices):
        q = trajectory.position[traj_idx]
        data.qpos[:n_joints] = q
        mujoco.mj_forward(model, data)

        if config.multi_camera:
            tiles = []
            for cam in cameras:
                tile = _render_single_view(
                    renderer, model, data, cam, config.show_ee_frame, config.axis_length
                )
                tiles.append(tile)
                if config.save_frames:
                    label = cam or "default"
                    path = frame_dirs[label] / f"{frame_idx:04d}.png"
                    iio.imwrite(str(path), tile)
            top = np.concatenate([tiles[0], tiles[1]], axis=1)
            bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
            grid = np.concatenate([top, bottom], axis=0)
            frames.append(grid)
        else:
            image = _render_single_view(
                renderer, model, data, config.camera, config.show_ee_frame, config.axis_length
            )
            frames.append(image)
            if config.save_frames:
                label = config.camera or "default"
                path = frame_dirs[label] / f"{frame_idx:04d}.png"
                iio.imwrite(str(path), image)

        if (frame_idx + 1) % 100 == 0 or frame_idx == n_video_frames - 1:
            t_traj = trajectory.time[traj_idx]
            print(f"  frame {frame_idx + 1}/{n_video_frames} (t={t_traj:.2f}s)")

    renderer.close()

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.stack(frames)
    iio.imwrite(str(output_path), video_array, fps=video_fps, macro_block_size=1)
    print(f"\nVideo saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    if config.save_frames:
        total = sum(len(list(d.glob("*.png"))) for d in frame_dirs.values())
        print(f"Saved {total} frame images across {len(frame_dirs)} camera(s)")


if __name__ == "__main__":
    main()
