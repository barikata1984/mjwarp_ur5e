"""Render each optimization result as a 2x2 multi-camera grid video."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.rendering import add_ee_frame_overlay

# 2x2 grid: top-left=overview, top-right=top, bottom-left=front, bottom-right=side
CAMERAS: list[str | None] = ["payload_overview", "view_z", "view_x", "view_y"]
CAM_LABELS = ["overview", "top", "front", "side"]

RESULTS = [
    ("scaled_dq2.0_5s", "results/ft_offset_dq2.0_5s/excitation_result.json"),
    ("scaled_dq1.5_10s", "results/ft_offset_dq1.5_10s/excitation_result.json"),
    ("noscale_dq2.0_5s", "results/ft_offset_noscale_dq2.0_5s/excitation_result.json"),
    ("noscale_dq1.5_10s", "results/ft_offset_noscale_dq1.5_10s/excitation_result.json"),
]


def render_one(
    label: str,
    result_path: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    tile_w: int,
    tile_h: int,
    output_dir: Path,
    video_fps: int = 30,
) -> None:
    print(f"\n=== {label} ===")
    result = load_optimization_result(result_path)
    traj = result_to_trajectory(result)
    duration = result.config.duration
    n_steps = len(traj.time)
    print(f"  cond={result.condition_number:.2f}, duration={duration:.1f}s, steps={n_steps}")

    n_video_frames = int(duration * video_fps)
    traj_indices = np.linspace(0, n_steps - 1, n_video_frames, dtype=int)

    frames: list[np.ndarray] = []
    n_joints = traj.position.shape[1]

    for fi, ti in enumerate(traj_indices):
        data.qpos[:n_joints] = traj.position[ti]
        mujoco.mj_forward(model, data)

        tiles = []
        for cam in CAMERAS:
            if cam:
                renderer.update_scene(data, camera=cam)
            else:
                renderer.update_scene(data)
            add_ee_frame_overlay(model, data, renderer.scene, 0.12)
            tiles.append(renderer.render().copy())

        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        frames.append(np.concatenate([top, bottom], axis=0))

        if (fi + 1) % 100 == 0 or fi == n_video_frames - 1:
            print(f"  frame {fi + 1}/{n_video_frames}")

    output_path = output_dir / f"{label}.mp4"
    video_array = np.stack(frames)
    iio.imwrite(str(output_path), video_array, fps=video_fps, macro_block_size=1)
    print(f"  -> {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    loaded = load_and_reset()
    model, data = loaded.model, loaded.data

    tile_w, tile_h = 640, 480
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)

    output_dir = Path("results/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, path in RESULTS:
        render_one(label, path, model, data, renderer, tile_w, tile_h, output_dir)

    renderer.close()
    print("\nAll videos saved to results/videos/")


if __name__ == "__main__":
    main()
