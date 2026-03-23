"""Render multiple excitation trajectory results as a 2x2 grid video for comparison."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.rendering import add_ee_frame_overlay


def _draw_label(image: np.ndarray, text: str) -> np.ndarray:
    """Draw a label background bar at the top of the image."""
    img = image.copy()
    bar_h = 28
    img[:bar_h, :] = (img[:bar_h, :].astype(np.float32) * 0.3).astype(np.uint8)
    return img


def main() -> None:
    result_paths = [
        ("scaled dq=2.0 5s", "results/ft_offset_dq2.0_5s/excitation_result.json"),
        ("scaled dq=1.5 10s", "results/ft_offset_dq1.5_10s/excitation_result.json"),
        ("unscaled dq=2.0 5s", "results/ft_offset_noscale_dq2.0_5s/excitation_result.json"),
        ("unscaled dq=1.5 10s", "results/ft_offset_noscale_dq1.5_10s/excitation_result.json"),
    ]

    # Load all results and trajectories
    results = []
    trajectories = []
    for label, path in result_paths:
        print(f"Loading {label}: {path}")
        r = load_optimization_result(path)
        t = result_to_trajectory(r)
        results.append(r)
        trajectories.append(t)
        print(
            f"  cond={r.condition_number:.2f}, duration={r.config.duration:.1f}s, "
            f"steps={len(t.time)}"
        )

    # Determine common video parameters
    # Normalize all trajectories to phase [0, 1] and play at the longest duration
    max_duration = max(r.config.duration for r in results)
    video_fps = 30
    video_duration = max_duration
    n_video_frames = int(video_duration * video_fps)

    loaded = load_and_reset()
    model, data = loaded.model, loaded.data

    tile_w, tile_h = 640, 480
    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)

    print(
        f"\nRendering {n_video_frames} frames at {video_fps} fps "
        f"({video_duration:.1f}s, grid {tile_w * 2}x{tile_h * 2})"
    )

    frames: list[np.ndarray] = []

    for fi in range(n_video_frames):
        t_video = fi / video_fps
        tiles = []

        for i, (label, _) in enumerate(result_paths):
            traj = trajectories[i]
            dur = results[i].config.duration
            n_steps = len(traj.time)

            # Map video time to trajectory time (loop if video is longer)
            t_traj = t_video % dur
            traj_idx = min(int(t_traj / dur * (n_steps - 1)), n_steps - 1)

            q = traj.position[traj_idx]
            data.qpos[: q.shape[0]] = q
            mujoco.mj_forward(model, data)

            renderer.update_scene(data)
            add_ee_frame_overlay(model, data, renderer.scene, 0.12)
            tile = renderer.render().copy()
            tile = _draw_label(tile, label)
            tiles.append(tile)

        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        frames.append(grid)

        if (fi + 1) % 100 == 0 or fi == n_video_frames - 1:
            print(f"  frame {fi + 1}/{n_video_frames} (t={t_video:.2f}s)")

    renderer.close()

    output_path = Path("results/grid_comparison.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.stack(frames)
    iio.imwrite(str(output_path), video_array, fps=video_fps, macro_block_size=1)
    print(f"\nGrid video saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
