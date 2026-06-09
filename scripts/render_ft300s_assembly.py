"""Render 4-view grid (overview, top, front, side) of the UR5e + FT300s + 2F-85 assembly."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def main() -> None:
    scene_xml = "assets/ur5e/mjcf/scene_with_ft300s_and_gripper.xml"
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    w, h = 960, 720
    renderer = mujoco.Renderer(model, height=h, width=w)

    cameras = ["overview", "top", "front", "side"]
    labels = ["Overview", "Top", "Front", "Side"]
    images: list[np.ndarray] = []

    for cam_name in cameras:
        renderer.update_scene(data, camera=cam_name)
        images.append(renderer.render().copy())

    renderer.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("UR5e + FT300s + Robotiq 2F-85 Assembly", fontsize=16, fontweight="bold")

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=13)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path("ur5e_ft300s_2f85_assembly.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
