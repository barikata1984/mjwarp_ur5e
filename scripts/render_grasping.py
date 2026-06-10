"""Render the UR5e + FT300s + 2F-85 assembly grasping a 5cm cube."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

CUBE_BODY_XML = """\
                          <body name="cube" pos="0 0 0.145">
                            <inertial mass="0.1" pos="0 0 0"
                              diaginertia="4.17e-05 4.17e-05 4.17e-05"/>
                            <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
                              rgba="0.85 0.32 0.1 1" friction="1.0 0.005 0.001"/>
                          </body>
"""

PINCH_SITE_TAG = (
    '<site name="pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>'
)


def _load_model(scene_xml: str) -> mujoco.MjModel:
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
    tmp_scene = mjcf_dir / "_tmp_scene_grasping.xml"
    tmp_scene.write_text(scene_text)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_scene))
    finally:
        tmp_robot.unlink(missing_ok=True)
        tmp_scene.unlink(missing_ok=True)
    return model


def main() -> None:
    scene_xml = "assets/ur5e/mjcf/scene_grasping.xml"
    model = _load_model(scene_xml)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ur_home = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    for i, c in enumerate(ur_home):
        data.ctrl[i] = c

    fingers_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    data.ctrl[fingers_id] = 255

    model.opt.gravity[:] = 0
    for _ in range(3000):
        mujoco.mj_step(model, data)

    pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    pinch_pos = data.site_xpos[pinch_id].copy()

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id].copy()
    print(f"Pinch: {pinch_pos}, Cube: {cube_pos}")

    w, h = 960, 720
    renderer = mujoco.Renderer(model, height=h, width=w)

    renderer.update_scene(data, camera="overview")
    img_overview = renderer.render().copy()

    def render_closeup(azimuth: float, elevation: float, distance: float = 0.3) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = pinch_pos
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()

    img_front = render_closeup(azimuth=90, elevation=-15)
    img_side = render_closeup(azimuth=0, elevation=-15)
    img_angle = render_closeup(azimuth=135, elevation=-25)

    renderer.close()

    images = [img_overview, img_front, img_side, img_angle]
    labels = ["Overview", "Front Close-up", "Side Close-up", "Angled Close-up"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "UR5e + FT300s + Robotiq 2F-85 — Grasping 5cm Cube",
        fontsize=16,
        fontweight="bold",
    )

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=13)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path("ur5e_ft300s_2f85_grasping.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
