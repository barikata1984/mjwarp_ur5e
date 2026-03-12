from __future__ import annotations

import tyro

from mjwarp_ur5e.cli import RenderConfig
from mjwarp_ur5e.rendering import RenderSceneRequest, parse_joint_overrides, render_scene


def main() -> None:
    config = tyro.cli(RenderConfig)
    joint_overrides = parse_joint_overrides(config.set_joint)
    output_path, metadata_path = render_scene(
        RenderSceneRequest(
            model_path=config.model,
            output_path=config.output,
            width=config.width,
            height=config.height,
            camera=config.camera,
            show_base_frame=config.show_base_frame,
            show_ee_frame=config.show_ee_frame,
            axis_length=config.axis_length,
            joint_overrides=joint_overrides,
        )
    )
    print(f"Rendered UR5e home pose from {config.model or 'default model'}")
    print(f"Base frame overlay: {config.show_base_frame}")
    print(f"EE frame overlay: {config.show_ee_frame}")
    print(f"Joint overrides: {joint_overrides}")
    print(f"Image: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
