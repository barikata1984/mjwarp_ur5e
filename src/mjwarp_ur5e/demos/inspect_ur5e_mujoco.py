from __future__ import annotations

import json

from mjwarp_ur5e.cli import InspectConfig, load_config
from mjwarp_ur5e.inspection import inspect_ur5e_model


def main() -> None:
    config = load_config(InspectConfig)
    result = inspect_ur5e_model(config.model)

    if config.json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Model path: {result.model_path}")
    print("Joint names:")
    for name in result.joint_names:
        print(f"- {name}")
    print("Site names:")
    for name in result.site_names:
        print(f"- {name}")
    print("Home qpos:")
    if result.home_qpos is None:
        print("- <missing>")
    else:
        print("- " + ", ".join(f"{value:.4f}" for value in result.home_qpos))
    print("End effector:")
    print(f"- site: {result.end_effector.site_name}")
    print(f"- body: {result.end_effector.body_name}")
    print("- position: " + ", ".join(f"{value:.4f}" for value in result.end_effector.position))


if __name__ == "__main__":
    main()
