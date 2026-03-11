from __future__ import annotations

import tyro

from mjwarp_ur5e.cli import LoadConfig
from mjwarp_ur5e.model import load_model, step_model


def main() -> None:
    config = tyro.cli(LoadConfig)
    loaded_model = load_model(config.model)
    step_model(loaded_model.model, loaded_model.data, config.steps)

    print(f"Loaded UR5e model: {loaded_model.model_path}")
    print(
        f"nq={loaded_model.model.nq}, nv={loaded_model.model.nv}, "
        f"nu={loaded_model.model.nu}, bodies={loaded_model.model.nbody}"
    )

    if config.headless:
        return

    import mujoco.viewer

    mujoco.viewer.launch(loaded_model.model, loaded_model.data)


if __name__ == "__main__":
    main()