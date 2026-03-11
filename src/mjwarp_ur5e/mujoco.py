from __future__ import annotations

from pathlib import Path

import mujoco

from mjwarp_ur5e.model import load_model as _load_model


def load_ur5e_model(model_path: str | Path | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    loaded_model = _load_model(model_path)
    return loaded_model.model, loaded_model.data, loaded_model.model_path


def step_model(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)