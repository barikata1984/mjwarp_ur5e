from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjwarp_ur5e.assets import resolve_ur5e_model_path


@dataclass
class LoadedModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    model_path: Path


def load_model(model_path: str | Path | None = None) -> LoadedModel:
    resolved_path = resolve_ur5e_model_path(model_path)
    model = mujoco.MjModel.from_xml_path(str(resolved_path))
    data = mujoco.MjData(model)
    return LoadedModel(model=model, data=data, model_path=resolved_path)


def step_model(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def get_object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
    names: list[str] = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        names.append(name if name is not None else f"<{object_type.name}:{index}>")
    return tuple(names)


def get_named_object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int | None:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        return None
    return int(object_id)


def get_home_qpos(model: mujoco.MjModel) -> tuple[float, ...] | None:
    key_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id is None:
        return None
    return tuple(float(value) for value in model.key_qpos[key_id][: model.nq])


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    key_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id is None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return False

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return True


def apply_joint_overrides(model: mujoco.MjModel, data: mujoco.MjData, overrides: dict[str, float]) -> None:
    for joint_name, joint_value in overrides.items():
        joint_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id is None:
            raise ValueError(f"Unknown joint name: {joint_name}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = joint_value
    mujoco.mj_forward(model, data)