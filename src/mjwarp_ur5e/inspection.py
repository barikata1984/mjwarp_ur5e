from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco

from mjwarp_ur5e.frames import get_site_body_name, get_site_frame
from mjwarp_ur5e.model import get_home_qpos, get_object_names, load_model, reset_to_home


@dataclass(frozen=True)
class EndEffectorInfo:
    site_name: str | None
    body_name: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class UR5eInspectionResult:
    model_path: Path
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    site_names: tuple[str, ...]
    home_qpos: tuple[float, ...] | None
    end_effector: EndEffectorInfo

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        return payload


def inspect_ur5e_model(model_path: str | Path | None = None) -> UR5eInspectionResult:
    loaded_model = load_model(model_path)
    model = loaded_model.model
    data = loaded_model.data
    reset_to_home(model, data)

    joint_names = get_object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = get_object_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = get_object_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)

    ee_frame = get_site_frame(model, data, "attachment_site")
    body_name = get_site_body_name(model, "attachment_site")
    if ee_frame is None:
        body_name = body_name or f"body_{model.nbody - 1}"
        position = tuple(float(value) for value in data.xpos[model.nbody - 1])
        site_name = None
    else:
        body_name = body_name or "wrist_3_link"
        position = ee_frame.position
        site_name = ee_frame.name

    return UR5eInspectionResult(
        model_path=loaded_model.model_path,
        joint_names=joint_names,
        body_names=body_names,
        site_names=site_names,
        home_qpos=get_home_qpos(model),
        end_effector=EndEffectorInfo(
            site_name=site_name,
            body_name=body_name,
            position=position,
        ),
    )
