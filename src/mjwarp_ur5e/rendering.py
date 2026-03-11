from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from mjwarp_ur5e.assets import get_asset_layout
from mjwarp_ur5e.frames import FramePose, get_body_frame, get_site_frame
from mjwarp_ur5e.model import (
    LoadedModel,
    apply_joint_overrides,
    get_home_qpos,
    get_object_names,
    load_model,
    reset_to_home,
)


@dataclass(frozen=True)
class RenderSceneRequest:
    model_path: str | Path | None = None
    output_path: str | Path | None = None
    width: int = 1280
    height: int = 960
    camera: str | None = None
    show_base_frame: bool = False
    show_ee_frame: bool = False
    axis_length: float = 0.3
    joint_overrides: dict[str, float] | None = None


def parse_joint_overrides(entries: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --set-joint value: {entry}")
        name, value = entry.split("=", maxsplit=1)
        overrides[name.strip()] = float(value)
    return overrides


def resolve_output_path(output_path: str | Path | None) -> Path:
    repo_root = get_asset_layout().repo_root
    if output_path is None:
        return repo_root / "debug" / "ur5e_home.png"

    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def resolve_render_size(model: mujoco.MjModel, requested_width: int, requested_height: int) -> tuple[int, int]:
    width = min(requested_width, int(model.vis.global_.offwidth))
    height = min(requested_height, int(model.vis.global_.offheight))
    return width, height


def _add_connector(scene: mujoco.MjvScene, from_point: np.ndarray, to_point: np.ndarray, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 16.0, from_point, to_point)
    scene.ngeom += 1


def _add_frame_overlay(scene: mujoco.MjvScene, frame: FramePose | None, axis_length: float, colors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    if frame is None:
        return

    origin = np.array(frame.position, dtype=np.float64)
    rotation = np.array(frame.rotation, dtype=np.float64)
    for axis_index, color in enumerate(colors):
        direction = rotation[:, axis_index]
        _add_connector(scene, origin, origin + axis_length * direction, color)


def add_base_frame_overlay(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene, axis_length: float) -> None:
    colors = (
        np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32),
    )
    _add_frame_overlay(scene, get_body_frame(model, data, "base"), axis_length, colors)


def add_ee_frame_overlay(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene, axis_length: float) -> None:
    colors = (
        np.array([1.0, 0.4, 0.4, 1.0], dtype=np.float32),
        np.array([0.4, 1.0, 0.4, 1.0], dtype=np.float32),
        np.array([0.4, 0.7, 1.0, 1.0], dtype=np.float32),
    )
    _add_frame_overlay(scene, get_site_frame(model, data, "attachment_site"), axis_length, colors)


def build_render_metadata(loaded_model: LoadedModel, joint_overrides: dict[str, float]) -> dict[str, object]:
    model = loaded_model.model
    data = loaded_model.data
    ee_frame = get_site_frame(model, data, "attachment_site")
    return {
        "model_path": str(loaded_model.model_path),
        "joint_names": list(get_object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)),
        "site_names": list(get_object_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)),
        "home_qpos": list(get_home_qpos(model) or ()),
        "render_qpos": [float(value) for value in data.qpos[: model.nq]],
        "joint_overrides": joint_overrides,
        "end_effector_position": None if ee_frame is None else list(ee_frame.position),
    }


def render_scene(request: RenderSceneRequest) -> tuple[Path, Path]:
    output_path = resolve_output_path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model(request.model_path)
    reset_to_home(loaded_model.model, loaded_model.data)
    joint_overrides = request.joint_overrides or {}
    apply_joint_overrides(loaded_model.model, loaded_model.data, joint_overrides)

    width, height = resolve_render_size(loaded_model.model, request.width, request.height)
    renderer = mujoco.Renderer(loaded_model.model, height=height, width=width)
    if request.camera is None:
        renderer.update_scene(loaded_model.data)
    else:
        renderer.update_scene(loaded_model.data, camera=request.camera)

    if request.show_base_frame:
        add_base_frame_overlay(loaded_model.model, loaded_model.data, renderer.scene, request.axis_length)
    if request.show_ee_frame:
        add_ee_frame_overlay(loaded_model.model, loaded_model.data, renderer.scene, request.axis_length)

    image = renderer.render()
    iio.imwrite(output_path, image)
    renderer.close()

    metadata_path = output_path.with_suffix(".json")
    metadata = build_render_metadata(loaded_model, joint_overrides)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path