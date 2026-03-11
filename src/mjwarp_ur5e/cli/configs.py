from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LoadConfig:
    model: str | None = None
    steps: int = 1
    headless: bool = False


@dataclass(slots=True)
class InspectConfig:
    model: str | None = None
    json_output: bool = False


@dataclass(slots=True)
class RenderConfig:
    model: str | None = None
    width: int = 1280
    height: int = 960
    output: str | None = None
    camera: str | None = None
    show_base_frame: bool = False
    show_ee_frame: bool = False
    axis_length: float = 0.3
    set_joint: list[str] = field(default_factory=list)