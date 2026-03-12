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


@dataclass(slots=True)
class OptimizeExcitationConfig:
    """Config for excitation trajectory optimization CLI."""

    model: str = ""
    num_harmonics: int = 5
    base_freq: float = 0.1
    duration: float = 10.0
    fps: float = 100.0
    subsample_factor: int = 10
    n_monte_carlo: int = 20
    max_iter: int = 200
    seed: int = 42
    max_displacement: float = 0.5
    enable_collision: bool = True
    output: str = "debug/excitation_result.json"


@dataclass(slots=True)
class ValidateExcitationConfig:
    """Config for excitation trajectory validation CLI."""

    result_json: str = "debug/excitation_result.json"
    model: str = ""
