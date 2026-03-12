"""Centralized CLI configuration dataclasses.

All demo and pipeline configs live here. YAML defaults are loaded from
configs/default.yaml and can be overridden by tyro CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Base configs (shared field groups)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelConfig:
    """Common model path field used by all demos."""

    model: str = ""


@dataclass(slots=True)
class ResultInputConfig(ModelConfig):
    """Common fields for demos that read an optimization result."""

    result_json: str = "results/excitation_result.json"


# ---------------------------------------------------------------------------
# Original demo configs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Excitation pipeline configs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OptimizeExcitationConfig(ModelConfig):
    """Config for excitation trajectory optimization CLI."""

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
    enable_payload_workspace: bool = True
    output: str = "results/excitation_result.json"
    trajectory_output: str = "results/excitation_trajectory.json"
    trajectory_fps: float = 0.0


@dataclass(slots=True)
class ExportTrajectoryConfig:
    """Config for exporting a sampled trajectory JSON from an optimization result."""

    result_json: str = "results/excitation_result.json"
    output: str = "results/excitation_trajectory.json"
    fps: float = 0.0


@dataclass(slots=True)
class ValidateExcitationConfig(ResultInputConfig):
    """Config for excitation trajectory validation CLI."""

    pass


# ---------------------------------------------------------------------------
# Identification pipeline configs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IdentificationDemoConfig(ResultInputConfig):
    """Config for the inertial identification demo."""

    estimator: str = "ls"  # "ls", "tls", or "rtls"
    noise_std: float = 0.0
    regularization: float = 0.0
    output: str = "results/identification_result.json"


@dataclass(slots=True)
class PlaybackDemoConfig(ResultInputConfig):
    """Config for the playback demo."""

    use_pd: bool = False
    noise_std: float = 0.0
    output: str = "results/playback_data.npz"


@dataclass(slots=True)
class RenderPlaybackConfig(ResultInputConfig):
    """Config for rendering excitation trajectory playback."""

    output: str = "results/excitation_playback.mp4"
    width: int = 960
    height: int = 544
    camera: str | None = None
    fps_video: int = 30
    show_ee_frame: bool = True
    axis_length: float = 0.15
    playback_speed: float = 1.0
    multi_camera: bool = False
    grid_cameras: tuple[str, ...] = ("", "view_x", "view_y", "view_z")
    save_frames: bool = False
    frames_dir: str = "results/frames"
