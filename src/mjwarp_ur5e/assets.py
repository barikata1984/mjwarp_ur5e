from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetLayout:
    repo_root: Path
    asset_root: Path
    ur5e_root: Path


def get_asset_layout() -> AssetLayout:
    repo_root = Path(__file__).resolve().parents[2]
    asset_root = repo_root / "assets"
    ur5e_root = asset_root / "ur5e"
    return AssetLayout(repo_root=repo_root, asset_root=asset_root, ur5e_root=ur5e_root)


def get_default_ur5e_model_candidates() -> list[Path]:
    ur5e_root = get_asset_layout().ur5e_root
    return [
        ur5e_root / "mjcf" / "scene.xml",
        ur5e_root / "mjcf" / "ur5e.xml",
        ur5e_root / "urdf" / "ur5e.urdf",
    ]


def resolve_ur5e_model_path(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        if not candidate.is_absolute():
            candidate = get_asset_layout().repo_root / candidate
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"UR5e model not found: {candidate}")

    for candidate in get_default_ur5e_model_candidates():
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in get_default_ur5e_model_candidates())
    raise FileNotFoundError(
        f"No UR5e model asset found. Place one of the following files:\n{searched}"
    )
