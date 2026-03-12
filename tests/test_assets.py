from pathlib import Path

from mjwarp_ur5e.assets import get_asset_layout, get_default_ur5e_model_candidates


def test_asset_layout_points_to_repo_assets() -> None:
    layout = get_asset_layout()

    assert layout.repo_root == Path(__file__).resolve().parents[1]
    assert layout.ur5e_root == layout.repo_root / "assets" / "ur5e"


def test_default_candidates_cover_mjcf_and_urdf() -> None:
    candidates = get_default_ur5e_model_candidates()

    assert candidates[0].as_posix().endswith("assets/ur5e/mjcf/scene.xml")
    assert candidates[1].as_posix().endswith("assets/ur5e/mjcf/ur5e.xml")
    assert candidates[2].as_posix().endswith("assets/ur5e/urdf/ur5e.urdf")
