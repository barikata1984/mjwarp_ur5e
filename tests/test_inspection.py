import pytest

mujoco = pytest.importorskip("mujoco")

from mjwarp_ur5e.inspection import inspect_ur5e_model


def test_inspection_returns_expected_ur5e_metadata() -> None:
    result = inspect_ur5e_model()

    assert result.joint_names == (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    assert result.site_names == ("attachment_site",)
    assert result.end_effector.site_name == "attachment_site"
    assert result.end_effector.body_name == "wrist_3_link"
    assert result.home_qpos is not None
    assert len(result.home_qpos) == 6