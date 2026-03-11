import pytest

pytest.importorskip("mujoco")

from mjwarp_ur5e.rendering import parse_joint_overrides


def test_parse_joint_overrides_parses_multiple_entries() -> None:
    overrides = parse_joint_overrides([
        "shoulder_pan_joint=1.5707963267948966",
        "wrist_3_joint=0.25",
    ])

    assert overrides == {
        "shoulder_pan_joint": 1.5707963267948966,
        "wrist_3_joint": 0.25,
    }