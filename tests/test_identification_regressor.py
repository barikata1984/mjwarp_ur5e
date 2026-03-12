import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mjwarp_ur5e.identification import (
    body_inertial_parameters_from_model,
    compute_condition_number,
    compute_stacked_body_regressor,
    compute_wrench_from_parameters,
    sample_body_regressor,
    set_model_state,
)
from mjwarp_ur5e.model import load_model, reset_to_home
from mjwarp_ur5e.trajectories import WindowedFourierTrajectory, WindowedFourierTrajectoryConfig


PAYLOAD_BODY_NAME = "payload_box_mount"
Q0 = np.array([
    np.pi / 2,
    -np.pi / 2,
    np.pi / 2,
    -np.pi / 2,
    -np.pi / 2,
    0.0,
])


def _load_payload_scene():
    loaded = load_model("assets/ur5e/mjcf/scene_with_box.xml")
    reset_to_home(loaded.model, loaded.data)
    return loaded


def _trajectory(scale: float = 0.1):
    generator = np.random.default_rng(42)
    coefficients = {
        "a": generator.uniform(-scale, scale, size=(6, 2)).tolist(),
        "b": generator.uniform(-scale, scale, size=(6, 2)).tolist(),
    }
    return WindowedFourierTrajectory(
        WindowedFourierTrajectoryConfig(
            duration=4.0,
            fps=50.0,
            num_joints=6,
            num_harmonics=2,
            base_freq=0.2,
            coefficients=coefficients,
            q0=Q0,
        )
    ).sample()


def test_body_inertial_parameters_extract_expected_payload_values() -> None:
    loaded = _load_payload_scene()
    parameters = body_inertial_parameters_from_model(loaded.model, PAYLOAD_BODY_NAME)

    np.testing.assert_allclose(parameters.mass, 1.0)
    np.testing.assert_allclose(parameters.first_moments, np.array([0.0, -0.1, 0.125]))
    np.testing.assert_allclose(
        parameters.inertia_matrix,
        np.array(
            [
                [0.0360417, 0.0, 0.0],
                [0.0, 0.0260417, 0.0125],
                [0.0, 0.0125, 0.0204167],
            ]
        ),
        atol=1e-6,
    )


def test_sample_body_regressor_has_expected_shape() -> None:
    loaded = _load_payload_scene()
    sample = sample_body_regressor(loaded.model, loaded.data, PAYLOAD_BODY_NAME)

    assert sample.regressor.shape == (6, 10)
    assert sample.kinematics.body_name == PAYLOAD_BODY_NAME


def test_static_pose_regressor_predicts_gravity_wrench() -> None:
    loaded = _load_payload_scene()
    parameters = body_inertial_parameters_from_model(loaded.model, PAYLOAD_BODY_NAME)
    sample = sample_body_regressor(loaded.model, loaded.data, PAYLOAD_BODY_NAME)
    wrench = compute_wrench_from_parameters(sample.regressor, parameters)

    assert wrench.shape == (6,)
    assert np.linalg.norm(wrench[3:]) > 1.0


def test_stacked_body_regressor_shape_and_condition_number_change_with_motion() -> None:
    loaded = _load_payload_scene()
    static_q = np.tile(Q0, (21, 1))
    static_dq = np.zeros((21, loaded.model.nv))
    static_ddq = np.zeros((21, loaded.model.nv))

    static_regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        static_q,
        static_dq,
        static_ddq,
        PAYLOAD_BODY_NAME,
    )

    dynamic_sample = _trajectory(scale=0.2)
    dynamic_regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        dynamic_sample.position,
        dynamic_sample.velocity,
        dynamic_sample.acceleration,
        PAYLOAD_BODY_NAME,
        subsample_factor=10,
    )

    assert static_regressor.shape == (21 * 6, 10)
    assert dynamic_regressor.shape[1] == 10

    static_condition = compute_condition_number(static_regressor)
    dynamic_condition = compute_condition_number(dynamic_regressor)
    static_rank = np.linalg.matrix_rank(static_regressor)
    dynamic_rank = np.linalg.matrix_rank(dynamic_regressor)

    assert np.isinf(static_condition) or static_condition > 1e8
    assert np.isinf(dynamic_condition)
    assert dynamic_rank > static_rank


def test_set_model_state_validates_shapes() -> None:
    loaded = _load_payload_scene()
    with pytest.raises(ValueError):
        set_model_state(
            loaded.model,
            loaded.data,
            np.zeros(loaded.model.nq + 1),
        )