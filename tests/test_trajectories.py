import numpy as np

from mjwarp_ur5e.trajectories import (
    FourierTrajectory,
    FourierTrajectoryConfig,
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)


Q0 = np.array([
    np.pi / 2,
    -np.pi / 2,
    np.pi / 2,
    -np.pi / 2,
    -np.pi / 2,
    np.pi / 2,
])


def _random_coefficients(
    num_joints: int = 6,
    num_harmonics: int = 3,
    scale: float = 0.2,
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    generator = np.random.default_rng(seed)
    return {
        "a": generator.uniform(-scale, scale, size=(num_joints, num_harmonics)).tolist(),
        "b": generator.uniform(-scale, scale, size=(num_joints, num_harmonics)).tolist(),
    }


def test_fourier_trajectory_zero_coefficients_is_static() -> None:
    trajectory = FourierTrajectory(
        FourierTrajectoryConfig(
            duration=5.0,
            fps=50.0,
            num_joints=6,
            num_harmonics=2,
            base_freq=0.2,
            q0=Q0,
        )
    )

    sample = trajectory.sample()

    assert sample.position.shape == (251, 6)
    np.testing.assert_allclose(sample.position, np.tile(Q0, (251, 1)))
    np.testing.assert_allclose(sample.velocity, 0.0)
    np.testing.assert_allclose(sample.acceleration, 0.0)


def test_windowed_fourier_boundary_conditions_hold() -> None:
    trajectory = WindowedFourierTrajectory(
        WindowedFourierTrajectoryConfig(
            duration=5.0,
            fps=50.0,
            num_joints=6,
            num_harmonics=2,
            base_freq=0.2,
            coefficients=_random_coefficients(num_harmonics=2),
            q0=Q0,
        )
    )

    sample = trajectory.sample()

    np.testing.assert_allclose(sample.position[0], Q0, atol=1e-10)
    np.testing.assert_allclose(sample.position[-1], Q0, atol=1e-10)
    np.testing.assert_allclose(sample.velocity[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(sample.velocity[-1], 0.0, atol=1e-10)
    np.testing.assert_allclose(sample.acceleration[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(sample.acceleration[-1], 0.0, atol=1e-10)


def test_windowed_fourier_produces_motion_with_nonzero_coefficients() -> None:
    trajectory = WindowedFourierTrajectory(
        WindowedFourierTrajectoryConfig(
            duration=5.0,
            fps=50.0,
            num_joints=6,
            num_harmonics=3,
            base_freq=0.2,
            coefficients=_random_coefficients(num_harmonics=3, scale=0.3),
            q0=Q0,
        )
    )

    sample = trajectory.sample()
    midpoint = len(sample.time) // 2

    assert np.max(np.abs(sample.position[midpoint] - Q0)) > 1e-2


def test_windowed_fourier_coefficients_round_trip_to_serializable_dict() -> None:
    coefficients = _random_coefficients(num_harmonics=2)
    trajectory = WindowedFourierTrajectory(
        WindowedFourierTrajectoryConfig(
            duration=3.0,
            fps=20.0,
            num_joints=6,
            num_harmonics=2,
            base_freq=0.3,
            coefficients=coefficients,
            q0=Q0,
        )
    )

    assert trajectory.as_serializable_coefficients() == coefficients