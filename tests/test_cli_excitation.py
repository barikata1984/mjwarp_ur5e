from __future__ import annotations

from mjwarp_ur5e.cli.configs import OptimizeExcitationConfig, ValidateExcitationConfig


def test_optimize_excitation_config_defaults() -> None:
    cfg = OptimizeExcitationConfig()
    assert cfg.model == "assets/ur5e/mjcf/scene_with_box.xml"
    assert cfg.num_harmonics == 3
    assert abs(cfg.base_freq - 1.0 / 3.0) < 1e-10
    assert cfg.duration == 3.0
    assert cfg.fps == 100.0
    assert cfg.subsample_factor == 5
    assert cfg.n_monte_carlo == 20
    assert cfg.max_iter == 200
    assert cfg.seed == 42
    assert cfg.max_displacement == 0.0
    assert cfg.enable_collision is True
    assert cfg.dq_max == 1.5
    assert cfg.ddq_max == 0.0
    assert cfg.output == "results/excitation_result.json"


def test_validate_excitation_config_defaults() -> None:
    cfg = ValidateExcitationConfig()
    assert cfg.result_json == "results/excitation_result.json"
    assert cfg.model == "assets/ur5e/mjcf/scene_with_box.xml"


def test_optimize_demo_importable() -> None:
    import mjwarp_ur5e.demos.optimize_excitation_trajectory  # noqa: F401


def test_validate_demo_importable() -> None:
    import mjwarp_ur5e.demos.validate_excitation_trajectory  # noqa: F401
