from __future__ import annotations

from mjwarp_ur5e.cli.configs import OptimizeExcitationConfig, ValidateExcitationConfig


def test_optimize_excitation_config_defaults() -> None:
    cfg = OptimizeExcitationConfig()
    assert cfg.model == ""
    assert cfg.num_harmonics == 5
    assert cfg.base_freq == 0.1
    assert cfg.duration == 10.0
    assert cfg.fps == 100.0
    assert cfg.subsample_factor == 10
    assert cfg.n_monte_carlo == 20
    assert cfg.max_iter == 200
    assert cfg.seed == 42
    assert cfg.max_displacement == 0.5
    assert cfg.enable_collision is True
    assert cfg.output == "results/excitation_result.json"


def test_validate_excitation_config_defaults() -> None:
    cfg = ValidateExcitationConfig()
    assert cfg.result_json == "results/excitation_result.json"
    assert cfg.model == ""


def test_optimize_demo_importable() -> None:
    import mjwarp_ur5e.demos.optimize_excitation_trajectory  # noqa: F401


def test_validate_demo_importable() -> None:
    import mjwarp_ur5e.demos.validate_excitation_trajectory  # noqa: F401
