from mjwarp_ur5e.cli import InspectConfig, LoadConfig, RenderConfig


def test_cli_configs_have_expected_defaults() -> None:
    load = LoadConfig()
    inspect = InspectConfig()
    render = RenderConfig()

    assert load.steps == 1
    assert load.headless is False
    assert inspect.json_output is False
    assert render.axis_length == 0.3
    assert render.set_joint == []