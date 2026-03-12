"""Export a sampled trajectory JSON from an existing optimization result."""

from __future__ import annotations

from pathlib import Path

import tyro

from mjwarp_ur5e.cli import ExportTrajectoryConfig
from mjwarp_ur5e.identification.io import (
    load_optimization_result,
    result_to_trajectory,
    save_trajectory_json,
)


def main() -> None:
    config = tyro.cli(ExportTrajectoryConfig)

    result = load_optimization_result(config.result_json)
    output_fps = config.fps if config.fps > 0 else None
    trajectory = result_to_trajectory(result, fps=output_fps)

    output_path = Path(config.output)
    save_trajectory_json(
        trajectory,
        output_path,
        condition_number=result.condition_number,
        source=Path(config.result_json).name,
    )

    effective_fps = output_fps if output_fps else result.config.fps
    print(f"Exported: {output_path}")
    print(f"  Source: {config.result_json}")
    print(f"  Steps: {len(trajectory.time)}, FPS: {effective_fps:.0f} Hz")
    print(f"  Duration: {result.config.duration:.1f}s")


if __name__ == "__main__":
    main()
