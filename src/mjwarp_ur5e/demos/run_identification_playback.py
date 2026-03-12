"""Demo CLI for trajectory playback and measurement collection."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import tyro

from mjwarp_ur5e.identification.execution import (
    PlaybackConfig,
    TrajectoryPlayback,
)
from mjwarp_ur5e.identification.io import (
    load_optimization_result,
    result_to_trajectory,
)
from mjwarp_ur5e.model import load_model, reset_to_home


@dataclasses.dataclass
class PlaybackDemoConfig:
    """Configuration for the playback demo."""

    result_json: str = "debug/excitation_result.json"
    model: str = ""
    use_pd: bool = False
    noise_std: float = 0.0
    output: str = "debug/playback_data.npz"


def main() -> None:
    config = tyro.cli(PlaybackDemoConfig)

    # Load optimization result and reconstruct trajectory
    print(f"Loading optimization result from {config.result_json}")
    result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(result)
    print(f"Trajectory: {len(trajectory.time)} steps, duration={trajectory.time[-1]:.2f}s")

    # Load MuJoCo model
    model_path = config.model if config.model else None
    loaded = load_model(model_path)
    reset_to_home(loaded.model, loaded.data)
    print(f"Loaded model: {loaded.model_path}")

    # Configure playback
    playback_config = PlaybackConfig(
        use_pd_control=config.use_pd,
        noise_std_q=config.noise_std,
        noise_std_dq=config.noise_std,
        noise_std_wrench=config.noise_std,
        body_name=result.config.body_name,
        site_name=result.config.site_name,
    )

    rng = np.random.default_rng(42) if config.noise_std > 0 else None

    # Execute playback
    playback = TrajectoryPlayback(loaded.model, loaded.data, playback_config)
    print("Executing trajectory playback...")
    buffer = playback.execute(trajectory, rng=rng)
    print(f"Collected {len(buffer)} samples")

    # Compute and print tracking error
    errors = playback.compute_tracking_error(trajectory, buffer)
    print("\nTracking error summary:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6f}")

    # Save data
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = buffer.to_arrays()
    np.savez(output_path, **arrays)
    print(f"\nSaved playback data to {output_path}")


if __name__ == "__main__":
    main()
