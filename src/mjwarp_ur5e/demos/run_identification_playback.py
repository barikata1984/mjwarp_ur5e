"""Demo CLI for trajectory playback and measurement collection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mjwarp_ur5e.cli import PlaybackDemoConfig, load_config
from mjwarp_ur5e.identification.execution import PlaybackConfig, TrajectoryPlayback
from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.model import load_and_reset


def main() -> None:
    config = load_config(PlaybackDemoConfig)

    print(f"Loading optimization result from {config.result_json}")
    result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(result)
    print(f"Trajectory: {len(trajectory.time)} steps, duration={trajectory.time[-1]:.2f}s")

    loaded = load_and_reset(config.model or None)
    print(f"Loaded model: {loaded.model_path}")

    playback_config = PlaybackConfig(
        use_pd_control=config.use_pd,
        noise_std_q=config.noise_std,
        noise_std_dq=config.noise_std,
        noise_std_wrench=config.noise_std,
        body_name=result.config.body_name,
        site_name=result.config.site_name,
    )

    rng = np.random.default_rng(42) if config.noise_std > 0 else None

    playback = TrajectoryPlayback(loaded.model, loaded.data, playback_config)
    print("Executing trajectory playback...")
    buffer = playback.execute(trajectory, rng=rng)
    print(f"Collected {len(buffer)} samples")

    errors = playback.compute_tracking_error(trajectory, buffer)
    print("\nTracking error summary:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6f}")

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = buffer.to_arrays()
    np.savez(output_path, **arrays)
    print(f"\nSaved playback data to {output_path}")


if __name__ == "__main__":
    main()
