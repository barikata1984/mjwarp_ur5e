from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mjwarp_ur5e.identification.constraints import build_trajectory_from_params
from mjwarp_ur5e.identification.optimizer import OptimizationResult, OptimizerConfig
from mjwarp_ur5e.trajectories.base import TrajectorySample


def save_optimization_result(
    result: OptimizationResult,
    path: str | Path,
) -> None:
    """Save an OptimizationResult to a JSON file."""
    cfg = result.config
    payload = {
        "x_opt": result.x_opt.tolist(),
        "condition_number": result.condition_number,
        "feasible": result.feasible,
        "constraint_margins": result.constraint_margins,
        "trajectory_stats": result.trajectory_stats,
        "a_opt": result.a_opt.tolist(),
        "b_opt": result.b_opt.tolist(),
        "q0": result.q0.tolist(),
        "n_evaluations": result.n_evaluations,
        "wall_time": result.wall_time,
        "n_restarts": result.n_restarts,
        "best_start_index": result.best_start_index,
        "config": {
            "num_joints": cfg.num_joints,
            "num_harmonics": cfg.num_harmonics,
            "base_freq": cfg.base_freq,
            "duration": cfg.duration,
            "fps": cfg.fps,
            "q0": np.asarray(cfg.q0).tolist(),
            "subsample_factor": cfg.subsample_factor,
            "n_monte_carlo": cfg.n_monte_carlo,
            "max_iter_per_start": cfg.max_iter_per_start,
            "optimizer_method": cfg.optimizer_method,
            "ftol": cfg.ftol,
            "seed": cfg.seed,
            "body_name": cfg.body_name,
            "site_name": cfg.site_name,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_optimization_result(path: str | Path) -> OptimizationResult:
    """Load an OptimizationResult from a JSON file."""
    with open(Path(path)) as f:
        payload = json.load(f)

    cfg_dict = payload["config"]
    config = OptimizerConfig(
        num_joints=cfg_dict["num_joints"],
        num_harmonics=cfg_dict["num_harmonics"],
        base_freq=cfg_dict["base_freq"],
        duration=cfg_dict["duration"],
        fps=cfg_dict["fps"],
        q0=np.array(cfg_dict["q0"], dtype=np.float64),
        subsample_factor=cfg_dict["subsample_factor"],
        n_monte_carlo=cfg_dict["n_monte_carlo"],
        max_iter_per_start=cfg_dict["max_iter_per_start"],
        optimizer_method=cfg_dict["optimizer_method"],
        ftol=cfg_dict["ftol"],
        seed=cfg_dict["seed"],
        body_name=cfg_dict["body_name"],
        site_name=cfg_dict["site_name"],
    )

    x_opt = np.array(payload["x_opt"], dtype=np.float64)
    a_opt = np.array(payload["a_opt"], dtype=np.float64).reshape(
        config.num_joints, config.num_harmonics
    )
    b_opt = np.array(payload["b_opt"], dtype=np.float64).reshape(
        config.num_joints, config.num_harmonics
    )

    return OptimizationResult(
        x_opt=x_opt,
        condition_number=float(payload["condition_number"]),
        a_opt=a_opt,
        b_opt=b_opt,
        q0=np.array(payload["q0"], dtype=np.float64),
        config=config,
        n_evaluations=int(payload["n_evaluations"]),
        wall_time=float(payload["wall_time"]),
        n_restarts=int(payload["n_restarts"]),
        best_start_index=int(payload["best_start_index"]),
    )


def result_to_trajectory(
    result: OptimizationResult,
    fps: float | None = None,
) -> TrajectorySample:
    """Reconstruct a full trajectory from an OptimizationResult.

    Args:
        result: Optimization result containing Fourier coefficients.
        fps: Override sampling rate. If None, uses the optimization fps.
    """
    cfg = result.config
    output_fps = fps if fps is not None else cfg.fps
    return build_trajectory_from_params(
        result.x_opt,
        cfg.num_joints,
        cfg.num_harmonics,
        cfg.base_freq,
        cfg.duration,
        output_fps,
        result.q0,
    )


# UR5e joint names in URDF order
_UR5E_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def save_trajectory_json(
    trajectory: TrajectorySample,
    path: str | Path,
    *,
    condition_number: float | None = None,
    source: str | None = None,
) -> None:
    """Save a sampled trajectory as a self-contained JSON for real robot playback.

    The output JSON contains all joint positions, velocities, and accelerations
    at each timestep, so it can be loaded by a ROS node or similar system
    without requiring the Fourier trajectory generation code.

    Args:
        trajectory: Sampled trajectory with time, position, velocity, acceleration.
        path: Output file path.
        condition_number: Condition number from optimization (for metadata).
        source: Source file name (for metadata).
    """
    n_steps = len(trajectory.time)
    n_joints = trajectory.position.shape[1]
    duration = float(trajectory.time[-1] - trajectory.time[0])
    dt = float(trajectory.time[1] - trajectory.time[0]) if n_steps > 1 else 0.0
    fps = 1.0 / dt if dt > 0.0 else 0.0

    metadata: dict = {
        "description": "Sampled excitation trajectory for UR5e",
        "num_joints": n_joints,
        "num_steps": n_steps,
        "duration": duration,
        "fps": fps,
        "dt": dt,
        "joint_names": _UR5E_JOINT_NAMES[:n_joints],
    }
    if condition_number is not None:
        metadata["condition_number"] = condition_number
    if source is not None:
        metadata["source"] = source

    waypoints: list[dict] = []
    for i in range(n_steps):
        waypoints.append(
            {
                "t": round(float(trajectory.time[i]), 6),
                "q": [round(float(v), 8) for v in trajectory.position[i]],
                "dq": [round(float(v), 8) for v in trajectory.velocity[i]],
                "ddq": [round(float(v), 8) for v in trajectory.acceleration[i]],
            }
        )

    payload = {"metadata": metadata, "trajectory": waypoints}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
