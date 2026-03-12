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
) -> TrajectorySample:
    """Reconstruct a full trajectory from an OptimizationResult."""
    cfg = result.config
    return build_trajectory_from_params(
        result.x_opt,
        cfg.num_joints,
        cfg.num_harmonics,
        cfg.base_freq,
        cfg.duration,
        cfg.fps,
        result.q0,
    )
