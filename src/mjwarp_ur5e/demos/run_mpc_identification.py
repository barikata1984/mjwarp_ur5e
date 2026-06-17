"""Demo CLI for MPC-based inertial parameter identification.

Closed-loop excitation: plan -> execute -> identify -> replan until convergence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mjwarp_ur5e.cli import MPCIdentificationConfig, load_config
from mjwarp_ur5e.identification.mpc.config import HorizonConfig, MPCConfig, PlannerConfig
from mjwarp_ur5e.identification.mpc.loop import MPCLoop, MPCResult
from mjwarp_ur5e.identification.regressor import body_inertial_parameters_from_model
from mjwarp_ur5e.model import load_and_reset

_PARAM_NAMES = ["m", "hx", "hy", "hz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]


def _save_result(result: MPCResult, true_vec: np.ndarray, path: str) -> None:
    out = {
        "converged": result.converged,
        "total_wall_time": result.total_wall_time,
        "total_samples": result.total_samples,
        "final_condition_number": result.final_condition_number,
        "true_params": true_vec.tolist(),
        "estimated_params": result.final_estimation.phi.tolist(),
        "steps": [
            {
                "step": s.step,
                "condition_number": s.condition_number_accumulated,
                "mass": s.estimation.mass,
                "n_samples": s.n_samples_total,
                "wall_time_plan": s.wall_time_plan,
                "wall_time_execute": s.wall_time_execute,
            }
            for s in result.steps
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


def main() -> None:
    config = load_config(MPCIdentificationConfig)
    loaded = load_and_reset(config.model)

    true_params = body_inertial_parameters_from_model(loaded.model, config.body_name)
    true_vec = true_params.to_vector()
    print(f"Ground truth: mass={true_params.mass:.4f} kg")

    mpc_config = MPCConfig(
        horizon=HorizonConfig(
            duration=config.horizon_duration,
            num_segments=config.num_segments,
            fps=config.fps,
            subsample_factor=config.subsample_factor,
        ),
        planner=PlannerConfig(
            n_restarts=config.n_restarts,
            max_iter_per_start=config.max_iter,
            seed=config.seed,
            waypoint_perturbation=config.waypoint_perturbation,
        ),
        replan_period=config.replan_period,
        max_mpc_steps=config.max_mpc_steps,
        convergence_threshold=config.convergence_threshold,
        body_name=config.body_name,
        site_name=config.site_name,
        model_path=config.model,
        use_pd_control=config.use_pd_control,
        noise_std_wrench=config.noise_std_wrench,
    )

    mpc = MPCLoop(mpc_config, loaded.model, loaded.data)
    result = mpc.run()

    print(f"\n{'Step':>4} {'Cond':>10} {'Mass':>8} {'Samples':>8} {'Plan[s]':>8} {'Exec[s]':>8}")
    print("-" * 56)
    for s in result.steps:
        print(
            f"{s.step:>4d} {s.condition_number_accumulated:>10.2f} "
            f"{s.estimation.mass:>8.4f} {s.n_samples_total:>8d} "
            f"{s.wall_time_plan:>8.1f} {s.wall_time_execute:>8.1f}"
        )

    est = result.final_estimation
    print(f"\nConverged: {result.converged}")
    print(f"Total time: {result.total_wall_time:.1f}s")
    print(f"\n{'Param':>8} {'True':>12} {'Estimated':>12} {'Error':>12} {'Rel%':>8}")
    print("-" * 56)
    for i, name in enumerate(_PARAM_NAMES):
        t, e = true_vec[i], est.phi[i]
        err = e - t
        rel = err / t * 100 if abs(t) > 1e-10 else float("nan")
        print(f"{name:>8} {t:>12.6f} {e:>12.6f} {err:>+12.6f} {rel:>+7.1f}%")

    _save_result(result, true_vec, config.output)

    traj_path = Path(config.output).with_suffix(".npz")
    q_all = result.executed_trajectory_q()
    np.savez(str(traj_path), q=q_all, fps=config.fps)
    print(f"Trajectory saved to {traj_path} ({len(q_all)} frames)")


if __name__ == "__main__":
    main()
