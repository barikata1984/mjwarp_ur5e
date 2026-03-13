from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro

from mjwarp_ur5e.cli import OptimizeExcitationConfig
from mjwarp_ur5e.identification.collision import CollisionConfig
from mjwarp_ur5e.identification.io import (
    result_to_trajectory,
    save_optimization_result,
    save_trajectory_json,
)
from mjwarp_ur5e.identification.optimizer import (
    EarlyStopConfig,
    ExcitationOptimizer,
    OptimizerConfig,
    WandbConfig,
)
from mjwarp_ur5e.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig
from mjwarp_ur5e.model import get_named_object_id, load_and_reset


def main() -> None:
    config = tyro.cli(OptimizeExcitationConfig)

    loaded = load_and_reset(config.model or None)
    q0 = np.array(loaded.data.qpos[: loaded.model.nq], dtype=np.float64)

    workspace_config: WorkspaceConstraintConfig | None = None
    if config.max_displacement > 0:
        workspace_config = WorkspaceConstraintConfig(max_displacement=config.max_displacement)

    collision_config: CollisionConfig | None = None
    if config.enable_collision:
        collision_config = CollisionConfig()

    payload_workspace_config: WorkspaceConstraintConfig | None = None
    if config.enable_payload_workspace:
        geom_id = get_named_object_id(
            loaded.model, mujoco.mjtObj.mjOBJ_GEOM, "workspace_region_geom"
        )
        if geom_id is not None:
            body_id = loaded.model.geom_bodyid[geom_id]
            mujoco.mj_kinematics(loaded.model, loaded.data)
            center = loaded.data.xpos[body_id].copy()
            half = loaded.model.geom_size[geom_id].copy()
            box_lower = center - half
            box_upper = center + half
            print(f"  payload workspace bounds: {box_lower} .. {box_upper}")
            payload_workspace_config = WorkspaceConstraintConfig(
                box_lower=box_lower, box_upper=box_upper
            )

    # EE velocity constraint
    ee_velocity_config: EeVelocityConfig | None = None
    if config.ee_max_linear_velocity > 0:
        ee_velocity_config = EeVelocityConfig(max_linear_velocity=config.ee_max_linear_velocity)

    # Joint velocity override
    joint_limits: JointLimits | None = None
    if config.dq_max > 0:
        from mjwarp_ur5e.identification.constraints import JointLimits

        joint_limits = JointLimits(
            dq_max=np.full(6, config.dq_max),
        )

    opt_config = OptimizerConfig(
        num_joints=6,
        num_harmonics=config.num_harmonics,
        base_freq=config.base_freq,
        duration=config.duration,
        fps=config.fps,
        q0=q0,
        subsample_factor=config.subsample_factor,
        n_monte_carlo=config.n_monte_carlo,
        max_iter_per_start=config.max_iter,
        seed=config.seed,
        joint_limits=joint_limits,
        workspace_config=workspace_config,
        collision_config=collision_config,
        payload_workspace_config=payload_workspace_config,
        ee_velocity_config=ee_velocity_config,
    )

    optimizer = ExcitationOptimizer(config=opt_config, model=loaded.model, data=loaded.data)

    wandb_cfg = WandbConfig(
        enabled=config.wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name,
    )
    early_stop_cfg = EarlyStopConfig(
        enabled=config.early_stop,
        patience=config.early_stop_patience,
        target_cond=config.early_stop_target_cond,
    )

    print("Starting excitation trajectory optimization...", flush=True)
    print(f"  harmonics={config.num_harmonics}, duration={config.duration}s", flush=True)
    print(f"  monte-carlo restarts={config.n_monte_carlo}", flush=True)
    print(f"  max_iter_per_start={config.max_iter}", flush=True)
    if config.wandb:
        print(f"  wandb: project={config.wandb_project}", flush=True)
    if config.ee_max_linear_velocity > 0:
        print(f"  EE velocity limit: {config.ee_max_linear_velocity} m/s", flush=True)
    if config.dq_max > 0:
        print(f"  joint velocity limit: {config.dq_max} rad/s (all joints)", flush=True)
    if config.early_stop:
        msg = f"  early stopping: patience={config.early_stop_patience}"
        if config.early_stop_target_cond > 0:
            msg += f", target_cond={config.early_stop_target_cond}"
        print(msg, flush=True)
    result = optimizer.optimize(wandb_config=wandb_cfg, early_stop_config=early_stop_cfg)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_optimization_result(result, output_path)

    print("\nOptimization complete:")
    print(f"  Condition number: {result.condition_number:.4f}")
    print(f"  Wall time: {result.wall_time:.1f}s")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Best start index: {result.best_start_index}")
    print(f"  Output: {output_path}")

    # Export sampled trajectory JSON for real robot playback
    traj_fps = config.trajectory_fps if config.trajectory_fps > 0 else None
    trajectory = result_to_trajectory(result, fps=traj_fps)
    traj_path = Path(config.trajectory_output)
    save_trajectory_json(
        trajectory,
        traj_path,
        condition_number=result.condition_number,
        source=output_path.name,
    )
    effective_fps = traj_fps if traj_fps else config.fps
    print(f"  Trajectory JSON: {traj_path} ({effective_fps:.0f} Hz, {len(trajectory.time)} steps)")


if __name__ == "__main__":
    main()
