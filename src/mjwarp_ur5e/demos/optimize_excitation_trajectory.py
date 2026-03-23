from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjwarp_ur5e.cli import OptimizeExcitationConfig, load_config
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
    config = load_config(OptimizeExcitationConfig)

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

    # Joint limits
    joint_limits: JointLimits | None = None
    has_vel = config.dq_max > 0
    has_acc = config.ddq_max > 0
    if has_vel or has_acc:
        from mjwarp_ur5e.identification.constraints import JointLimits

        kwargs: dict = {}
        if has_vel:
            kwargs["dq_max"] = np.full(6, config.dq_max)
        if has_acc:
            kwargs["ddq_max"] = np.full(6, config.ddq_max)
        joint_limits = JointLimits(**kwargs)

    # Fourier bounds only apply when velocity/acceleration limits exist
    use_fourier_bounds = config.use_fourier_bounds and (has_vel or has_acc)

    # When Fourier bounds are enabled, disable the per-timestep velocity constraint
    # (the bounds already guarantee velocity feasibility structurally)
    enable_vel_constraint = has_vel and not use_fourier_bounds

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
        objective_type=config.objective,
        seed=config.seed,
        joint_limits=joint_limits,
        workspace_config=workspace_config,
        collision_config=collision_config,
        payload_workspace_config=payload_workspace_config,
        ee_velocity_config=ee_velocity_config,
        enable_velocity_constraint=enable_vel_constraint,
        enable_acceleration_constraint=has_acc,
        use_fourier_bounds=use_fourier_bounds,
        with_ft_offset=config.with_ft_offset,
        ft_offset_column_scale=config.ft_offset_column_scale,
        n_workers=config.n_workers,
        model_path=str(loaded.model_path) if config.n_workers > 1 else None,
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
    print(f"  objective={config.objective}", flush=True)
    print(f"  harmonics={config.num_harmonics}, duration={config.duration}s", flush=True)
    print(f"  monte-carlo restarts={config.n_monte_carlo}", flush=True)
    print(f"  max_iter_per_start={config.max_iter}", flush=True)
    if config.n_workers > 1:
        print(f"  parallel workers={config.n_workers}", flush=True)
    if config.wandb:
        print(f"  wandb: project={config.wandb_project}", flush=True)
    if config.ee_max_linear_velocity > 0:
        print(f"  EE velocity limit: {config.ee_max_linear_velocity} m/s", flush=True)
    if has_vel:
        print(f"  joint velocity limit: {config.dq_max} rad/s (all joints)", flush=True)
    if has_acc:
        print(f"  joint acceleration limit: {config.ddq_max} rad/s^2 (all joints)", flush=True)
    if use_fourier_bounds:
        print(
            "  Fourier coefficient bounds: ENABLED (velocity constraint via box bounds)", flush=True
        )
    if config.with_ft_offset:
        scale_str = "column-scaled" if config.ft_offset_column_scale else "unscaled"
        print(f"  FT sensor offset estimation: ENABLED (16 params, {scale_str})", flush=True)
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
    print(f"  Feasible: {result.feasible}")
    print(f"  Wall time: {result.wall_time:.1f}s")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Best start index: {result.best_start_index}")
    if result.constraint_margins:
        violated = {k: v for k, v in result.constraint_margins.items() if v < 0}
        satisfied = {k: v for k, v in result.constraint_margins.items() if v >= 0}
        print("  Constraint margins:")
        for name, margin in satisfied.items():
            print(f"    {name}: {margin:.6f}  [OK]")
        if violated:
            print(f"  VIOLATED constraints ({len(violated)}/{len(result.constraint_margins)}):")
            for name, margin in violated.items():
                print(f"    {name}: {margin:.6f}  (violation = {-margin:.6f})")
    if result.trajectory_stats:
        print("  Trajectory stats:")
        for key, val in result.trajectory_stats.items():
            if isinstance(val, list):
                formatted = ", ".join(f"{v:.4f}" for v in val)
                print(f"    {key}: [{formatted}]")
            else:
                print(f"    {key}: {val:.4f}")
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
