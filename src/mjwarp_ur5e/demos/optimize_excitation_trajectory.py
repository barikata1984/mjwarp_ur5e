from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro

from mjwarp_ur5e.cli import OptimizeExcitationConfig
from mjwarp_ur5e.identification.collision import CollisionConfig
from mjwarp_ur5e.identification.io import save_optimization_result
from mjwarp_ur5e.identification.optimizer import ExcitationOptimizer, OptimizerConfig
from mjwarp_ur5e.identification.workspace import WorkspaceConstraintConfig
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
        workspace_config=workspace_config,
        collision_config=collision_config,
        payload_workspace_config=payload_workspace_config,
    )

    optimizer = ExcitationOptimizer(config=opt_config, model=loaded.model, data=loaded.data)

    print("Starting excitation trajectory optimization...")
    print(f"  harmonics={config.num_harmonics}, duration={config.duration}s")
    print(f"  monte-carlo restarts={config.n_monte_carlo}")
    print(f"  max_iter_per_start={config.max_iter}")
    result = optimizer.optimize()

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_optimization_result(result, output_path)

    print("\nOptimization complete:")
    print(f"  Condition number: {result.condition_number:.4f}")
    print(f"  Wall time: {result.wall_time:.1f}s")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Best start index: {result.best_start_index}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
