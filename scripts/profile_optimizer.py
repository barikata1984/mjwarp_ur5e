"""Profile 1 SLSQP iteration to identify the optimization bottleneck.

Measures:
  - Trajectory generation (cache miss)
  - Each constraint evaluation individually
  - Objective function (regressor + condition number)
  - Simulated full iteration (objective + all constraints × (1 + n_vars) for finite differences)
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from mjwarp_ur5e.identification.collision import CollisionConfig
from mjwarp_ur5e.identification.constraints import (
    JointLimits,
    _TrajectoryCache,
    build_scipy_constraints,
    make_joint_acceleration_constraint,
    make_joint_position_constraint,
    make_joint_velocity_constraint,
)
from mjwarp_ur5e.identification.objective import condition_number_objective
from mjwarp_ur5e.identification.workspace import (
    EeVelocityConfig,
    WorkspaceConstraintConfig,
    make_ee_velocity_constraint,
    make_payload_workspace_constraint,
    make_workspace_constraint,
)
from mjwarp_ur5e.model import get_named_object_id, load_and_reset


def main() -> None:
    # --- Setup (same as optimize_excitation_trajectory with current defaults) ---
    loaded = load_and_reset(None)
    model, data = loaded.model, loaded.data
    q0 = np.array(data.qpos[: model.nq], dtype=np.float64)

    num_joints = 6
    num_harmonics = 3
    base_freq = 1.0 / 3.0
    duration = 3.0
    fps = 100.0
    dq_max = 0.0873  # 5 deg/s

    joint_limits = JointLimits(dq_max=np.full(6, dq_max))

    workspace_config = WorkspaceConstraintConfig(max_displacement=0.5)

    geom_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "workspace_region_geom")
    payload_workspace_config = None
    if geom_id is not None:
        body_id = model.geom_bodyid[geom_id]
        mujoco.mj_kinematics(model, data)
        center = data.xpos[body_id].copy()
        half = model.geom_size[geom_id].copy()
        payload_workspace_config = WorkspaceConstraintConfig(
            box_lower=center - half, box_upper=center + half
        )

    collision_config = CollisionConfig()
    ee_velocity_config = EeVelocityConfig(max_linear_velocity=0.25)

    n_vars = 2 * num_joints * num_harmonics
    n_timesteps = int(duration * fps) + 1
    print(f"Decision variables: {n_vars}")
    print(f"Timesteps: {n_timesteps}")
    print(f"FD evals per iteration: {1 + n_vars} = {1 + n_vars}")
    print()

    # --- Build cache and individual constraint functions ---
    cache = _TrajectoryCache(
        num_joints=num_joints,
        num_harmonics=num_harmonics,
        base_freq=base_freq,
        duration=duration,
        fps=fps,
        q0=q0,
    )

    rng = np.random.default_rng(42)
    x0 = np.zeros(n_vars, dtype=np.float64)
    for k in range(num_harmonics):
        scale = 0.3 / (k + 1)
        x0[k * num_joints : (k + 1) * num_joints] = rng.uniform(-scale, scale, size=num_joints)
        x0[
            num_joints * num_harmonics + k * num_joints : num_joints * num_harmonics
            + (k + 1) * num_joints
        ] = rng.uniform(-scale, scale, size=num_joints)

    # --- Profile individual components ---
    print("=" * 60)
    print("SINGLE EVALUATION TIMING (1 call each)")
    print("=" * 60)

    # Trajectory generation (cache miss)
    cache._cache_key = None  # force miss
    t0 = time.perf_counter()
    sample = cache.get(x0)
    t_traj = time.perf_counter() - t0
    print(f"  trajectory generation:    {t_traj * 1000:8.2f} ms")

    # Joint position constraint (pure NumPy)
    fn_pos = make_joint_position_constraint(cache, joint_limits)
    t0 = time.perf_counter()
    fn_pos(x0)
    t_pos = time.perf_counter() - t0
    print(f"  joint position constraint: {t_pos * 1000:8.2f} ms")

    # Joint velocity constraint (pure NumPy)
    fn_vel = make_joint_velocity_constraint(cache, joint_limits)
    t0 = time.perf_counter()
    fn_vel(x0)
    t_vel = time.perf_counter() - t0
    print(f"  joint velocity constraint: {t_vel * 1000:8.2f} ms")

    # Joint acceleration constraint (pure NumPy)
    fn_acc = make_joint_acceleration_constraint(cache, joint_limits)
    t0 = time.perf_counter()
    fn_acc(x0)
    t_acc = time.perf_counter() - t0
    print(f"  joint accel constraint:    {t_acc * 1000:8.2f} ms")

    # Workspace constraint (FK loop)
    fn_ws = make_workspace_constraint(cache, workspace_config, model, data)
    cache._cache_key = None
    t0 = time.perf_counter()
    fn_ws(x0)
    t_ws = time.perf_counter() - t0
    print(f"  workspace constraint:      {t_ws * 1000:8.2f} ms  (FK × {n_timesteps})")

    # Payload workspace constraint (FK loop)
    if payload_workspace_config is not None:
        fn_pws = make_payload_workspace_constraint(
            cache, payload_workspace_config, model, data, "payload_box_mount"
        )
        cache._cache_key = None
        t0 = time.perf_counter()
        fn_pws(x0)
        t_pws = time.perf_counter() - t0
        print(f"  payload workspace:         {t_pws * 1000:8.2f} ms  (FK × {n_timesteps})")
    else:
        t_pws = 0.0
        print("  payload workspace:         SKIPPED (no workspace_region_geom)")

    # Collision constraint (FK loop)
    from mjwarp_ur5e.identification.collision import CollisionChecker, make_collision_constraint

    checker = CollisionChecker(model, data, collision_config)
    fn_coll = make_collision_constraint(cache, checker)
    cache._cache_key = None
    t0 = time.perf_counter()
    fn_coll(x0)
    t_coll = time.perf_counter() - t0
    print(f"  collision constraint:      {t_coll * 1000:8.2f} ms  (FK × {n_timesteps})")

    # EE velocity constraint (FK + Jacobian loop)
    fn_eev = make_ee_velocity_constraint(cache, ee_velocity_config, model, data)
    cache._cache_key = None
    t0 = time.perf_counter()
    fn_eev(x0)
    t_eev = time.perf_counter() - t0
    print(f"  EE velocity constraint:    {t_eev * 1000:8.2f} ms  (FK+Jac × {n_timesteps})")

    # Objective function (regressor + cond number)
    cache._cache_key = None
    t0 = time.perf_counter()
    cond = condition_number_objective(x0, cache, model, data, "payload_box_mount", 1)
    t_obj = time.perf_counter() - t0
    print(f"  objective (cond number):   {t_obj * 1000:8.2f} ms  (regressor × {n_timesteps})")
    print(f"    -> condition number = {cond:.2f}")

    # --- Build all constraints via build_scipy_constraints ---
    all_constraints = build_scipy_constraints(
        cache,
        joint_limits,
        workspace_config=workspace_config,
        collision_config=collision_config,
        model=model,
        data=data,
        payload_workspace_config=payload_workspace_config,
        payload_body_name="payload_box_mount",
        ee_velocity_config=ee_velocity_config,
        site_name="attachment_site",
    )

    # Time one full evaluation of all constraints
    cache._cache_key = None
    t0 = time.perf_counter()
    for c in all_constraints:
        c["fun"](x0)
    t_all_constraints = time.perf_counter() - t0
    print(f"\n  all {len(all_constraints)} constraints total: {t_all_constraints * 1000:8.2f} ms")

    # --- Simulate 1 SLSQP iteration ---
    print()
    print("=" * 60)
    print("SIMULATED 1 SLSQP ITERATION")
    print(f"  = 1 objective + {len(all_constraints)} constraints × (1 + {n_vars}) FD evals")
    print("=" * 60)

    n_fd = 1 + n_vars  # 1 base + n_vars finite difference perturbations
    eps = 1e-8

    t0 = time.perf_counter()

    # Base evaluation
    cache._cache_key = None
    condition_number_objective(x0, cache, model, data, "payload_box_mount", 1)
    for c in all_constraints:
        cache._cache_key = None
        c["fun"](x0)

    # Finite difference evaluations
    for j in range(n_vars):
        x_pert = x0.copy()
        x_pert[j] += eps
        cache._cache_key = None
        condition_number_objective(x_pert, cache, model, data, "payload_box_mount", 1)
        for c in all_constraints:
            cache._cache_key = None
            c["fun"](x_pert)

    t_iter = time.perf_counter() - t0
    print(f"\n  Total 1 iteration: {t_iter:.2f}s ({t_iter / 60:.1f} min)")

    # Breakdown
    t_single = t_obj + t_all_constraints
    t_estimated = t_single * n_fd
    print(f"  Estimated from single evals: {t_estimated:.2f}s")
    print(f"  Ratio actual/estimated: {t_iter / t_estimated:.2f}x")

    # Per-component breakdown for 1 iteration
    print()
    print("Breakdown per iteration (estimated, seconds):")
    t_numpy_per_call = t_pos + t_vel + t_acc
    t_fk_per_call = t_ws + t_pws + t_coll + t_eev
    print(f"  NumPy constraints (pos+vel+acc):   {t_numpy_per_call * n_fd * 1000:8.1f} ms")
    print(f"  FK constraints (ws+pws+coll+eev):  {t_fk_per_call * n_fd:8.2f} s")
    print(f"  Objective:                          {t_obj * n_fd:8.2f} s")
    print(f"  Trajectory generation (cache miss): {t_traj * n_fd:8.2f} s")
    pct_fk = t_fk_per_call / (t_fk_per_call + t_obj + t_numpy_per_call) * 100
    pct_obj = t_obj / (t_fk_per_call + t_obj + t_numpy_per_call) * 100
    print(f"\n  FK constraints: {pct_fk:.1f}% of compute")
    print(f"  Objective:      {pct_obj:.1f}% of compute")

    # How many iterations per hour?
    iters_per_hour = 3600.0 / t_iter if t_iter > 0 else float("inf")
    print(f"\n  Iterations per hour: ~{iters_per_hour:.0f}")
    print(f"  200 iterations would take: ~{200 * t_iter / 3600:.1f} hours")


if __name__ == "__main__":
    main()
