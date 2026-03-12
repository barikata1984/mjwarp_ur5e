from __future__ import annotations

import numpy as np
import tyro

from mjwarp_ur5e.cli import ValidateExcitationConfig
from mjwarp_ur5e.identification.io import (
    load_optimization_result,
    result_to_trajectory,
)
from mjwarp_ur5e.identification.optimizer import ExcitationOptimizer
from mjwarp_ur5e.model import load_model, reset_to_home


def main() -> None:
    config = tyro.cli(ValidateExcitationConfig)

    result = load_optimization_result(config.result_json)

    model_path = config.model if config.model else None
    if model_path is None:
        model_path = "assets/ur5e/mjcf/scene_with_box.xml"

    loaded = load_model(model_path)
    reset_to_home(loaded.model, loaded.data)

    optimizer = ExcitationOptimizer(
        config=result.config,
        model=loaded.model,
        data=loaded.data,
    )

    print(f"Validating: {config.result_json}")
    report = optimizer.validate_trajectory(result)

    print("\nValidation report:")
    print(f"  Condition number (full resolution): {report['condition_number']:.4f}")
    print(f"  All constraints satisfied: {report['all_constraints_satisfied']}")

    margins = report["constraint_margins"]
    for i, m in enumerate(margins):
        status = "OK" if m >= 0 else "VIOLATED"
        print(f"  Constraint {i}: margin={m:.6f} [{status}]")

    traj = result_to_trajectory(result)
    q = traj.position
    dq = traj.velocity
    ddq = traj.acceleration

    print("\nTrajectory summary:")
    print(f"  Timesteps: {q.shape[0]}")
    print(f"  q  range: [{np.min(q):.4f}, {np.max(q):.4f}] rad")
    print(f"  dq range: [{np.min(dq):.4f}, {np.max(dq):.4f}] rad/s")
    print(f"  ddq range: [{np.min(ddq):.4f}, {np.max(ddq):.4f}] rad/s^2")


if __name__ == "__main__":
    main()
