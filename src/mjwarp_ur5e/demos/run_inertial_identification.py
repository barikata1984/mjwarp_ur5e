"""Demo CLI for inertial parameter identification.

Full pipeline: load optimized trajectory -> playback -> estimate -> compare with truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mjwarp_ur5e.cli import IdentificationDemoConfig, load_config
from mjwarp_ur5e.identification.estimators import (
    BatchLeastSquares,
    BatchLSConfig,
    BatchTLSConfig,
    BatchTotalLeastSquares,
    RecursiveTotalLeastSquares,
    RTLSConfig,
)
from mjwarp_ur5e.identification.estimators.types import EstimationResult
from mjwarp_ur5e.identification.execution import PlaybackConfig, TrajectoryPlayback
from mjwarp_ur5e.identification.io import load_optimization_result, result_to_trajectory
from mjwarp_ur5e.identification.regressor import body_inertial_parameters_from_model
from mjwarp_ur5e.model import load_and_reset

_N_INERTIAL_PARAMS = 10


def _run_estimator(
    estimator_name: str,
    A: np.ndarray,
    y: np.ndarray,
    regularization: float,
) -> EstimationResult:
    """Run the selected estimator on regressor data."""
    if estimator_name == "ls":
        return BatchLeastSquares(BatchLSConfig(regularization=regularization)).estimate(A, y)
    elif estimator_name == "tls":
        return BatchTotalLeastSquares(BatchTLSConfig(regularization=regularization)).estimate(A, y)
    elif estimator_name == "rtls":
        rtls = RecursiveTotalLeastSquares(RTLSConfig())
        n_rows = A.shape[0]
        n_init = max(n_rows // 2, _N_INERTIAL_PARAMS + 1)
        n_init = min(n_init, n_rows)
        rtls.initialize(A[:n_init], y[:n_init])
        block_size = 6
        for i in range(n_init, n_rows, block_size):
            end = min(i + block_size, n_rows)
            rtls.update(A[i:end], y[i:end])
        return rtls.get_current_estimate()
    else:
        raise ValueError(f"Unknown estimator '{estimator_name}'. Choose 'ls', 'tls', or 'rtls'.")


def _print_comparison(true_params: np.ndarray, result: EstimationResult) -> None:
    """Print true vs estimated parameters with relative errors."""
    true_result = EstimationResult(phi=true_params)

    print("\n" + "=" * 70)
    print("IDENTIFICATION RESULTS")
    print("=" * 70)

    true_m, est_m = true_result.mass, result.mass
    rel_m = abs(est_m - true_m) / max(abs(true_m), 1e-15) * 100
    print("\n  Mass:")
    print(f"    True:      {true_m:.6f} kg")
    print(f"    Estimated: {est_m:.6f} kg")
    print(f"    Rel error: {rel_m:.4f}%")

    true_com, est_com = true_result.center_of_mass, result.center_of_mass
    print("\n  Center of mass [m]:")
    print(f"    True:      [{true_com[0]:.6f}, {true_com[1]:.6f}, {true_com[2]:.6f}]")
    print(f"    Estimated: [{est_com[0]:.6f}, {est_com[1]:.6f}, {est_com[2]:.6f}]")
    com_err = np.linalg.norm(est_com - true_com)
    com_ref = max(np.linalg.norm(true_com), 1e-15)
    print(f"    Abs error: {com_err:.6e} m")
    print(f"    Rel error: {com_err / com_ref * 100:.4f}%")

    true_I, est_I = true_result.inertia_at_com, result.inertia_at_com
    print("\n  Inertia at CoM [kg*m^2]:")
    print(f"    True diag:      [{true_I[0, 0]:.6e}, {true_I[1, 1]:.6e}, {true_I[2, 2]:.6e}]")
    print(f"    Estimated diag: [{est_I[0, 0]:.6e}, {est_I[1, 1]:.6e}, {est_I[2, 2]:.6e}]")
    I_err = np.linalg.norm(est_I - true_I, "fro")
    I_ref = max(np.linalg.norm(true_I, "fro"), 1e-15)
    print(f"    Frobenius error: {I_err:.6e}")
    print(f"    Rel error:       {I_err / I_ref * 100:.4f}%")

    print(f"\n  Condition number: {result.condition_number:.2f}")
    print(f"  Residual norm:    {result.residual_norm:.6e}")
    print(f"  Samples used:     {result.n_samples}")
    print("=" * 70)


def main() -> None:
    config = load_config(IdentificationDemoConfig)

    result_path = Path(config.result_json)
    if not result_path.exists():
        print(f"Error: optimization result not found: {result_path}")
        print("Run optimize_excitation_trajectory first to generate it.")
        sys.exit(1)

    print(f"Loading optimization result from {config.result_json}")
    opt_result = load_optimization_result(config.result_json)
    trajectory = result_to_trajectory(opt_result)
    print(f"Trajectory: {len(trajectory.time)} steps, duration={trajectory.time[-1]:.2f}s")

    loaded = load_and_reset(config.model or None)
    print(f"Loaded model: {loaded.model_path}")

    body_name = opt_result.config.body_name
    true_params_obj = body_inertial_parameters_from_model(loaded.model, body_name)
    true_phi = true_params_obj.to_vector()
    print(f"True parameters for '{body_name}': mass={true_params_obj.mass:.6f} kg")

    playback_config = PlaybackConfig(
        use_pd_control=False,
        noise_std_q=config.noise_std,
        noise_std_dq=config.noise_std,
        noise_std_wrench=config.noise_std,
        body_name=body_name,
        site_name=opt_result.config.site_name,
    )
    rng = np.random.default_rng(42) if config.noise_std > 0 else None

    playback = TrajectoryPlayback(loaded.model, loaded.data, playback_config)
    print("Executing trajectory playback...")
    buffer = playback.execute(trajectory, rng=rng)
    print(f"Collected {len(buffer)} samples")

    print("Building regressor matrices...")
    A, y_vec = buffer.build_regressor_data(loaded.model, loaded.data, body_name)
    print(f"Regressor shape: A={A.shape}, y={y_vec.shape}")

    print(f"Running estimator: {config.estimator}")
    result = _run_estimator(config.estimator, A, y_vec, config.regularization)

    _print_comparison(true_phi, result)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "estimator": config.estimator,
        "noise_std": config.noise_std,
        "regularization": config.regularization,
        "true_parameters": {
            "phi": true_phi.tolist(),
            "mass": float(true_phi[0]),
            "center_of_mass": (true_phi[1:4] / max(abs(true_phi[0]), 1e-15)).tolist(),
        },
        "estimated": result.to_dict(),
        "source_trajectory": config.result_json,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved identification result to {output_path}")


if __name__ == "__main__":
    main()
