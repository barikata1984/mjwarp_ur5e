"""Run inertial parameter identification on sim wrench from replay_trajectory_video.py.

Loads the real joint trajectory (recording.npz) and the MuJoCo-simulated wrench
(replay_ft.npz), feeds them into IdentificationPipeline, and prints the results
with a sim-vs-real comparison table.

Usage:
    python scripts/identify_from_sim.py
    python scripts/identify_from_sim.py --recording data/replay_recording_*/recording.npz
    python scripts/identify_from_sim.py --wrench results/replay/replay_ft.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

IPARAM_SRC = Path(__file__).resolve().parent.parent / "iparam_identification" / "src"
sys.path.insert(0, str(IPARAM_SRC))

from identifiers.pipeline import IdentificationPipeline, IdentificationResult
from identifiers.tls import ScalingMode
from utilities.identification_utils import PARAM_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IPARAM_ROOT = PROJECT_ROOT / "iparam_identification"
URDF_PATH = str(IPARAM_ROOT / "data" / "urdf" / "ur5e_ft300s_robotiq85.urdf")
FRAME_NAME = "robotiq_ft_frame_id"
DEFAULT_GRIPPER_CAL = IPARAM_ROOT / "data" / "calibration" / "gripper.json"
METHODS = ["OLS", "TLS", "OLS+bias", "TLS+bias"]


def find_latest_recording() -> Path:
    data_dir = Path("data")
    dirs = sorted(data_dir.glob("replay_recording_*"))
    if not dirs:
        sys.exit("No replay_recording_* directories found under data/")
    return dirs[-1] / "recording.npz"


def _params_dict(result: IdentificationResult) -> dict[str, np.ndarray]:
    return {
        "OLS": result.ols,
        "TLS": result.tls.x,
        "OLS+bias": result.ols_bias,
        "TLS+bias": result.tls_bias.x[:10],
    }


def print_result(result: IdentificationResult, label: str) -> None:
    all_params = _params_dict(result)

    print()
    print("=" * 76)
    print(f"  {label}")
    print("=" * 76)
    print(f"  Frames used: {result.meta['trimmed_frames']}")
    print(f"  TLS scaling: {result.meta['tls_scaling']}")
    print(f"  Torque weight: {result.meta['torque_weight']:.4f}")
    print()
    header = f"  {'param':10s}" + "".join(f" {m:>14s}" for m in METHODS)
    print(header)
    print("  " + "-" * (10 + 15 * len(METHODS)))
    for i, name in enumerate(PARAM_NAMES):
        row = f"  {name:10s}"
        for m in METHODS:
            row += f" {all_params[m][i]:>14.6f}"
        print(row)

    print()
    bias_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    print(f"  {'bias':10s} {'OLS+bias':>14s} {'TLS+bias':>14s}")
    print("  " + "-" * 40)
    for i, bl in enumerate(bias_labels):
        unit = "N" if i < 3 else "Nm"
        print(f"  {bl:10s} {result.bias_ols[i]:>13.4f}{unit} {result.bias_tls[i]:>13.4f}{unit}")
    print("=" * 76)

    if result.object_params is not None:
        print()
        print("  OBJECT INERTIA (difference method)")
        print("  " + "-" * 60)
        available = [m for m in METHODS if m in result.object_params]
        header = f"  {'param':10s}" + "".join(f" {m:>14s}" for m in available)
        print(header)
        print("  " + "-" * (10 + 15 * len(available)))
        for i, name in enumerate(PARAM_NAMES):
            row = f"  {name:10s}"
            for m in available:
                row += f" {result.object_params[m][i]:>14.6f}"
            print(row)
        print("=" * 76)


def print_comparison(
    result_sim: IdentificationResult,
    result_real: IdentificationResult,
    method: str = "OLS+bias",
) -> None:
    sim_obj = result_sim.object_params
    real_obj = result_real.object_params
    if sim_obj is None or real_obj is None:
        return
    if method not in sim_obj or method not in real_obj:
        return

    sim_p = sim_obj[method]
    real_p = real_obj[method]
    diff = sim_p - real_p

    print()
    print("=" * 76)
    print(f"  OBJECT INERTIA COMPARISON: sim vs real ({method})")
    print("=" * 76)
    header = f"  {'param':10s} {'sim':>12s} {'real':>12s} {'diff':>12s} {'rel%':>10s}"
    print(header)
    print("  " + "-" * 58)
    for i, name in enumerate(PARAM_NAMES):
        rel = diff[i] / real_p[i] * 100 if abs(real_p[i]) > 1e-8 else float("nan")
        print(f"  {name:10s} {sim_p[i]:>12.6f} {real_p[i]:>12.6f} {diff[i]:>+12.6f} {rel:>+9.1f}%")
    print("=" * 76)

    # Also compare total params
    sim_total = _params_dict(result_sim)[method]
    real_total = _params_dict(result_real)[method]
    diff_total = sim_total - real_total

    print()
    print(f"  TOTAL INERTIA COMPARISON: sim vs real ({method})")
    print("  " + "-" * 58)
    header = f"  {'param':10s} {'sim':>12s} {'real':>12s} {'diff':>12s} {'rel%':>10s}"
    print(header)
    print("  " + "-" * 58)
    for i, name in enumerate(PARAM_NAMES):
        rel = diff_total[i] / real_total[i] * 100 if abs(real_total[i]) > 1e-8 else float("nan")
        print(
            f"  {name:10s} {sim_total[i]:>12.6f} {real_total[i]:>12.6f}"
            f" {diff_total[i]:>+12.6f} {rel:>+9.1f}%"
        )
    print("=" * 76)


def result_to_dict(result: IdentificationResult) -> dict:
    out = {
        "meta": result.meta,
        "results": {},
        "bias": {
            "ols": result.bias_ols.tolist(),
            "tls": result.bias_tls.tolist(),
        },
    }
    all_params = _params_dict(result)
    for m in METHODS:
        out["results"][m] = {"params": all_params[m].tolist()}
    if result.object_params is not None:
        out["object_results"] = {m: {"params": p.tolist()} for m, p in result.object_params.items()}
    return out


def save_results(
    result_sim: IdentificationResult,
    result_real: IdentificationResult,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "sim": result_to_dict(result_sim),
        "real": result_to_dict(result_real),
    }
    out_path = out_dir / "identification_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return out_path


def save_comparison_csv(
    result_sim: IdentificationResult,
    result_real: IdentificationResult,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "identification_comparison.csv"

    sim_params = _params_dict(result_sim)
    real_params = _params_dict(result_real)
    param_units = [
        "kg",
        "kg·m",
        "kg·m",
        "kg·m",
        "kg·m²",
        "kg·m²",
        "kg·m²",
        "kg·m²",
        "kg·m²",
        "kg·m²",
    ]
    bias_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    bias_units = ["N", "N", "N", "Nm", "Nm", "Nm"]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "", "Real", "", "", "", "Sim", "", "", ""])
        w.writerow(["param", "unit"] + METHODS + METHODS)

        w.writerow(["--- TOTAL ---"] + [""] * 9)
        for i, name in enumerate(PARAM_NAMES):
            row = [name, param_units[i]]
            for m in METHODS:
                row.append(real_params[m][i])
            for m in METHODS:
                row.append(sim_params[m][i])
            w.writerow(row)

        if result_real.object_params is not None and result_sim.object_params is not None:
            w.writerow(["--- OBJECT ---"] + [""] * 9)
            for i, name in enumerate(PARAM_NAMES):
                row = [name, param_units[i]]
                for m in METHODS:
                    row.append(result_real.object_params.get(m, np.zeros(10))[i])
                for m in METHODS:
                    row.append(result_sim.object_params.get(m, np.zeros(10))[i])
                w.writerow(row)

        w.writerow(["--- BIAS ---"] + [""] * 9)
        for i, bl in enumerate(bias_labels):
            row = [bl, bias_units[i], "", ""]
            row.append(result_real.bias_ols[i])
            row.append(result_real.bias_tls[i])
            row.extend(["", ""])
            row.append(result_sim.bias_ols[i])
            row.append(result_sim.bias_tls[i])
            w.writerow(row)

    return out_path


def load_gripper_cal(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with open(path) as f:
        cal = json.load(f)
    if "methods" in cal and "OLS+bias" in cal["methods"]:
        return np.array(cal["methods"]["OLS+bias"]["params"])
    return None


def run_pipeline(
    q: np.ndarray,
    dq: np.ndarray,
    time_arr: np.ndarray,
    wrench: np.ndarray,
    gripper_cal: np.ndarray | None,
) -> IdentificationResult:
    pipeline = IdentificationPipeline(
        urdf_path=URDF_PATH,
        frame_name=FRAME_NAME,
        tls_scaling=ScalingMode.NOISE_VARIANCE,
    )
    for i in range(len(q)):
        pipeline.process_frame(q[i], dq[i], time_arr[i], wrench[i])
    return pipeline.identify(gripper_cal=gripper_cal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        default=None,
        help="Path to recording.npz (default: latest under data/)",
    )
    parser.add_argument(
        "--wrench",
        type=Path,
        default=Path("results/replay/replay_ft.npz"),
        help="Path to replay_ft.npz with sim wrench",
    )
    parser.add_argument(
        "--gripper-cal",
        type=Path,
        default=DEFAULT_GRIPPER_CAL,
        help="Gripper calibration JSON for difference method",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replay"),
        help="Directory to save identification_result.json",
    )
    parser.add_argument(
        "--method",
        default="OLS+bias",
        choices=METHODS,
        help="Method to use for comparison table",
    )
    args = parser.parse_args()

    recording_path = args.recording or find_latest_recording()
    print(f"Recording:    {recording_path}")
    print(f"Sim wrench:   {args.wrench}")
    print(f"Gripper cal:  {args.gripper_cal}")

    rec = np.load(str(recording_path))
    q = rec["joint_position"]
    dq = rec["joint_velocity"]
    time_arr = rec["time"] - rec["time"][0]

    ft = np.load(str(args.wrench))
    wrench_sim = ft["wrench_sim"]
    if len(wrench_sim) != len(q):
        sys.exit(f"Length mismatch: recording has {len(q)} frames, wrench has {len(wrench_sim)}")

    wrench_real = rec["wrench"]
    wrench_real_tared = wrench_real - wrench_real[0]

    gripper_cal = load_gripper_cal(args.gripper_cal)
    if gripper_cal is not None:
        print(f"  Gripper cal loaded ({args.gripper_cal})")
    else:
        print("  No gripper calibration (object inertia will not be computed)")

    print(f"Frames: {len(q)}, duration: {time_arr[-1]:.2f}s")
    print()

    # --- Sim identification ---
    result_sim = run_pipeline(q, dq, time_arr, wrench_sim, gripper_cal)
    print_result(result_sim, f"SIM: {ft['note']}")

    # --- Real identification ---
    result_real = run_pipeline(q, dq, time_arr, wrench_real_tared, gripper_cal)
    print_result(result_real, "REAL (frame0-tared)")

    # --- Comparison ---
    print_comparison(result_sim, result_real, method=args.method)

    # --- Save ---
    out_path = save_results(result_sim, result_real, args.output_dir)
    csv_path = save_comparison_csv(result_sim, result_real, args.output_dir)
    print(f"\nResults saved to: {out_path}")
    print(f"Comparison CSV:   {csv_path}")


if __name__ == "__main__":
    main()
