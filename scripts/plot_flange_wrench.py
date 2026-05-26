"""Plot flange wrench (force/torque in tool0 frame) from a playback npz.

The wrench is read from the MuJoCo force/torque sensors at the ``ft_sensor``
site, which sits at the ``payload_box_mount`` body origin and is aligned with
the ``attachment_site`` (tool0) frame. The sensor convention orders the wrench
as ``[Fx, Fy, Fz, Mx, My, Mz]`` (force first), expressed in the tool0 frame.

With ``subtract_initial`` the first recorded frame is removed as a tare. Because
playback settles into its gravity-loaded equilibrium before recording, the
first frame holds the static gravity wrench and is a valid bias.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(
    npz: str = "results/playback_mc20.npz",
    output: str = "results/flange_wrench_mc20.png",
    subtract_initial: bool = True,
) -> None:
    data = np.load(npz)
    t = data["timestamp"]
    wrench = data["wrench"]  # (N, 6): [Fx, Fy, Fz, Mx, My, Mz] in tool0 frame

    if subtract_initial:
        bias = wrench[0].copy()
        wrench = wrench - bias
        print(f"  initial-frame bias [Fx,Fy,Fz,Mx,My,Mz]: {np.round(bias, 4)}")

    force = wrench[:, :3]
    torque = wrench[:, 3:]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    force_labels = ["Fx", "Fy", "Fz"]
    torque_labels = ["Mx", "My", "Mz"]

    # Shared y-axis span across the three force axes (and likewise for torque),
    # so the relative magnitudes are visually comparable across x/y/z.
    def _padded_span(arr: np.ndarray) -> tuple[float, float]:
        lo, hi = float(arr.min()), float(arr.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        return lo - pad, hi + pad

    force_ylim = _padded_span(force)
    torque_ylim = _padded_span(torque)

    for j in range(3):
        axes[0, j].plot(t, force[:, j], color="tab:blue")
        axes[0, j].set_title(f"{force_labels[j]} [N]")
        axes[0, j].set_ylim(force_ylim)
        axes[0, j].grid(True, alpha=0.3)
        axes[0, j].axhline(0.0, color="k", lw=0.5)

        axes[1, j].plot(t, torque[:, j], color="tab:red")
        axes[1, j].set_title(f"{torque_labels[j]} [Nm]")
        axes[1, j].set_ylim(torque_ylim)
        axes[1, j].set_xlabel("time [s]")
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].axhline(0.0, color="k", lw=0.5)

    fig.suptitle(f"Flange wrench at tool0 frame ({Path(npz).stem})")
    fig.tight_layout()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")
    print(f"  |F| max: {np.abs(force).max(axis=0)} N")
    print(f"  |M| max: {np.abs(torque).max(axis=0)} Nm")


if __name__ == "__main__":
    tyro.cli(main)
