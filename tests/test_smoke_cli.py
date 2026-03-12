"""Smoke tests for CLI entry points (--help should exit cleanly)."""

from __future__ import annotations

import subprocess
import sys


def test_optimize_excitation_help() -> None:
    """CLI --help should exit cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjwarp_ur5e.demos.optimize_excitation_trajectory",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "num-harmonics" in result.stdout


def test_validate_excitation_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjwarp_ur5e.demos.validate_excitation_trajectory",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0


def test_run_playback_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjwarp_ur5e.demos.run_identification_playback",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0


def test_run_identification_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjwarp_ur5e.demos.run_inertial_identification",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
