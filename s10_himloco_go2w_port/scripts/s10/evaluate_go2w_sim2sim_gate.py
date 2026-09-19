#!/usr/bin/env python3
"""Run deterministic MuJoCo deployment gates for an S10 HIM checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAYER = ROOT / "scripts/s10/play_go2w_him_mujoco.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


CASES = (
    {
        "name": "flat_vx04",
        "args": ("--terrain", "flat", "--vx", "0.4", "--duration", "5"),
        "limits": {
            "mean_vx_body": (0.2, 0.6),
            "max_abs_lateral_displacement": (None, 0.35),
            "mean_abs_roll_deg": (None, 10.0),
            "mean_abs_pitch_deg": (None, 10.0),
        },
    },
    {
        "name": "flat_vx09",
        "args": ("--terrain", "flat", "--vx", "0.9", "--duration", "5"),
        "limits": {
            "mean_vx_body": (0.55, 1.2),
            "max_abs_lateral_displacement": (None, 0.50),
            "mean_abs_roll_deg": (None, 12.0),
            "mean_abs_pitch_deg": (None, 12.0),
        },
    },
    {
        "name": "flat_vx04_yaw05",
        "args": (
            "--terrain", "flat", "--vx", "0.4", "--omega", "0.5", "--duration", "5"
        ),
        "limits": {
            "mean_vx_body": (0.15, 0.65),
            "mean_omega_body": (0.25, 0.75),
            "mean_abs_roll_deg": (None, 12.0),
            "mean_abs_pitch_deg": (None, 12.0),
        },
    },
    {
        "name": "stairs_l5_vx09",
        "args": (
            "--terrain", "stairs", "--level", "5", "--vx", "0.9", "--duration", "7"
        ),
        "limits": {
            "dx": (2.0, None),
            "max_abs_lateral_displacement": (None, 0.75),
            "min_base_height": (0.25, None),
            "mean_abs_roll_deg": (None, 20.0),
            "mean_abs_pitch_deg": (None, 20.0),
        },
    },
)


def _run_case(checkpoint: Path, device: str, case: dict) -> dict:
    command = [
        sys.executable,
        "-u",
        str(PLAYER),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--headless",
        "--no-real-time",
        "--print-every",
        "0",
        *case["args"],
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case['name']} player failed with code {completed.returncode}:\n{completed.stderr}"
        )
    summary_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("SIM2SIM_SUMMARY ")),
        None,
    )
    if summary_line is None:
        raise RuntimeError(f"{case['name']} produced no SIM2SIM_SUMMARY")
    summary = json.loads(summary_line.removeprefix("SIM2SIM_SUMMARY "))
    failures: list[str] = []
    if summary["fell"]:
        failures.append("fell")
    for metric, (minimum, maximum) in case["limits"].items():
        value = float(summary[metric])
        if minimum is not None and value < minimum:
            failures.append(f"{metric}={value:.4g} < {minimum:.4g}")
        if maximum is not None and value > maximum:
            failures.append(f"{metric}={value:.4g} > {maximum:.4g}")
    return {"name": case["name"], "passed": not failures, "failures": failures, "summary": summary}


def main() -> None:
    args = _parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    results = [_run_case(checkpoint, args.device, case) for case in CASES]
    report = {
        "checkpoint": str(checkpoint),
        "passed": all(result["passed"] for result in results),
        "cases": results,
    }
    print("SIM2SIM_GATE " + json.dumps(report, sort_keys=True), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
