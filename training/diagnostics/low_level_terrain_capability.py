"""Measure the official S10 low-level controller's platform and stair limits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from sru_training.s10_mujoco_backend import (
    DONE_NONE,
    STANDING_BASE_CLEARANCE,
    S10NativeMujocoBackend,
)
from training.terrains.capability_course import build_capability_course
from training.terrains.constants import HEIGHTS, STAIRS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOW_LEVEL = (
    REPO_ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "training/evidence/low_level_terrain_capability_official_20260828.json"
)


@dataclass(frozen=True)
class Trial:
    speed: float
    yaw_offset_deg: float


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def _yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _run_height(
    kind: str,
    height: float,
    trials: tuple[Trial, ...],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    model, course = build_capability_course(kind, height)
    backend = S10NativeMujocoBackend(
        num_envs=len(trials),
        model_override=model,
        device="cpu",
        low_level="official_onnx",
        low_level_checkpoint=args.low_level_checkpoint,
        low_level_profile=args.low_level_profile,
        max_episode_length=args.max_steps + 1,
        single_episode_length=args.max_steps + 1,
        reset_mode="fixed",
        use_lidar=False,
        use_height=False,
        physics_workers=min(args.physics_workers, len(trials)),
        contact_threshold=args.contact_threshold,
        contact_persistence_steps=args.contact_persistence_steps,
        record_contact_diagnostics=True,
        seed=args.seed,
    )
    yaw_offsets = np.deg2rad([trial.yaw_offset_deg for trial in trials])
    backend.reset_route_segments(
        np.zeros(len(trials), dtype=np.int64),
        np.zeros(len(trials), dtype=np.float64),
        goal_indices=np.ones(len(trials), dtype=np.int64),
        yaws=yaw_offsets,
    )

    active = np.ones(len(trials), dtype=bool)
    passed = np.zeros(len(trials), dtype=bool)
    stable_steps = np.zeros(len(trials), dtype=np.int64)
    done_reason = np.full(len(trials), "probe_timeout", dtype=object)
    done_step = np.full(len(trials), args.max_steps, dtype=np.int64)
    max_x = np.full(len(trials), course.start[0], dtype=np.float64)
    max_z = np.zeros(len(trials), dtype=np.float64)
    max_tilt = np.zeros(len(trials), dtype=np.float64)
    max_contact = np.zeros(len(trials), dtype=np.float64)
    body_force_peaks: list[dict[str, float]] = [dict() for _ in trials]

    try:
        for step in range(1, args.max_steps + 1):
            commands = np.zeros((len(trials), 3), dtype=np.float32)
            for index in np.flatnonzero(active):
                heading_error = _wrap(-_yaw(backend.data[index].qpos[3:7]))
                commands[index, 2] = np.clip(
                    args.heading_gain * heading_error,
                    -args.max_yaw_rate,
                    args.max_yaw_rate,
                )
                commands[index, 0] = trials[index].speed * max(
                    0.0, math.cos(heading_error)
                )

            _, _, dones, info = backend.step(torch.from_numpy(commands))
            positions = np.asarray(info["terminal_base_position"], dtype=np.float64)
            tilts = np.asarray(info["terminal_tilt_deg"], dtype=np.float64)
            contacts = np.asarray(info["max_illegal_contact_force"], dtype=np.float64)
            max_x[active] = np.maximum(max_x[active], positions[active, 0])
            max_z[active] = np.maximum(max_z[active], positions[active, 2])
            max_tilt[active] = np.maximum(max_tilt[active], tilts[active])
            max_contact[active] = np.maximum(max_contact[active], contacts[active])
            for index in np.flatnonzero(active):
                for body, force in info["contact_force_peaks_by_body"][index].items():
                    body_force_peaks[index][body] = max(
                        body_force_peaks[index].get(body, 0.0), float(force)
                    )

            upright_on_top = (
                active
                & (positions[:, 0] >= course.stable_check_x)
                & (
                    positions[:, 2]
                    >= course.total_height + STANDING_BASE_CLEARANCE - 0.10
                )
                & (tilts < args.stable_tilt_deg)
            )
            stable_steps[upright_on_top] += 1
            stable_steps[active & ~upright_on_top] = 0
            newly_passed = active & (stable_steps >= args.stable_steps)
            passed[newly_passed] = True
            done_reason[newly_passed] = "passed"
            done_step[newly_passed] = step
            active[newly_passed] = False

            done_array = dones.detach().cpu().numpy().astype(bool)
            failed = active & done_array
            for index in np.flatnonzero(failed):
                reason = str(info["done_reason"][index])
                done_reason[index] = reason if reason != DONE_NONE else "terminated"
                done_step[index] = step
            active[failed] = False
            if not active.any():
                break
    finally:
        backend.close()

    return [
        {
            "kind": kind,
            "height_m": height,
            "total_height_m": course.total_height,
            **asdict(trial),
            "success": bool(passed[index]),
            "reason": str(done_reason[index]),
            "steps": int(done_step[index]),
            "max_x": float(max_x[index]),
            "progress_x": float(max_x[index] - course.start[0]),
            "max_base_z": float(max_z[index]),
            "max_tilt_deg": float(max_tilt[index]),
            "max_illegal_contact_force": float(max_contact[index]),
            "body_force_peaks": dict(sorted(body_force_peaks[index].items())),
        }
        for index, trial in enumerate(trials)
    ]


def _summarize(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[
            (str(result["kind"]), float(result["height_m"]), float(result["speed"]))
        ].append(result)
    summaries = []
    for (kind, height, speed), items in sorted(grouped.items()):
        successes = sum(bool(item["success"]) for item in items)
        summary = {
            "kind": kind,
            "height_m": height,
            "speed": speed,
            "trials": len(items),
            "successes": successes,
            "success_rate": successes / len(items),
            "reasons": dict(Counter(str(item["reason"]) for item in items)),
            "mean_progress_x": float(np.mean([item["progress_x"] for item in items])),
            "max_tilt_deg": float(max(item["max_tilt_deg"] for item in items)),
            "max_illegal_contact_force": float(
                max(item["max_illegal_contact_force"] for item in items)
            ),
        }
        summaries.append(summary)
        print(
            f"{kind:8s} height={height:.2f}m vx={speed:.2f} "
            f"success={successes}/{len(items)} reasons={summary['reasons']} "
            f"progress={summary['mean_progress_x']:.2f}m "
            f"tilt={summary['max_tilt_deg']:.1f}deg",
            flush=True,
        )
    return summaries


def _reliable_limits(summaries: list[dict[str, object]]) -> dict[str, object]:
    limits: dict[str, object] = {}
    requirements = {
        "platform": HEIGHTS.ramp_platform_meters,
        "stairs": STAIRS.STEP_HEIGHT_METERS,
    }
    for kind, required in requirements.items():
        candidates = [
            row
            for row in summaries
            if row["kind"] == kind and float(row["success_rate"]) >= 1.0
        ]
        max_height = max((float(row["height_m"]) for row in candidates), default=None)
        required_rows = [
            row
            for row in summaries
            if row["kind"] == kind
            and abs(float(row["height_m"]) - required) < 1.0e-9
        ]
        limits[kind] = {
            "terrain_required_height_m": required,
            "maximum_fully_reliable_tested_height_m": max_height,
            "terrain_required_passed_at_any_speed": any(
                float(row["success_rate"]) >= 1.0 for row in required_rows
            ),
            "reliable_speeds_at_terrain_height": [
                float(row["speed"])
                for row in required_rows
                if float(row["success_rate"]) >= 1.0
            ],
        }
    return limits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform-heights",
        type=_csv_floats,
        default=(0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
    )
    parser.add_argument(
        "--stair-heights",
        type=_csv_floats,
        default=(0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    )
    parser.add_argument("--speeds", type=_csv_floats, default=(0.25, 0.45, 0.65))
    parser.add_argument(
        "--yaw-offsets-deg", type=_csv_floats, default=(-5.0, 0.0, 5.0)
    )
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--stable-steps", type=int, default=5)
    parser.add_argument("--stable-tilt-deg", type=float, default=30.0)
    parser.add_argument("--heading-gain", type=float, default=1.4)
    parser.add_argument("--max-yaw-rate", type=float, default=0.6)
    parser.add_argument("--physics-workers", type=int, default=8)
    parser.add_argument("--contact-threshold", type=float, default=500.0)
    parser.add_argument("--contact-persistence-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_LOW_LEVEL)
    parser.add_argument(
        "--low-level-profile",
        choices=("legacy", "official_20260828"),
        default="official_20260828",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(height <= 0.0 for height in args.platform_heights + args.stair_heights):
        raise ValueError("all tested heights must be positive")
    if any(speed <= 0.0 or speed > 1.0 for speed in args.speeds):
        raise ValueError("speeds must be in (0, 1]")
    if args.max_steps < 1 or args.stable_steps < 1 or args.physics_workers < 1:
        raise ValueError("step and worker counts must be positive")

    trials = tuple(
        Trial(speed=speed, yaw_offset_deg=yaw)
        for speed in args.speeds
        for yaw in args.yaw_offsets_deg
    )
    results: list[dict[str, object]] = []
    for kind, heights in (
        ("platform", args.platform_heights),
        ("stairs", args.stair_heights),
    ):
        for height in heights:
            print(f"[capability] running {kind} height={height:.2f}m", flush=True)
            results.extend(_run_height(kind, height, trials, args))

    summaries = _summarize(results)
    limits = _reliable_limits(summaries)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controller": {
            "checkpoint": str(args.low_level_checkpoint.expanduser().resolve()),
            "profile": args.low_level_profile,
        },
        "terrain_contract": {
            "single_platform_height_m": HEIGHTS.ramp_platform_meters,
            "stair_step_height_m": STAIRS.STEP_HEIGHT_METERS,
            "stair_steps": STAIRS.NUM_STEPS,
            "stair_tread_depth_m": 0.4,
            "stair_total_height_m": STAIRS.total_height_meters,
            "hfield_horizontal_scale_m": 0.1,
        },
        "test_config": {
            "platform_heights": list(args.platform_heights),
            "stair_heights": list(args.stair_heights),
            "speeds": list(args.speeds),
            "yaw_offsets_deg": list(args.yaw_offsets_deg),
            "max_steps": args.max_steps,
            "stable_steps": args.stable_steps,
            "stable_tilt_deg": args.stable_tilt_deg,
            "contact_threshold": args.contact_threshold,
            "contact_persistence_steps": args.contact_persistence_steps,
            "seed": args.seed,
        },
        "reliable_limits": limits,
        "summaries": summaries,
        "trials": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(limits, indent=2), flush=True)
    print(f"LOW_LEVEL_TERRAIN_CAPABILITY_OK {args.json_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
