"""Native multi-environment MuJoCo backend for S10 high-level training.

This backend intentionally keeps the training boundary free of ROS/DDS. One
``MjModel`` is shared by several independent ``MjData`` objects. The default
low-level controller reproduces the official 57-dim ONNX deployment policy;
standing PD remains available only as a diagnostic fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from training.terrains import (
    SruAtlasConfig,
    SruPositionSampler,
    assign_terrain_tiles,
    build_sru_mujoco_model,
    generate_sru_atlas,
)

from .s10_height_scan import HeightFeatureEncoder, MuJoCoHeightScan
from .s10_height_scan import HEIGHT_GRID_SIZE, HEIGHT_VALUE_OFFSET
from .s10_entry_state_bank import EntryStateBank, EntryStateBankContract
from .s10_lidar_encoder import (
    INVALID_RANGE_THRESHOLD_M,
    MIN_RANGE_M,
    S10_FRONT_POS,
    S10_FRONT_ROT_WXYZ,
    S10LegacyLidarEncoder,
    S10_REAR_POS,
    S10_REAR_ROT_WXYZ,
    build_sensor_frame_directions,
    gather_aux_at_min_distance,
    native_to_90,
    world_z_native,
)
from .s10_mujoco_env import S10RawState
from .s10_policy_config import S10ActionSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = REPO_ROOT / "src/S10_sdk_deploy/interface/robot/simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))
from s10_lidar import S10LidarSampler  # noqa: E402


TRACK_XML = REPO_ROOT / "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml"
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
BASE_BODY_NAME = "base_link"
DOF = 16
ROOT_QPOS = 7
ROOT_QVEL = 6
HEIGHT_CENTER = HEIGHT_GRID_SIZE // 2
ROBOT_ORDER = ("fl_hipx", "fl_hipy", "fl_knee", "fl_wheel", "fr_hipx", "fr_hipy", "fr_knee", "fr_wheel", "hl_hipx", "hl_hipy", "hl_knee", "hl_wheel", "hr_hipx", "hr_hipy", "hr_knee", "hr_wheel")
POLICY_ORDER = ("fl_hipx", "fl_hipy", "fl_knee", "fr_hipx", "fr_hipy", "fr_knee", "hl_hipx", "hl_hipy", "hl_knee", "hr_hipx", "hr_hipy", "hr_knee", "fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
# Correct policy order from the official C++ runner. Keep this explicit to
# make a future ONNX export mismatch fail in tests instead of silently moving
# a leg.
# Indices used by the C++ ``robot2policy_idx`` and ``policy2robot_idx``.
POLICY_ORDER_ROBOT_INDICES = np.asarray(
    [ROBOT_ORDER.index(name) for name in POLICY_ORDER], dtype=np.int64
)
ROBOT_ORDER_POLICY_INDICES = np.argsort(POLICY_ORDER_ROBOT_INDICES)
ACTION_SCALE_ROBOT = np.asarray((0.125, 0.25, 0.25, 5.0) * 4, dtype=np.float32)
DEFAULT_ROBOT = np.asarray((0.0, -0.3, 0.6, 0.0, 0.0, -0.3, 0.6, 0.0, 0.0, 0.3, -0.6, 0.0, 0.0, 0.3, -0.6, 0.0), dtype=np.float32)
# The C++ runner stores a separate default in policy order (all twelve leg
# joints, then four wheels). Derive it from the robot-order constant so the two
# representations cannot silently drift apart.
DEFAULT_POLICY = DEFAULT_ROBOT[POLICY_ORDER_ROBOT_INDICES].copy()


@dataclass(frozen=True)
class S10LowLevelProfile:
    """Parameters that must stay paired with an official low-level ONNX."""

    name: str
    default_robot: tuple[float, ...]
    wheel_kd: float
    command_scale: tuple[float, float, float]
    startup_ramp_steps: int

    @property
    def default_robot_array(self) -> np.ndarray:
        return np.asarray(self.default_robot, dtype=np.float32)


LOW_LEVEL_PROFILES = {
    "legacy": S10LowLevelProfile(
        name="legacy",
        default_robot=tuple(float(value) for value in DEFAULT_ROBOT),
        wheel_kd=0.6,
        command_scale=(1.0, 1.0, 1.0),
        startup_ramp_steps=0,
    ),
    "official_20260828": S10LowLevelProfile(
        name="official_20260828",
        default_robot=(
            0.05, -0.35, 0.65, 0.0,
            -0.05, -0.35, 0.65, 0.0,
            0.05, 0.35, -0.65, 0.0,
            -0.05, 0.35, -0.65, 0.0,
        ),
        wheel_kd=0.8,
        # Keep this paired with the official 2026-08-20 C++ runner. These
        # factors convert bounded UserCommand values into the ONNX command
        # observation; they are separate from SRU's high-level [1.5, 1.0]
        # actor-action scaling.
        command_scale=(1.5, 0.5, 0.6),
        startup_ramp_steps=150,
    ),
}

# The C++ state machine hands control to the ONNX policy only after its
# four-second stand-up sequence.  At that point the leg pose is the low-level
# policy default and the base settles at about 0.422 m on a flat surface.
STANDING_BASE_CLEARANCE = 0.424


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: str | Path) -> str:
    """Hash names and contents so included MJCF/mesh changes invalidate resume."""

    root = Path(path).resolve()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    return np.asarray(
        ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
         (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
         (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        dtype=np.float64,
    )


@dataclass
class BackendMetrics:
    mean_forward_speed: float = 0.0
    mean_yaw_speed: float = 0.0
    mean_reward: float = 0.0
    done_count: int = 0


class AdaptiveSegmentSampler:
    """Sample route tasks according to online mastery.

    ``mastery`` is a continuous value in [0, 1]. For the retained adjacent-goal
    task it is identical to success. The separate binary success EMA keeps the
    curriculum behavior explicit in logs and checkpoints.
    """

    def __init__(
        self,
        segments: np.ndarray,
        *,
        enabled: bool,
        uniform_mix: float,
        ema_alpha: float,
        difficulty_power: float,
        warmup_attempts: int,
        max_probability: float = 1.0,
        priority_segment: int | None = None,
        priority_min_probability: float = 0.0,
        strategy: str = "difficulty",
        progress_fast_alpha: float = 0.10,
        progress_slow_alpha: float = 0.01,
        progress_min_mastery: float = 0.05,
        progress_max_mastery: float = 0.95,
        progress_epsilon: float = 1.0e-3,
    ) -> None:
        self.segments = np.asarray(segments, dtype=np.int64)
        if self.segments.ndim != 1 or not len(self.segments):
            raise ValueError("adaptive sampler requires at least one segment")
        if len(np.unique(self.segments)) != len(self.segments):
            raise ValueError("adaptive sampler segments must be unique")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError("adaptive uniform mix must be in [0, 1]")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("adaptive EMA alpha must be in (0, 1]")
        if difficulty_power <= 0.0:
            raise ValueError("adaptive difficulty power must be positive")
        if warmup_attempts < 0:
            raise ValueError("adaptive warmup attempts must be non-negative")
        if not 1.0 / len(self.segments) <= max_probability <= 1.0:
            raise ValueError(
                "adaptive max probability must be in [1 / segment_count, 1]"
            )
        if priority_segment is not None and int(priority_segment) not in set(self.segments.tolist()):
            raise ValueError("adaptive priority segment must be part of the sampler")
        if not 0.0 <= priority_min_probability < 1.0:
            raise ValueError("adaptive priority minimum probability must be in [0, 1)")
        if priority_segment is not None and priority_min_probability > 1.0 - 1.0 / len(self.segments):
            raise ValueError(
                "adaptive priority minimum probability leaves no mass for other segments"
            )
        if strategy not in {"difficulty", "learning_progress"}:
            raise ValueError("adaptive strategy must be 'difficulty' or 'learning_progress'")
        if not 0.0 < progress_fast_alpha <= 1.0:
            raise ValueError("progress fast EMA alpha must be in (0, 1]")
        if not 0.0 < progress_slow_alpha < progress_fast_alpha:
            raise ValueError("progress slow EMA alpha must be in (0, fast_alpha)")
        if not 0.0 <= progress_min_mastery < progress_max_mastery <= 1.0:
            raise ValueError("progress mastery band must lie inside [0, 1]")
        if progress_epsilon < 0.0:
            raise ValueError("progress epsilon must be non-negative")
        self.enabled = bool(enabled)
        self.uniform_mix = float(uniform_mix)
        self.ema_alpha = float(ema_alpha)
        self.difficulty_power = float(difficulty_power)
        self.warmup_attempts = int(warmup_attempts)
        self.max_probability = float(max_probability)
        self.priority_segment = None if priority_segment is None else int(priority_segment)
        self.priority_min_probability = float(priority_min_probability)
        self.strategy = strategy
        self.progress_fast_alpha = float(progress_fast_alpha)
        self.progress_slow_alpha = float(progress_slow_alpha)
        self.progress_min_mastery = float(progress_min_mastery)
        self.progress_max_mastery = float(progress_max_mastery)
        self.progress_epsilon = float(progress_epsilon)
        self.attempts = np.zeros(len(self.segments), dtype=np.int64)
        self.successes = np.zeros(len(self.segments), dtype=np.int64)
        self.success_ema = np.full(len(self.segments), 0.5, dtype=np.float64)
        self.mastery_ema = np.full(len(self.segments), 0.5, dtype=np.float64)
        self.progress_fast_ema = np.full(len(self.segments), 0.5, dtype=np.float64)
        self.progress_slow_ema = np.full(len(self.segments), 0.5, dtype=np.float64)
        self._segment_to_slot = {
            int(segment): slot for slot, segment in enumerate(self.segments)
        }

    @property
    def warmup_complete(self) -> bool:
        return bool((self.attempts >= self.warmup_attempts).all())

    def probabilities(self) -> np.ndarray:
        uniform = np.full(len(self.segments), 1.0 / len(self.segments))
        if not self.enabled or not self.warmup_complete:
            probabilities = uniform
        else:
            if self.strategy == "difficulty":
                score = (
                    np.maximum(1.0 - self.mastery_ema, 1.0e-3)
                    ** self.difficulty_power
                )
            else:
                progress = np.abs(
                    self.progress_fast_ema - self.progress_slow_ema
                )
                learnable = (
                    (self.progress_fast_ema >= self.progress_min_mastery)
                    & (self.progress_fast_ema <= self.progress_max_mastery)
                )
                score = np.where(
                    learnable,
                    progress ** self.difficulty_power + self.progress_epsilon,
                    0.0,
                )
                if float(score.sum()) <= 1.0e-12:
                    score = np.ones_like(score)
            adaptive = score / score.sum()
            probabilities = self.uniform_mix * uniform + (1.0 - self.uniform_mix) * adaptive
        probabilities = self._cap_probabilities(probabilities)
        if self.priority_segment is not None and self.priority_min_probability > 0.0:
            slot = self._segment_to_slot[self.priority_segment]
            if probabilities[slot] < self.priority_min_probability:
                deficit = self.priority_min_probability - probabilities[slot]
                donors = np.ones(len(probabilities), dtype=bool)
                donors[slot] = False
                donor_mass = float(probabilities[donors].sum())
                if donor_mass <= deficit:
                    raise RuntimeError("adaptive priority minimum cannot be satisfied")
                probabilities[donors] -= deficit * probabilities[donors] / donor_mass
                probabilities[slot] = self.priority_min_probability
        return probabilities / probabilities.sum()

    def _cap_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        if self.max_probability >= 1.0:
            return probabilities
        capped = np.zeros_like(probabilities)
        free = np.ones(len(probabilities), dtype=bool)
        remaining_mass = 1.0
        while free.any():
            weights = probabilities[free]
            proposed = remaining_mass * weights / weights.sum()
            over = proposed > self.max_probability + 1.0e-12
            if not over.any():
                capped[free] = proposed
                break
            free_indices = np.flatnonzero(free)
            capped[free_indices[over]] = self.max_probability
            free[free_indices[over]] = False
            remaining_mass = 1.0 - float(capped.sum())
        return capped / capped.sum()

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.segments, p=self.probabilities()))

    def update(
        self,
        segments: np.ndarray,
        mastery: np.ndarray,
        successes: np.ndarray | None = None,
    ) -> None:
        segments = np.asarray(segments, dtype=np.int64)
        mastery = np.asarray(mastery, dtype=np.float64)
        if segments.shape != mastery.shape:
            raise ValueError("adaptive sampler segments and mastery must have matching shapes")
        if not np.isfinite(mastery).all() or (mastery < 0.0).any() or (mastery > 1.0).any():
            raise ValueError("adaptive sampler mastery must be finite and in [0, 1]")
        if successes is None:
            successes = mastery >= 1.0 - 1.0e-12
        successes = np.asarray(successes, dtype=bool)
        if successes.shape != segments.shape:
            raise ValueError("adaptive sampler successes must match segment shape")
        for segment, outcome, succeeded in zip(segments, mastery, successes):
            slot = self._segment_to_slot.get(int(segment))
            if slot is None:
                continue
            success = float(succeeded)
            self.attempts[slot] += 1
            self.successes[slot] += int(success)
            self.success_ema[slot] += self.ema_alpha * (
                success - self.success_ema[slot]
            )
            self.mastery_ema[slot] += self.ema_alpha * (
                float(outcome) - self.mastery_ema[slot]
            )
            self.progress_fast_ema[slot] += self.progress_fast_alpha * (
                float(outcome) - self.progress_fast_ema[slot]
            )
            self.progress_slow_ema[slot] += self.progress_slow_alpha * (
                float(outcome) - self.progress_slow_ema[slot]
            )

    def summary(self) -> dict[str, Any]:
        probabilities = self.probabilities()
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12))))
        effective_segments = float(np.exp(entropy))
        return {
            "enabled": self.enabled,
            "warmup_complete": self.warmup_complete,
            "total_episodes": int(self.attempts.sum()),
            "segments": self.segments.tolist(),
            "attempts": self.attempts.tolist(),
            "successes": self.successes.tolist(),
            "success_ema": self.success_ema.tolist(),
            "mastery_ema": self.mastery_ema.tolist(),
            "progress_fast_ema": self.progress_fast_ema.tolist(),
            "progress_slow_ema": self.progress_slow_ema.tolist(),
            "learning_progress": np.abs(
                self.progress_fast_ema - self.progress_slow_ema
            ).tolist(),
            "probabilities": probabilities.tolist(),
            "effective_segments": effective_segments,
            "max_probability": self.max_probability,
            "priority_segment": self.priority_segment,
            "priority_min_probability": self.priority_min_probability,
            "strategy": self.strategy,
            "progress_mastery_band": [
                self.progress_min_mastery,
                self.progress_max_mastery,
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "segments": self.segments.tolist(),
            "attempts": self.attempts.tolist(),
            "successes": self.successes.tolist(),
            "success_ema": self.success_ema.tolist(),
            "mastery_ema": self.mastery_ema.tolist(),
            "progress_fast_ema": self.progress_fast_ema.tolist(),
            "progress_slow_ema": self.progress_slow_ema.tolist(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        segments = np.asarray(state.get("segments"), dtype=np.int64)
        if not np.array_equal(segments, self.segments):
            raise ValueError("adaptive sampler state does not match eligible route segments")
        attempts = np.asarray(state.get("attempts"), dtype=np.int64)
        successes = np.asarray(state.get("successes"), dtype=np.int64)
        success_ema = np.asarray(state.get("success_ema"), dtype=np.float64)
        # Version-1 checkpoints only recorded binary success. They remain
        # loadable for exact resume, while task-transfer runs start a fresh
        # curriculum by default.
        mastery_ema = np.asarray(state.get("mastery_ema", success_ema), dtype=np.float64)
        progress_fast_ema = np.asarray(
            state.get("progress_fast_ema", mastery_ema), dtype=np.float64
        )
        progress_slow_ema = np.asarray(
            state.get("progress_slow_ema", mastery_ema), dtype=np.float64
        )
        expected = (len(self.segments),)
        if (
            attempts.shape != expected
            or successes.shape != expected
            or success_ema.shape != expected
            or mastery_ema.shape != expected
            or progress_fast_ema.shape != expected
            or progress_slow_ema.shape != expected
        ):
            raise ValueError("adaptive sampler state has invalid array dimensions")
        if (
            (attempts < 0).any()
            or (successes < 0).any()
            or (successes > attempts).any()
            or not np.isfinite(success_ema).all()
            or not np.isfinite(mastery_ema).all()
            or not np.isfinite(progress_fast_ema).all()
            or not np.isfinite(progress_slow_ema).all()
            or (success_ema < 0.0).any()
            or (success_ema > 1.0).any()
            or (mastery_ema < 0.0).any()
            or (mastery_ema > 1.0).any()
            or (progress_fast_ema < 0.0).any()
            or (progress_fast_ema > 1.0).any()
            or (progress_slow_ema < 0.0).any()
            or (progress_slow_ema > 1.0).any()
        ):
            raise ValueError("adaptive sampler state contains invalid values")
        self.attempts[:] = attempts
        self.successes[:] = successes
        self.success_ema[:] = success_ema
        self.mastery_ema[:] = mastery_ema
        self.progress_fast_ema[:] = progress_fast_ema
        self.progress_slow_ema[:] = progress_slow_ema


@dataclass(frozen=True)
class S10RewardConfig:
    """SRU navigation reward terms at the high-level step rate."""

    joint_acc_l2: float = -1.0e-7
    lateral_movement: float = -0.1
    rot_movement: float = -1.0e-5
    action_rate_l1: float = -0.1
    episode_termination: float = -50.0
    reach_goal_xy_soft: float = 0.25
    reach_goal_xy_tight: float = 1.5
    soft_sigmoid: float = 2.5
    soft_time_scale: float = 1.0
    tight_sigmoid: float = 0.25
    tight_time_scale: float = 0.1
    random_goal_reward_probability: float = 0.01
    # Optional dense shaping for the flat-navigation warmup. Zero preserves
    # the original sparse SRU reward; maintained Stage 1 opts in.
    goal_progress: float = 0.0
    goal_distance_threshold: float = 0.5
    required_time_at_goal: float = 4.0
    # Waypoint z is terrain height, while qpos z is the robot base height.
    # This threshold separates stacked floors without requiring exact base
    # clearance on stairs and slopes.
    goal_height_threshold: float = 0.55


def compute_goal_progress_reward(
    previous_distance_xy: float,
    current_distance_xy: float,
    coefficient: float,
) -> float:
    """Return signed reward for reducing XY distance during one policy step."""

    if coefficient < 0.0:
        raise ValueError("goal progress coefficient must be non-negative")
    if not np.isfinite(previous_distance_xy) or not np.isfinite(current_distance_xy):
        raise ValueError("goal distances must be finite")
    return float(coefficient * (previous_distance_xy - current_distance_xy))
TASK_SINGLE = 0
TASK_SEQUENTIAL_ROUTE = 1
TASK_NAMES = {
    TASK_SINGLE: "single",
    TASK_SEQUENTIAL_ROUTE: "sequential_route",
}

# Waypoint 15 was moved from the trench onto the validated bridge centerline.
# Every route node is now an enabled adjacent target, including 14->15->16.
SKIPPED_ROUTE_WAYPOINTS = frozenset()
DISABLED_ROUTE_START_SEGMENTS = SKIPPED_ROUTE_WAYPOINTS


def route_waypoint_sequence(start: int, final: int) -> tuple[int, ...]:
    """Return ordered navigation targets after start, excluding disabled points."""

    if start >= final:
        raise ValueError("route start must precede the final waypoint")
    return tuple(
        index
        for index in range(start + 1, final + 1)
        if index not in SKIPPED_ROUTE_WAYPOINTS
    )


def next_route_waypoint(start: int, final: int) -> int:
    targets = route_waypoint_sequence(start, final)
    if not targets:
        raise ValueError(f"route {start}->{final} has no enabled target")
    return targets[0]


def route_target_count(start: int, final: int) -> int:
    return len(route_waypoint_sequence(start, final))


DONE_NONE = "none"
DONE_TIMEOUT = "timeout"
DONE_FALLEN = "fallen"
DONE_NONFINITE = "nonfinite"
DONE_COMPLETE = "goal_complete"
DONE_BASE_CONTACT = "base_contact"
DONE_LARGE_ANGLE = "large_angle"
DONE_TERRAIN_FALL = "terrain_fall"


class OfficialS10OnnxController:
    """Python reproduction of ``S10PolicyRunner::getRobotAction``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        profile: S10LowLevelProfile = LOW_LEVEL_PROFILES["legacy"],
        intra_op_threads: int = 1,
    ):
        import onnx
        import onnxruntime as ort

        model = onnx.load(str(Path(checkpoint).expanduser().resolve()))
        # The shipped graph contains only Gemm/Elu operations, but its export
        # metadata fixes the leading dimension to one. Those operations are
        # naturally batch-safe, so make only the input/output metadata dynamic
        # and retain every learned parameter unchanged.
        for value_info in (model.graph.input[0], model.graph.output[0]):
            batch_dim = value_info.type.tensor_type.shape.dim[0]
            batch_dim.ClearField("dim_value")
            batch_dim.dim_param = "batch"
        self.session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
            sess_options=self._session_options(ort, intra_op_threads),
        )
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        if input_meta.name != "obs" or output_meta.name != "actions":
            raise RuntimeError(f"Unexpected S10 ONNX names: {input_meta.name}, {output_meta.name}")
        self.profile = profile
        self.default_robot = profile.default_robot_array
        self.default_policy = self.default_robot[POLICY_ORDER_ROBOT_INDICES].copy()
        self.command_scale = np.asarray(profile.command_scale, dtype=np.float32)
        self.last_action = np.zeros((1, 16), dtype=np.float32)
        self.run_count = np.zeros(1, dtype=np.int64)
        self.kp_scale = np.ones(1, dtype=np.float64)
        self.kd_scale = np.ones(1, dtype=np.float64)

    @staticmethod
    def _session_options(ort, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Some shipped graphs retain an internal fixed-batch value-info entry
        # after the public input/output metadata is made dynamic. ORT executes
        # the batch correctly but otherwise emits one VerifyOutputSizes warning
        # per inference, which makes multi-environment diagnostics unusable.
        options.log_severity_level = 3
        return options

    def reset(self, num_envs: int, indices: np.ndarray | None = None) -> None:
        if indices is None:
            self.last_action = np.zeros((num_envs, 16), dtype=np.float32)
            self.run_count = np.zeros(num_envs, dtype=np.int64)
            self.kp_scale = np.ones(num_envs, dtype=np.float64)
            self.kd_scale = np.ones(num_envs, dtype=np.float64)
        else:
            selected = np.asarray(indices, dtype=np.int64)
            self.last_action[selected] = 0.0
            self.run_count[selected] = 0
            self.kp_scale[selected] = 1.0
            self.kd_scale[selected] = 1.0

    def infer(self, data: list[mujoco.MjData], commands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_envs = len(data)
        if self.last_action.shape[0] != num_envs:
            self.reset(num_envs)
        observations = np.zeros((num_envs, 57), dtype=np.float32)
        for i, item in enumerate(data):
            if item.sensordata.shape[0] < 10:
                raise RuntimeError("S10 MuJoCo model is missing the quaternion/accelerometer/gyro sensor contract")
            # Match ``mujoco_simulation_ros2._publish_robot_state`` exactly.
            # Its gyro reading is already expressed in the IMU/body frame;
            # rotating free-joint qvel again mixes roll/pitch whenever the
            # robot has non-zero yaw and rapidly destabilizes the policy.
            rot = quat_wxyz_to_rotmat(item.sensordata[:4])
            omega_body = item.sensordata[7:10]
            gravity_body = rot.T @ np.asarray((0.0, 0.0, -1.0))
            # The ROS simulator converts raw MuJoCo joints to wire format and
            # S10Interface converts them back before invoking this policy.
            # Native execution therefore feeds the raw MuJoCo coordinates.
            joint_pos_policy = (
                item.qpos[7:7 + DOF][POLICY_ORDER_ROBOT_INDICES] - self.default_policy
            )
            joint_vel_policy = (
                item.qvel[6:6 + DOF][POLICY_ORDER_ROBOT_INDICES] * 0.05
            )
            joint_pos_policy[12:16] = 0.0
            observations[i] = np.concatenate((
                omega_body.astype(np.float32) * 0.25,
                gravity_body.astype(np.float32),
                np.asarray(commands[i], dtype=np.float32) * self.command_scale,
                joint_pos_policy.astype(np.float32),
                joint_vel_policy.astype(np.float32),
                self.last_action[i],
            ))
        actions = np.asarray(
            self.session.run(["actions"], {"obs": observations})[0], dtype=np.float32
        )
        if actions.shape != (num_envs, 16):
            raise RuntimeError(f"S10 ONNX returned {actions.shape}, expected {(num_envs, 16)}")
        self.last_action = actions.copy()
        emitted_actions = actions.copy()
        ramp_steps = self.profile.startup_ramp_steps
        if ramp_steps > 0:
            ramping = self.run_count < ramp_steps
            ratios = self.run_count.astype(np.float64) / float(ramp_steps)
            emitted_actions[ramping] *= ratios[ramping, None].astype(np.float32)
            self.kp_scale[ramping] = 0.2 + 0.8 * ratios[ramping]
            self.kd_scale[ramping] = 1.5
            self.kp_scale[~ramping] = 1.0
            self.kd_scale[~ramping] = 1.0
        else:
            self.kp_scale.fill(1.0)
            self.kd_scale.fill(1.0)
        self.run_count += 1
        action_robot = (
            emitted_actions[:, ROBOT_ORDER_POLICY_INDICES]
            * ACTION_SCALE_ROBOT[None, :]
            + self.default_robot[None, :]
        )
        return action_robot[:, :], emitted_actions


class S10NativeMujocoBackend:
    """A batch-first, native MuJoCo backend implementing S10RawState."""

    def __init__(
        self,
        num_envs: int = 1,
        *,
        xml_path: str | Path = TRACK_XML,
        model_override: mujoco.MjModel | None = None,
        waypoint_route: str | Path | None = None,
        task_mode: str = "waypoint",
        terrain_seed: int | None = None,
        surface_seed: int | None = None,
        terrain_profile: str = "legacy_full",
        grass_fraction: float = 0.25,
        gravel_fraction: float = 0.25,
        terrain_rows: int = 6,
        terrain_cols: int = 30,
        device: torch.device | str = "cpu",
        max_episode_length: int = 500,
        low_level: str = "official_onnx",
        low_level_checkpoint: str | Path | None = None,
        low_level_profile: str = "legacy",
        low_level_ready_after_reset: bool = False,
        lidar_encoder_checkpoint: str | Path | None = None,
        use_lidar: bool = True,
        use_height: bool = True,
        sensor_workers: int | None = None,
        physics_workers: int | None = None,
        lidar_horizontal_samples: int = 900,
        sensor_backend: str = "cpu",
        record_lidar_for_visualization: bool = False,
        record_contact_diagnostics: bool = False,
        record_raw_contact_diagnostics: bool = False,
        record_imu_history: bool = False,
        imu_sample_hz: float = 200.0,
        seed: int = 42,
        command_noise: float = 0.0,
        reset_mode: str = "fixed",
        reset_position_noise: float = 0.0,
        reset_yaw_noise: float = 0.0,
        reset_velocity_probability: float = 0.0,
        reset_forward_speed_min: float = 0.0,
        reset_forward_speed_max: float = 0.0,
        reset_lateral_speed_max: float = 0.0,
        reset_yaw_rate_max: float = 0.0,
        entry_state_bank: str | Path | None = None,
        entry_state_probability: float = 0.0,
        randomize_waypoint_yaw: bool = True,
        training_spawn_mode: str = "waypoint_start",
        waypoint_yaw_jitter_deg: float = 10.0,
        goal_min_gap: int = 1,
        goal_max_gap: int | None = None,
        single_episode_length: int | None = None,
        adaptive_segment_sampling: bool = False,
        adaptive_uniform_mix: float = 0.6,
        adaptive_ema_alpha: float = 0.05,
        adaptive_difficulty_power: float = 1.0,
        adaptive_warmup_attempts: int = 5,
        adaptive_max_probability: float = 1.0,
        adaptive_sampling_strategy: str = "difficulty",
        adaptive_progress_fast_alpha: float = 0.10,
        adaptive_progress_slow_alpha: float = 0.01,
        adaptive_progress_min_mastery: float = 0.05,
        adaptive_progress_max_mastery: float = 0.95,
        adaptive_progress_epsilon: float = 1.0e-3,
        safe_spawn_candidates: str | Path | None = None,
        contact_threshold: float = 500.0,
        terrain_fall_height: float = -2.0,
        include_self_contacts: bool = False,
        ignore_world_body_contacts: bool = True,
        reset_settle_physics_steps: int = 40,
        contact_persistence_steps: int = 1,
        waypoint_start_probability: float | None = None,
        reward_config: S10RewardConfig | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if low_level not in {"stand_pd", "official_onnx", "pim_him", "none"}:
            raise ValueError(
                "low_level must be 'official_onnx', 'pim_him', 'stand_pd', or 'none'"
            )
        if low_level_profile not in LOW_LEVEL_PROFILES:
            raise ValueError(
                f"low_level_profile must be one of {sorted(LOW_LEVEL_PROFILES)}"
            )
        if low_level == "pim_him" and low_level_profile != "official_20260828":
            raise ValueError(
                "pim_him requires low_level_profile='official_20260828' for its "
                "default pose and wheel damping contract"
            )
        if task_mode not in {"waypoint", "random_goal_sru"}:
            raise ValueError("task_mode must be 'waypoint' or 'random_goal_sru'")
        if reset_mode not in {"fixed", "random_waypoint"}:
            raise ValueError("reset_mode must be 'fixed' or 'random_waypoint'")
        if task_mode == "random_goal_sru" and terrain_cols != 30:
            raise ValueError("equivalent SRU random-goal training requires 30 terrain columns")
        if model_override is not None and task_mode != "waypoint":
            raise ValueError("model_override is only available for diagnostic waypoint scenes")
        if training_spawn_mode not in {"waypoint_start", "safe_fraction"}:
            raise ValueError(
                "training_spawn_mode must be 'waypoint_start' or 'safe_fraction'"
            )
        if not 0.0 <= waypoint_yaw_jitter_deg <= 180.0:
            raise ValueError("waypoint_yaw_jitter_deg must be in [0, 180]")
        self.num_envs = int(num_envs)
        self.task_mode = task_mode
        self.terrain_seed = seed if terrain_seed is None else int(terrain_seed)
        self.surface_seed = surface_seed
        self.terrain_profile = str(terrain_profile)
        self.grass_fraction = float(grass_fraction)
        self.gravel_fraction = float(gravel_fraction)
        self.terrain_atlas = None
        self.terrain_position_sampler = None
        self.max_episode_length = int(max_episode_length)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA learner requested but torch.cuda.is_available() is false")
        self.low_level = low_level
        self.low_level_profile = LOW_LEVEL_PROFILES[low_level_profile]
        self.low_level_ready_after_reset = bool(low_level_ready_after_reset)
        self.lidar_encoder_checkpoint = (
            None
            if lidar_encoder_checkpoint is None
            else Path(lidar_encoder_checkpoint).expanduser().resolve()
        )
        self.use_lidar = bool(use_lidar)
        self.use_height = bool(use_height)
        if sensor_workers is not None and sensor_workers < 1:
            raise ValueError("sensor_workers must be positive when specified")
        if physics_workers is not None and physics_workers < 1:
            raise ValueError("physics_workers must be positive when specified")
        if not (self.use_lidar or self.use_height):
            self.sensor_workers = 0
        elif sensor_workers is None:
            self.sensor_workers = min(self.num_envs, 8)
        else:
            self.sensor_workers = int(sensor_workers)
        self.physics_workers = (
            min(self.num_envs, 8) if physics_workers is None else int(physics_workers)
        )
        self.lidar_horizontal_samples = int(lidar_horizontal_samples)
        if self.lidar_horizontal_samples < 90 or self.lidar_horizontal_samples % 90 != 0:
            raise ValueError("lidar_horizontal_samples must be a multiple of 90")
        if sensor_backend not in {"cpu", "warp"}:
            raise ValueError("sensor_backend must be 'cpu' or 'warp'")
        if sensor_backend == "warp" and self.device.type != "cuda":
            raise ValueError("sensor_backend='warp' requires a CUDA device")
        self.sensor_backend = sensor_backend
        self.record_lidar_for_visualization = bool(record_lidar_for_visualization)
        self._visual_lidar_scans: tuple[np.ndarray, np.ndarray] | None = None
        self.record_contact_diagnostics = bool(record_contact_diagnostics)
        self.record_raw_contact_diagnostics = bool(record_raw_contact_diagnostics)
        if self.record_raw_contact_diagnostics and not self.record_contact_diagnostics:
            raise ValueError(
                "record_raw_contact_diagnostics requires record_contact_diagnostics"
            )
        self._contact_force_peaks_by_body: list[dict[str, float]] = [
            {} for _ in range(self.num_envs)
        ]
        self._contact_force_peaks_by_pair: list[dict[str, float]] = [
            {} for _ in range(self.num_envs)
        ]
        self._raw_contact_force_peaks_by_pair: list[dict[str, float]] = [
            {} for _ in range(self.num_envs)
        ]
        self.record_imu_history = bool(record_imu_history)
        if imu_sample_hz <= 0.0:
            raise ValueError("imu_sample_hz must be positive")
        self._requested_imu_sample_hz = float(imu_sample_hz)
        self.command_noise = float(command_noise)
        self.reset_mode = reset_mode
        self.reset_position_noise = float(reset_position_noise)
        self.reset_yaw_noise = float(reset_yaw_noise)
        self.reset_velocity_probability = float(reset_velocity_probability)
        self.reset_forward_speed_min = float(reset_forward_speed_min)
        self.reset_forward_speed_max = float(reset_forward_speed_max)
        self.reset_lateral_speed_max = float(reset_lateral_speed_max)
        self.reset_yaw_rate_max = float(reset_yaw_rate_max)
        self.entry_state_bank_path = (
            None if entry_state_bank is None else Path(entry_state_bank).expanduser().resolve()
        )
        if self.low_level == "pim_him" and self.entry_state_bank_path is not None:
            raise ValueError(
                "PIM-HIM cannot restore an official-ONNX entry-state bank because "
                "the bank does not contain its six-frame observation history"
            )
        self.waypoint_route_path = (
            None
            if waypoint_route is None
            else Path(waypoint_route).expanduser().resolve()
        )
        self.entry_state_probability = float(entry_state_probability)
        if not 0.0 <= self.reset_velocity_probability <= 1.0:
            raise ValueError("reset_velocity_probability must be in [0, 1]")
        if self.reset_forward_speed_min < 0.0:
            raise ValueError("reset_forward_speed_min must be non-negative")
        if self.reset_forward_speed_max < self.reset_forward_speed_min:
            raise ValueError("reset_forward_speed_max must be >= reset_forward_speed_min")
        if self.reset_lateral_speed_max < 0.0 or self.reset_yaw_rate_max < 0.0:
            raise ValueError("reset lateral speed and yaw rate limits must be non-negative")
        if not 0.0 <= self.entry_state_probability <= 1.0:
            raise ValueError("entry_state_probability must be in [0, 1]")
        if self.entry_state_probability > 0.0 and self.entry_state_bank_path is None:
            raise ValueError("entry_state_probability requires entry_state_bank")
        if self.entry_state_bank_path is not None and reset_mode != "random_waypoint":
            raise ValueError("entry-state banks require reset_mode='random_waypoint'")
        self.randomize_waypoint_yaw = bool(randomize_waypoint_yaw)
        self.training_spawn_mode = training_spawn_mode
        self.waypoint_start_probability = (
            float(waypoint_start_probability)
            if waypoint_start_probability is not None
            else (1.0 if training_spawn_mode == "waypoint_start" else 0.0)
        )
        if not 0.0 <= self.waypoint_start_probability <= 1.0:
            raise ValueError("waypoint_start_probability must be in [0, 1]")
        self.waypoint_yaw_jitter_deg = float(waypoint_yaw_jitter_deg)
        self.goal_min_gap = int(goal_min_gap)
        self.goal_max_gap = None if goal_max_gap is None else int(goal_max_gap)
        if self.goal_min_gap < 1:
            raise ValueError("goal_min_gap must be at least one waypoint")
        if self.goal_max_gap is not None and self.goal_max_gap < self.goal_min_gap:
            raise ValueError("goal_max_gap must be >= goal_min_gap")
        if single_episode_length is not None and single_episode_length < 1:
            raise ValueError("single_episode_length must be positive")
        self.single_episode_length = min(
            self.max_episode_length,
            self.max_episode_length if single_episode_length is None else int(single_episode_length),
        )
        self.adaptive_segment_sampling = bool(adaptive_segment_sampling)
        self.adaptive_uniform_mix = float(adaptive_uniform_mix)
        self.adaptive_ema_alpha = float(adaptive_ema_alpha)
        self.adaptive_difficulty_power = float(adaptive_difficulty_power)
        self.adaptive_warmup_attempts = int(adaptive_warmup_attempts)
        self.adaptive_max_probability = float(adaptive_max_probability)
        self.adaptive_sampling_strategy = adaptive_sampling_strategy
        self.adaptive_progress_fast_alpha = float(adaptive_progress_fast_alpha)
        self.adaptive_progress_slow_alpha = float(adaptive_progress_slow_alpha)
        self.adaptive_progress_min_mastery = float(adaptive_progress_min_mastery)
        self.adaptive_progress_max_mastery = float(adaptive_progress_max_mastery)
        self.adaptive_progress_epsilon = float(adaptive_progress_epsilon)
        self.contact_threshold = float(contact_threshold)
        self.terrain_fall_height = float(terrain_fall_height)
        self.include_self_contacts = bool(include_self_contacts)
        self.ignore_world_body_contacts = bool(ignore_world_body_contacts)
        self.reset_settle_physics_steps = int(reset_settle_physics_steps)
        self.contact_persistence_steps = int(contact_persistence_steps)
        if self.contact_threshold < 0.0:
            raise ValueError("contact_threshold must be non-negative")
        if self.reset_settle_physics_steps < 0:
            raise ValueError("reset_settle_physics_steps must be non-negative")
        if self.contact_persistence_steps < 1:
            raise ValueError("contact_persistence_steps must be positive")
        self.reward_config = reward_config or S10RewardConfig()
        if self.reward_config.required_time_at_goal <= 0.0:
            raise ValueError("required_time_at_goal must be positive")
        if self.reward_config.goal_progress < 0.0:
            raise ValueError("goal_progress must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.action_spec = S10ActionSpec()
        self.physics_steps = self.action_spec.physics_steps_per_policy_step
        self.low_level_decimation = self.action_spec.low_level_decimation
        self.dt = self.action_spec.mujoco_dt
        self.required_goal_hold_steps = max(
            1,
            int(
                round(
                    self.reward_config.required_time_at_goal
                    / (self.physics_steps * self.dt)
                )
            ),
        )
        self.imu_decimation = max(
            1, int(round(1.0 / (self._requested_imu_sample_hz * self.dt)))
        )
        self.imu_sample_hz = 1.0 / (self.imu_decimation * self.dt)
        self._imu_histories: list[list[tuple[object, ...]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._imu_capture_substeps = np.zeros(self.num_envs, dtype=np.int64)
        self._onnx_inference_calls = 0
        self.reset_velocity_body = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.reset_yaw_rate = np.zeros(self.num_envs, dtype=np.float64)
        self.entry_state_applied = np.zeros(self.num_envs, dtype=bool)
        self.entry_state_samples = 0
        self.entry_state_fallbacks = 0
        self._entry_filtered_cmd = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._entry_filter_alpha = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._entry_policy_scale = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._entry_policy_bias = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._pending_entry_snapshots: list[dict[str, Any] | None] = [
            None for _ in range(self.num_envs)
        ]

        self.xml_path = Path(xml_path).expanduser().resolve()
        if model_override is not None:
            self.simulation_assets_sha256 = _sha256_tree(
                REPO_ROOT / "src/S10_sdk_deploy/S10_description/s10_mjcf"
            )
            self.model = model_override
            self.terrain_levels = None
            self.terrain_types = None
            self.environment_tile_indices = None
        elif self.task_mode == "random_goal_sru":
            self.terrain_atlas = generate_sru_atlas(
                SruAtlasConfig(
                    seed=self.terrain_seed,
                    surface_seed=self.surface_seed,
                    terrain_profile=self.terrain_profile,
                    grass_fraction=self.grass_fraction,
                    gravel_fraction=self.gravel_fraction,
                    num_rows=int(terrain_rows),
                    num_cols=int(terrain_cols),
                )
            )
            self.model = build_sru_mujoco_model(self.terrain_atlas)
            self.simulation_assets_sha256 = _sha256_tree(
                REPO_ROOT / "src/S10_sdk_deploy/S10_description/s10_mjcf"
            )
            self.terrain_position_sampler = SruPositionSampler(self.terrain_atlas)
            (
                self.terrain_levels,
                self.terrain_types,
                self.environment_tile_indices,
            ) = assign_terrain_tiles(self.num_envs, self.terrain_atlas, self.rng)
        else:
            self.simulation_assets_sha256 = _sha256_tree(self.xml_path.parent.parent)
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.terrain_levels = None
            self.terrain_types = None
            self.environment_tile_indices = None
        self.model.opt.timestep = self.dt
        if self.model.nu != DOF or self.model.nq < ROOT_QPOS + DOF:
            raise RuntimeError(f"Unexpected S10 model dimensions nq={self.model.nq}, nu={self.model.nu}")
        # The imported terrain is itself attached to a free body with a
        # non-zero XML pose. Preserve that joint state when resetting the
        # robot; clearing all qpos would silently move the terrain away from
        # the waypoint overlay.
        template = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, template)
        self.static_qpos = template.qpos[ROOT_QPOS + DOF:].copy()
        self.data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
        if self.base_body_id < 0:
            raise RuntimeError("S10 MuJoCo model is missing base_link")
        self.terrain_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main_body")
        self.contact_body_ids = {
            body_id
            for body_id in (
                self.base_body_id,
                *(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                  for name in ("fl_hipx", "fr_hipx", "hl_hipx", "hr_hipx",
                               "fl_hipy", "fr_hipy", "hl_hipy", "hr_hipy")),
            )
            if body_id >= 0
        }
        self.terrain_contact_body_ids = {0}
        if self.terrain_body_id >= 0:
            self.terrain_contact_body_ids.add(self.terrain_body_id)
        self.robot_body_ids = set(range(1, self.model.nbody)) - self.terrain_contact_body_ids
        terrain_ray_body_ids = {
            int(self.model.geom_bodyid[geom_id])
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_group[geom_id]) == 0
        }
        if not terrain_ray_body_ids.issubset(self.terrain_contact_body_ids):
            raise RuntimeError(
                "terrain reset ray group contains non-terrain bodies: "
                f"{sorted(terrain_ray_body_ids - self.terrain_contact_body_ids)}"
            )

        self.joint_target = self.low_level_profile.default_robot_array.astype(
            np.float64, copy=True
        )
        self.kp = np.asarray((80.0, 80.0, 80.0, 0.0) * 4, dtype=np.float64)
        self.kd = np.asarray(
            (2.0, 2.0, 2.0, self.low_level_profile.wheel_kd) * 4,
            dtype=np.float64,
        )
        self.ctrl_range = self.model.actuator_ctrlrange[:DOF].copy()
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        # This counter resets at target switches during continuous evaluation.
        self.target_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.last_cmd = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._previous_high_level_action = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._pending_high_level_action: np.ndarray | None = None
        self._reward_joint_acc = np.zeros((self.num_envs, DOF), dtype=np.float64)
        self._goal_hold_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._goal_was_reached = np.zeros(self.num_envs, dtype=bool)
        self._max_illegal_contact_force = np.zeros(self.num_envs, dtype=np.float64)
        self._last_illegal_contact_force = np.zeros(self.num_envs, dtype=np.float64)
        self._illegal_contact_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_done_reason = np.full(self.num_envs, DONE_NONE, dtype=object)
        self.last_reward_components: dict[str, np.ndarray] = {}
        self.last_termination_components: dict[str, np.ndarray] = {}
        self.metrics = BackendMetrics()
        self.learned_controller = None
        self.onnx_controller = None
        if self.low_level == "official_onnx":
            checkpoint = Path(
                low_level_checkpoint or REPO_ROOT / "src/S10_sdk_deploy/policy/policy.onnx"
            ).expanduser().resolve()
            self.low_level_checkpoint_sha256 = _sha256_file(checkpoint)
            self.onnx_controller = OfficialS10OnnxController(
                checkpoint, profile=self.low_level_profile
            )
            self.onnx_controller.reset(self.num_envs)
            self.learned_controller = self.onnx_controller
        elif self.low_level == "pim_him":
            if low_level_checkpoint is None:
                raise ValueError("pim_him requires an explicit low_level_checkpoint")
            checkpoint = Path(low_level_checkpoint).expanduser().resolve()
            self.low_level_checkpoint_sha256 = _sha256_file(checkpoint)
            from s10_locomotion.deploy.mujoco_controller import MujocoPIMHIMController

            self.learned_controller = MujocoPIMHIMController(
                self.model, checkpoint, device=str(self.device)
            )
            self.learned_controller.reset(self.num_envs)
        else:
            self.low_level_checkpoint_sha256 = None

        self.waypoints = self._load_waypoints()
        self.entry_state_bank = None
        self.entry_state_bank_sha256 = None
        if self.entry_state_bank_path is not None:
            self.entry_state_bank_sha256 = _sha256_file(self.entry_state_bank_path)
            self.entry_state_bank = EntryStateBank(
                self.entry_state_bank_path,
                contract=EntryStateBankContract(
                    nq=self.model.nq,
                    nv=self.model.nv,
                    nu=self.model.nu,
                    dof=DOF,
                    simulation_assets_sha256=self.simulation_assets_sha256,
                    low_level_checkpoint_sha256=self.low_level_checkpoint_sha256,
                    waypoint_sha256=self.waypoint_sha256(),
                ),
            )
        self.safe_spawn_sha256: str | None = None
        self.safe_spawn_yaw_offsets: np.ndarray | None = None
        self.safe_spawn_candidates = self._load_safe_spawn_candidates(safe_spawn_candidates)
        self.strict_spawn_candidates = self._build_waypoint_start_candidates()
        self.training_spawn_candidates = self._build_training_spawn_candidates()
        if self.safe_spawn_candidates is not None:
            all_segments = np.unique(self.training_spawn_candidates[:, 0]).astype(np.int64)
            eligible_segments = all_segments
            eligible_segments = eligible_segments[
                ~np.isin(
                    eligible_segments,
                    np.asarray(tuple(DISABLED_ROUTE_START_SEGMENTS), dtype=np.int64),
                )
            ]
            self.segment_sampler = AdaptiveSegmentSampler(
                eligible_segments,
                enabled=self.adaptive_segment_sampling,
                uniform_mix=self.adaptive_uniform_mix,
                ema_alpha=self.adaptive_ema_alpha,
                difficulty_power=self.adaptive_difficulty_power,
                warmup_attempts=self.adaptive_warmup_attempts,
                max_probability=self.adaptive_max_probability,
                strategy=self.adaptive_sampling_strategy,
                progress_fast_alpha=self.adaptive_progress_fast_alpha,
                progress_slow_alpha=self.adaptive_progress_slow_alpha,
                progress_min_mastery=self.adaptive_progress_min_mastery,
                progress_max_mastery=self.adaptive_progress_max_mastery,
                progress_epsilon=self.adaptive_progress_epsilon,
            )
        else:
            self.segment_sampler = None
        self.start_waypoint_indices = np.zeros(self.num_envs, dtype=np.int64)
        self.goal_waypoint_indices = np.zeros(self.num_envs, dtype=np.int64)
        self.final_waypoint_indices = np.zeros(self.num_envs, dtype=np.int64)
        self.context_final_waypoint_indices = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.waypoint_task_types = np.full(self.num_envs, TASK_SINGLE, dtype=np.int64)
        self.waypoints_reached_this_step = np.zeros(self.num_envs, dtype=np.int64)
        self.reached_waypoint_indices_this_step = np.full(
            self.num_envs, -1, dtype=np.int64
        )
        self._target_distance_xy_this_step = np.zeros(self.num_envs, dtype=np.float64)
        self._previous_goal_distance_xy = np.zeros(self.num_envs, dtype=np.float64)
        self._target_height_error_this_step = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.random_spawn_positions = np.zeros(
            (self.num_envs, 3), dtype=np.float64
        )
        self.random_goal_positions = np.zeros(
            (self.num_envs, 3), dtype=np.float64
        )
        self.episode_step_limits = np.full(
            self.num_envs, self.single_episode_length, dtype=np.int64
        )
        self.sensor_dirs = build_sensor_frame_directions(self.lidar_horizontal_samples)
        self.lidar_samplers = [
            S10LidarSampler(
                self.model, None, horizontal_samples=self.lidar_horizontal_samples
            )
            for _ in range(self.num_envs)
        ] if self.use_lidar and self.sensor_backend == "cpu" else []
        # MuJoCo ray queries are read-only with respect to MjModel/MjData, but
        # each scanner owns mutable output buffers. Keep one scanner per
        # environment so independent sensor captures can safely run in C from
        # a persistent thread pool.
        self.height_scanners = [MuJoCoHeightScan(self.model) for _ in range(self.num_envs)] if self.use_height else []
        self._sensor_executor = (
            ThreadPoolExecutor(max_workers=self.sensor_workers, thread_name_prefix="s10-sensor")
            if self.sensor_workers > 1 else None
        )
        if self.low_level == "pim_him":
            self.learned_controller.sensor_executor = self._sensor_executor
        self._physics_chunks = tuple(
            chunk for chunk in np.array_split(np.arange(self.num_envs), self.physics_workers)
            if len(chunk)
        )
        self._physics_executor = (
            ThreadPoolExecutor(max_workers=self.physics_workers, thread_name_prefix="s10-physics")
            if self.physics_workers > 1 else None
        )
        self.lidar_encoder = (
            S10LegacyLidarEncoder(
                device=self.device,
                **(
                    {}
                    if self.lidar_encoder_checkpoint is None
                    else {"checkpoint": self.lidar_encoder_checkpoint}
                ),
            )
            if self.use_lidar
            else None
        )
        self.height_encoder = HeightFeatureEncoder(device=self.device) if self.use_height else None
        self.lidar_encoder_sha256 = (
            _sha256_file(self.lidar_encoder.checkpoint_path)
            if self.lidar_encoder is not None else None
        )
        self.height_encoder_sha256 = (
            _sha256_file(self.height_encoder.checkpoint_path)
            if self.height_encoder is not None else None
        )
        if self.use_lidar and self.sensor_backend == "warp":
            from .s10_warp_raycast import WarpStaticTerrainLidar

            self.warp_lidar = WarpStaticTerrainLidar(
                self.model,
                horizontal_samples=self.lidar_horizontal_samples,
                device=str(self.device),
            )
        else:
            self.warp_lidar = None

        self.reset()

    def _load_waypoints(self) -> np.ndarray:
        if self.task_mode == "random_goal_sru":
            self.waypoint_names = ("random_spawn", "random_goal")
            return np.zeros((2, 3), dtype=np.float64)
        if self.waypoint_route_path is not None:
            if not self.waypoint_route_path.is_file():
                raise FileNotFoundError(
                    f"waypoint route does not exist: {self.waypoint_route_path}"
                )
            payload = yaml.safe_load(
                self.waypoint_route_path.read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise RuntimeError(
                    f"unsupported waypoint route schema: {self.waypoint_route_path}"
                )
            if payload.get("skipped_waypoints") != sorted(SKIPPED_ROUTE_WAYPOINTS):
                raise RuntimeError(
                    "waypoint route skipped_waypoints must match the backend route protocol"
                )
            nodes = payload.get("nodes")
            if not isinstance(nodes, list) or len(nodes) < 2:
                raise RuntimeError("waypoint route must define at least two nodes")
            names: list[str] = []
            positions: list[np.ndarray] = []
            for index, node in enumerate(nodes):
                if not isinstance(node, dict):
                    raise RuntimeError(f"waypoint route node {index} is not a mapping")
                name = str(node.get("name", "")).strip()
                position = np.asarray(node.get("position"), dtype=np.float64)
                if not name or position.shape != (3,) or not np.isfinite(position).all():
                    raise RuntimeError(f"invalid waypoint route node {index}: {node}")
                enabled = bool(node.get("enabled", True))
                if enabled == (index in SKIPPED_ROUTE_WAYPOINTS):
                    raise RuntimeError(
                        f"waypoint route node {index} enabled state conflicts with skipped_waypoints"
                    )
                names.append(name)
                positions.append(position)
            if len(set(names)) != len(names):
                raise RuntimeError("waypoint route node names must be unique")
            self.waypoint_names = tuple(names)
            return np.stack(positions, axis=0)

        points: dict[int, np.ndarray] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name or not name.startswith(TRACK_WAYPOINT_PREFIX):
                continue
            suffix = name[len(TRACK_WAYPOINT_PREFIX):].split("_", 1)[0]
            if suffix.isdigit():
                points[int(suffix)] = self.model.geom_pos[geom_id].copy()
        if not points:
            route = np.asarray(((0.0, -2.5, 0.0),), dtype=np.float64)
        else:
            route = np.stack([points[i] for i in range(max(points) + 1)], axis=0)
        self.waypoint_names = tuple(f"official_{index:02d}" for index in range(len(route)))
        return route

    def waypoint_sha256(self) -> str:
        """Return a stable fingerprint for the route used by spawn calibration."""

        route = np.ascontiguousarray(self.waypoints, dtype="<f8")
        return hashlib.sha256(route.tobytes()).hexdigest()

    def _load_safe_spawn_candidates(
        self, path: str | Path | None
    ) -> np.ndarray | None:
        if self.task_mode == "random_goal_sru":
            return None
        if self.reset_mode != "random_waypoint":
            return None
        if path is None:
            raise ValueError(
                "random_waypoint reset requires calibrated safe_spawn_candidates"
            )
        candidate_path = Path(path).expanduser().resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"Safe spawn calibration is missing: {candidate_path}. Run "
                "python -m sru_training.calibrate_s10_spawn_candidates first."
            )
        candidate_bytes = candidate_path.read_bytes()
        payload = json.loads(candidate_bytes.decode("utf-8"))
        if payload.get("schema_version") != 4:
            raise RuntimeError(
                f"Safe spawn calibration uses the obsolete physical-segment schema: "
                f"{candidate_path}. This legacy waypoint asset is not compatible with "
                "the current logical route-edge schema."
            )
        expected_route_protocol = {
            "skipped_waypoints": sorted(SKIPPED_ROUTE_WAYPOINTS),
            "edge_semantics": "next_enabled_waypoint",
        }
        if payload.get("route_protocol") != expected_route_protocol:
            raise RuntimeError(
                f"Safe spawn calibration route protocol does not match: {candidate_path}"
            )
        expected_hash = self.waypoint_sha256()
        if payload.get("waypoint_sha256") != expected_hash:
            raise RuntimeError(
                f"Safe spawn calibration does not match the loaded route: {candidate_path}"
            )
        candidates = np.asarray(
            [(item["segment"], item["fraction"]) for item in payload.get("safe_candidates", ())],
            dtype=np.float64,
        )
        if candidates.ndim != 2 or candidates.shape[1:] != (2,) or not len(candidates):
            raise RuntimeError(f"Safe spawn calibration contains no candidates: {candidate_path}")
        calibration = payload.get("calibration", {})
        if (
            calibration.get("contact_threshold") != self.contact_threshold
            or calibration.get("contact_persistence_steps")
            != self.contact_persistence_steps
            or calibration.get("ignore_world_body_contacts")
            != self.ignore_world_body_contacts
            or calibration.get("contact_force_statistic")
            != "max_selected_contact_over_policy_step"
        ):
            raise RuntimeError(
                f"Safe spawn calibration contact contract does not match: {candidate_path}"
            )
        max_layer_envelope_error = calibration.get("max_layer_envelope_error")
        if (
            not isinstance(max_layer_envelope_error, (int, float))
            or not np.isfinite(max_layer_envelope_error)
            or max_layer_envelope_error < 0.0
        ):
            raise RuntimeError(
                f"Safe spawn calibration has no valid route-layer contract: {candidate_path}"
            )
        if any(
            not isinstance(item.get("max_layer_envelope_error"), (int, float))
            or float(item["max_layer_envelope_error"]) > float(max_layer_envelope_error)
            for item in payload.get("safe_candidates", ())
        ):
            raise RuntimeError(
                f"Safe spawn calibration contains a wrong-layer candidate: {candidate_path}"
            )
        min_goal_distance = calibration.get("min_goal_distance")
        settled_min_goal_distance = calibration.get("settled_min_goal_distance")
        if (
            not isinstance(min_goal_distance, (int, float))
            or not np.isfinite(min_goal_distance)
            or min_goal_distance <= self.reward_config.goal_distance_threshold
            or not isinstance(settled_min_goal_distance, (int, float))
            or not np.isfinite(settled_min_goal_distance)
            or settled_min_goal_distance <= self.reward_config.goal_distance_threshold
            or settled_min_goal_distance > min_goal_distance
            or any(
                not isinstance(item.get("min_settled_goal_distance"), (int, float))
                or float(item["min_settled_goal_distance"])
                < float(settled_min_goal_distance)
                for item in payload.get("safe_candidates", ())
            )
        ):
            raise RuntimeError(
                f"Safe spawn calibration violates its settled goal margin: {candidate_path}"
            )
        segments = candidates[:, 0]
        fractions = candidates[:, 1]
        if (
            not np.equal(segments, np.floor(segments)).all()
            or (segments < 0).any()
            or (segments >= len(self.waypoints) - 1).any()
            or (fractions < 0.0).any()
            or (fractions > 1.0).any()
        ):
            raise RuntimeError(f"Safe spawn calibration has invalid entries: {candidate_path}")
        logical_goals = np.asarray(
            [
                next_route_waypoint(int(segment), len(self.waypoints) - 1)
                for segment in segments
            ],
            dtype=np.int64,
        )
        edge_lengths = np.linalg.norm(
            self.waypoints[logical_goals, :2]
            - self.waypoints[segments.astype(np.int64), :2],
            axis=1,
        )
        initial_goal_distances = edge_lengths * (1.0 - fractions)
        if np.any(initial_goal_distances <= self.reward_config.goal_distance_threshold):
            raise RuntimeError(
                "Safe spawn calibration contains a reset inside the first-entry "
                f"goal threshold: {candidate_path}"
            )
        eligible = np.asarray(
            [
                int(segment) not in DISABLED_ROUTE_START_SEGMENTS
                and route_target_count(int(segment), len(self.waypoints) - 1)
                >= self.goal_min_gap
                for segment in segments
            ],
            dtype=bool,
        )
        candidates = candidates[eligible]
        if not len(candidates):
            raise RuntimeError(
                "Safe spawn calibration has no candidate compatible with goal_min_gap="
                f"{self.goal_min_gap}"
            )
        if self.randomize_waypoint_yaw:
            yaw_offsets_deg = payload.get("calibration", {}).get("yaw_offsets_deg")
            if not isinstance(yaw_offsets_deg, list) or not yaw_offsets_deg:
                raise RuntimeError(
                    f"Safe spawn calibration has no yaw coverage: {candidate_path}. "
                    "Run python -m sru_training.calibrate_s10_spawn_candidates again."
                )
            yaw_offsets = np.deg2rad(np.asarray(yaw_offsets_deg, dtype=np.float64))
            if yaw_offsets.ndim != 1 or not np.isfinite(yaw_offsets).all():
                raise RuntimeError(f"Safe spawn calibration has invalid yaw offsets: {candidate_path}")
            self.safe_spawn_yaw_offsets = yaw_offsets
        self.safe_spawn_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
        print(
            f"[S10 reset] loaded {len(candidates)} calibrated spawn candidates "
            f"from {candidate_path}",
            flush=True,
        )
        return candidates

    def _build_training_spawn_candidates(self) -> np.ndarray | None:
        """Select the calibrated reset support used by sampled training tasks."""

        if self.safe_spawn_candidates is None:
            return None
        if self.waypoint_start_probability < 1.0:
            return self.safe_spawn_candidates.copy()

        return self.strict_spawn_candidates.copy()

    def _build_waypoint_start_candidates(self) -> np.ndarray | None:
        """Build one strict start per logical segment from calibrated support."""

        if self.safe_spawn_candidates is None:
            return None

        selected = []
        for segment in np.unique(self.safe_spawn_candidates[:, 0]).astype(np.int64):
            segment_candidates = self.safe_spawn_candidates[
                self.safe_spawn_candidates[:, 0] == segment
            ]
            # Exact waypoint starts are the contract. A segment may use the
            # nearest calibrated fallback only when fraction=0 is physically
            # unstable (currently the final 31->32 edge).
            selected.append(segment_candidates[np.argmin(segment_candidates[:, 1])])
        candidates = np.asarray(selected, dtype=np.float64)
        exact_count = int(np.count_nonzero(candidates[:, 1] == 0.0))
        fallback = {
            int(segment): float(fraction)
            for segment, fraction in candidates
            if fraction != 0.0
        }
        print(
            "[S10 reset] training_spawn_mode=waypoint_start "
            f"exact_waypoints={exact_count}/{len(candidates)} fallback={fallback}; "
            f"yaw_jitter_deg={self.waypoint_yaw_jitter_deg if self.randomize_waypoint_yaw else 0.0}",
            flush=True,
        )
        return candidates

    def _sample_training_task(
        self, env_index: int | None = None
    ) -> tuple[int, int, int, int, int]:
        """Sample one independent adjacent-goal training episode."""

        if self.segment_sampler is None:
            raise RuntimeError("random waypoint reset has no segment sampler")
        start = self.segment_sampler.sample(self.rng)
        targets = route_waypoint_sequence(start, len(self.waypoints) - 1)
        max_gap = min(self.goal_max_gap or len(targets), len(targets))
        if max_gap < self.goal_min_gap:
            raise RuntimeError(
                f"sampled route start {start} has fewer than "
                f"{self.goal_min_gap} enabled targets"
            )
        gap = int(self.rng.integers(self.goal_min_gap, max_gap + 1))
        goal = targets[gap - 1]
        return start, goal, goal, TASK_SINGLE, -1

    def _configure_episode_task(
        self,
        index: int,
        *,
        start: int,
        active_goal: int,
        final_goal: int,
        task_type: int,
        template_id: int,
    ) -> None:
        self.start_waypoint_indices[index] = start
        self.goal_waypoint_indices[index] = active_goal
        self.final_waypoint_indices[index] = final_goal
        self.context_final_waypoint_indices[index] = final_goal
        self.waypoint_task_types[index] = task_type
        self.waypoints_reached_this_step[index] = 0
        self.reached_waypoint_indices_this_step[index] = -1
        self.episode_step_limits[index] = (
            self.single_episode_length
            if task_type == TASK_SINGLE
            else self.max_episode_length
        )

    def _reset_one(
        self,
        index: int,
        *,
        settle: bool = True,
        clear_done_reason: bool = True,
        route_start: int | None = None,
        route_goal: int | None = None,
        route_fraction: float | None = None,
        route_yaw: float | None = None,
        route_sequential: bool = False,
        route_context_final: int | None = None,
    ) -> None:
        if self.task_mode == "random_goal_sru":
            if any(
                value is not None
                for value in (
                    route_start,
                    route_goal,
                    route_fraction,
                    route_yaw,
                    route_context_final,
                )
            ) or route_sequential:
                raise ValueError("route-specific reset is unavailable in random_goal_sru mode")
            self._reset_one_random_goal(
                index,
                settle=settle,
                clear_done_reason=clear_done_reason,
            )
            return
        data = self.data[index]
        self.entry_state_applied[index] = False
        self._pending_entry_snapshots[index] = None
        data.qpos[:] = self.model.qpos0
        data.qpos[ROOT_QPOS + DOF:] = self.static_qpos
        data.qvel[:] = 0.0
        # Training starts at the state handed to RL control, after the
        # official stand-up sequence, rather than at the crouched ROS boot
        # pose.  The standing wheel centers are one wheel radius above ground.
        position = self.waypoints[0, :2].copy()
        position_z = float(self.waypoints[0, 2])
        first_delta = self.waypoints[min(1, len(self.waypoints) - 1), :2] - self.waypoints[0, :2]
        yaw = float(np.arctan2(first_delta[1], first_delta[0])) if np.linalg.norm(first_delta) else 0.0
        start_index = 0
        goal_index = len(self.waypoints) - 1
        final_index = goal_index
        task_type = TASK_SINGLE
        template_id = -1
        if route_start is not None:
            if len(self.waypoints) < 2:
                raise ValueError("route-specific reset requires at least two waypoints")
            start_index = int(route_start)
            if not 0 <= start_index < len(self.waypoints) - 1:
                raise ValueError(f"route_start must be in [0, {len(self.waypoints) - 2}]")
            final_index = (
                next_route_waypoint(start_index, len(self.waypoints) - 1)
                if route_goal is None
                else int(route_goal)
            )
            if not start_index < final_index < len(self.waypoints):
                raise ValueError("route_goal must be after route_start and within the waypoint list")
            if route_sequential:
                task_type = TASK_SEQUENTIAL_ROUTE
                goal_index = next_route_waypoint(start_index, final_index)
            else:
                if final_index in SKIPPED_ROUTE_WAYPOINTS:
                    raise ValueError(
                        f"waypoint {final_index} is disabled as a navigation target"
                    )
                goal_index = final_index
            fraction = 0.5 if route_fraction is None else float(route_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("route_fraction must be in [0, 1]")
            spawn_target_index = next_route_waypoint(start_index, final_index)
            position = self.waypoints[start_index, :2] * (1.0 - fraction) + self.waypoints[spawn_target_index, :2] * fraction
            position_z = float(
                self.waypoints[start_index, 2] * (1.0 - fraction)
                + self.waypoints[spawn_target_index, 2] * fraction
            )
            tangent = self.waypoints[spawn_target_index, :2] - self.waypoints[start_index, :2]
            yaw = float(np.arctan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) else 0.0
        elif self.reset_mode == "random_waypoint" and len(self.waypoints) > 1:
            if self.training_spawn_candidates is None:
                raise RuntimeError("random_waypoint reset has no calibrated spawn candidates")
            (
                start_index,
                goal_index,
                final_index,
                task_type,
                template_id,
            ) = self._sample_training_task(index)
            strict_spawn = bool(
                self.rng.random() < self.waypoint_start_probability
            )
            spawn_candidates = (
                self.strict_spawn_candidates if strict_spawn else self.safe_spawn_candidates
            )
            if spawn_candidates is None:
                raise RuntimeError("random waypoint reset has no spawn candidates")
            segment_candidates = spawn_candidates[spawn_candidates[:, 0] == start_index]
            # Training always samples one independent adjacent target.
            candidate = segment_candidates[
                int(self.rng.integers(0, len(segment_candidates)))
            ]
            segment = start_index
            fraction = float(candidate[1])
            spawn_target_index = next_route_waypoint(segment, final_index)
            position = self.waypoints[segment, :2] * (1.0 - fraction) + self.waypoints[spawn_target_index, :2] * fraction
            position_z = float(
                self.waypoints[segment, 2] * (1.0 - fraction)
                + self.waypoints[spawn_target_index, 2] * fraction
            )
            tangent = self.waypoints[spawn_target_index, :2] - self.waypoints[segment, :2]
            yaw = float(np.arctan2(tangent[1], tangent[0]))
            if self.randomize_waypoint_yaw:
                if strict_spawn:
                    yaw += float(
                        self.rng.uniform(
                            -np.deg2rad(self.waypoint_yaw_jitter_deg),
                            np.deg2rad(self.waypoint_yaw_jitter_deg),
                        )
                    )
                else:
                    if self.safe_spawn_yaw_offsets is None:
                        raise RuntimeError("random waypoint yaw has no calibrated offsets")
                    yaw += float(
                        self.safe_spawn_yaw_offsets[
                            int(self.rng.integers(0, len(self.safe_spawn_yaw_offsets)))
                        ]
                    )
        if route_yaw is not None:
            yaw = float(route_yaw)
        if (
            route_start is None
            and self.entry_state_bank is not None
            and self.entry_state_probability > 0.0
            and float(self.rng.random()) < self.entry_state_probability
        ):
            snapshot = self.entry_state_bank.sample(start_index, self.rng)
            if snapshot is None:
                self.entry_state_fallbacks += 1
            else:
                self._pending_entry_snapshots[index] = snapshot
        if self.reset_position_noise > 0.0:
            position += self.rng.normal(0.0, self.reset_position_noise, 2)
        if self.reset_yaw_noise > 0.0:
            yaw += float(self.rng.normal(0.0, self.reset_yaw_noise))
        # Search only near the route's expected layer. Casting from z=20 would
        # place final-section starts on an overhead bridge, while using linear
        # waypoint z directly can embed the robot in a stair tread.
        surface_z = self._terrain_z(
            data,
            position,
            ray_start_z=position_z + 0.75,
            fallback_height=position_z,
        )
        data.qpos[:3] = (
            position[0],
            position[1],
            surface_z + STANDING_BASE_CLEARANCE,
        )
        data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        data.qpos[7:7 + DOF] = self.joint_target
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)
        # Let the imported MuJoCo contact geometry settle before the first RL
        # transition. This removes reset impact impulses from base_contact;
        # contacts generated after this window remain task terminations.
        if settle:
            self._settle_reset_one(index)
        self.episode_steps[index] = 0
        self.target_steps[index] = 0
        self._configure_episode_task(
            index,
            start=start_index,
            active_goal=goal_index,
            final_goal=final_index,
            task_type=task_type,
            template_id=template_id,
        )
        if route_context_final is not None:
            context_final = int(route_context_final)
            if not goal_index <= context_final < len(self.waypoints):
                raise ValueError(
                    "route_context_final must be at or after the active goal"
                )
            self.context_final_waypoint_indices[index] = context_final
        self.last_cmd[index] = 0.0
        self._goal_hold_steps[index] = 0
        self._goal_was_reached[index] = False
        self._max_illegal_contact_force[index] = 0.0
        self._last_illegal_contact_force[index] = 0.0
        self._illegal_contact_steps[index] = 0
        self._previous_high_level_action[index] = 0.0
        self.last_action[index].zero_()
        self._reward_joint_acc[index] = 0.0
        if clear_done_reason:
            self.last_done_reason[index] = DONE_NONE
        if settle:
            self._finalize_reset_state(index)

    def _reset_one_random_goal(
        self,
        index: int,
        *,
        settle: bool,
        clear_done_reason: bool,
    ) -> None:
        if self.terrain_position_sampler is None or self.environment_tile_indices is None:
            raise RuntimeError("random-goal terrain sampling was not initialized")

        data = self.data[index]
        self.entry_state_applied[index] = False
        self._pending_entry_snapshots[index] = None
        data.qpos[:] = self.model.qpos0
        data.qpos[ROOT_QPOS + DOF:] = self.static_qpos
        data.qvel[:] = 0.0

        sample = self.terrain_position_sampler.sample(
            int(self.environment_tile_indices[index]), self.rng
        )
        self.random_spawn_positions[index] = sample.spawn_world
        self.random_goal_positions[index] = sample.goal_world
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        data.qpos[:3] = (
            sample.spawn_world[0],
            sample.spawn_world[1],
            sample.spawn_world[2] + 0.05 + STANDING_BASE_CLEARANCE,
        )
        data.qpos[3:7] = (
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        )
        data.qpos[7:7 + DOF] = self.joint_target
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)
        if settle:
            self._settle_reset_one(index)

        self.episode_steps[index] = 0
        self.target_steps[index] = 0
        self._configure_episode_task(
            index,
            start=0,
            active_goal=1,
            final_goal=1,
            task_type=TASK_SINGLE,
            template_id=-1,
        )
        self.last_cmd[index] = 0.0
        self._goal_hold_steps[index] = 0
        self._goal_was_reached[index] = False
        self._max_illegal_contact_force[index] = 0.0
        self._last_illegal_contact_force[index] = 0.0
        self._illegal_contact_steps[index] = 0
        self._previous_high_level_action[index] = 0.0
        self.last_action[index].zero_()
        self._reward_joint_acc[index] = 0.0
        if clear_done_reason:
            self.last_done_reason[index] = DONE_NONE
        if settle:
            self._finalize_reset_state(index)

    def _settle_reset_one(self, index: int) -> None:
        data = self.data[index]
        for _ in range(self.reset_settle_physics_steps):
            if self.low_level in {"official_onnx", "pim_him", "stand_pd"}:
                # Use deterministic standing PD while settling. Advancing the
                # recurrent ONNX action state here would make reset history
                # depend on how many environments ended together.
                self._apply_low_level(data, np.zeros(3, dtype=np.float64))
            mujoco.mj_step(self.model, data)

    def _settle_reset_chunk(self, indices: np.ndarray) -> None:
        for index in indices:
            self._settle_reset_one(int(index))

    def _reset_indices(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return
        if self.learned_controller is not None:
            self.learned_controller.reset(self.num_envs, indices)
        # Keep route RNG consumption deterministic on the caller thread, then
        # settle independent MjData instances concurrently.
        for index in indices:
            self._reset_one(int(index), settle=False)
        if self.reset_settle_physics_steps <= 0:
            for index in indices:
                self._finalize_reset_state(int(index))
            self._mark_low_level_ready(indices)
            return
        chunks = tuple(
            chunk for chunk in np.array_split(indices, min(self.physics_workers, len(indices)))
            if len(chunk)
        )
        if self._physics_executor is None:
            self._settle_reset_chunk(chunks[0])
        else:
            futures = [self._physics_executor.submit(self._settle_reset_chunk, chunk) for chunk in chunks]
            for future in futures:
                future.result()
        for index in indices:
            self._finalize_reset_state(int(index))
        self._mark_low_level_ready(indices)

    def _mark_low_level_ready(self, indices: np.ndarray) -> None:
        if self.onnx_controller is not None and self.low_level_ready_after_reset:
            # An RL reset places the robot directly in its settled control
            # pose; it is not another hardware state-machine transition. The
            # official 150-cycle ramp remains enabled for GUI/deployment starts.
            self.onnx_controller.run_count[indices] = (
                self.low_level_profile.startup_ramp_steps
            )
            self.onnx_controller.kp_scale[indices] = 1.0
            self.onnx_controller.kd_scale[indices] = 1.0

    def _finalize_reset_state(self, index: int) -> None:
        snapshot = self._pending_entry_snapshots[index]
        if snapshot is None:
            self._initialize_reset_velocity(index)
            self._previous_goal_distance_xy[index] = self._distance_to_goal(index, flat=True)
            return
        data = self.data[index]
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.ctrl[:] = snapshot["ctrl"]
        mujoco.mj_forward(self.model, data)
        if self.onnx_controller is not None:
            self.onnx_controller.last_action[index] = snapshot["onnx_last_action"]
        self.last_cmd[index] = snapshot["backend_last_cmd"]
        self.last_action[index].copy_(
            torch.as_tensor(
                snapshot["backend_last_action"],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._previous_high_level_action[index] = snapshot[
            "previous_high_level_action"
        ]
        self._reward_joint_acc[index] = snapshot["reward_joint_acc"]
        self._entry_filtered_cmd[index] = snapshot["filtered_cmd"]
        self._entry_filter_alpha[index] = snapshot["filter_alpha"]
        self._entry_policy_scale[index] = snapshot["policy_scale"]
        self._entry_policy_bias[index] = snapshot["policy_bias"]
        rotation = quat_wxyz_to_rotmat(data.qpos[3:7])
        body_velocity = rotation.T @ data.qvel[:3]
        body_angular_velocity = rotation.T @ data.qvel[3:6]
        self.reset_velocity_body[index] = body_velocity
        self.reset_yaw_rate[index] = body_angular_velocity[2]
        self.entry_state_applied[index] = True
        self.entry_state_samples += 1
        self._pending_entry_snapshots[index] = None
        self._previous_goal_distance_xy[index] = self._distance_to_goal(index, flat=True)

    def reset_action_processing_state(self) -> dict[str, np.ndarray]:
        """Return VecEnv-owned state associated with the latest reset."""

        return {
            "valid": self.entry_state_applied.copy(),
            "filtered_cmd": self._entry_filtered_cmd.copy(),
            "filter_alpha": self._entry_filter_alpha.copy(),
            "policy_scale": self._entry_policy_scale.copy(),
            "policy_bias": self._entry_policy_bias.copy(),
        }

    def _initialize_reset_velocity(self, index: int) -> None:
        """Apply an optional moving start after standing/contact settling."""

        body_velocity = np.zeros(3, dtype=np.float64)
        yaw_rate = 0.0
        if (
            self.reset_velocity_probability > 0.0
            and float(self.rng.random()) < self.reset_velocity_probability
        ):
            body_velocity[0] = self.rng.uniform(
                self.reset_forward_speed_min, self.reset_forward_speed_max
            )
            body_velocity[1] = self.rng.uniform(
                -self.reset_lateral_speed_max, self.reset_lateral_speed_max
            )
            yaw_rate = float(
                self.rng.uniform(-self.reset_yaw_rate_max, self.reset_yaw_rate_max)
            )
            rotation = quat_wxyz_to_rotmat(self.data[index].qpos[3:7])
            self.data[index].qvel[:3] = rotation @ body_velocity
            self.data[index].qvel[3:6] = rotation @ np.asarray(
                (0.0, 0.0, yaw_rate), dtype=np.float64
            )
            mujoco.mj_forward(self.model, self.data[index])
        self.reset_velocity_body[index] = body_velocity
        self.reset_yaw_rate[index] = yaw_rate

    def _terrain_z(
        self,
        data: mujoco.MjData,
        position_xy: np.ndarray,
        *,
        ray_start_z: float = 20.0,
        fallback_height: float | None = None,
    ) -> float:
        """Estimate terrain height using an unclipped terrain-only ray."""

        saved_qpos = data.qpos.copy()
        data.qpos[:] = self.model.qpos0
        data.qpos[ROOT_QPOS + DOF:] = self.static_qpos
        data.qpos[:3] = (position_xy[0], position_xy[1], ray_start_z)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        point = np.asarray((position_xy[0], position_xy[1], ray_start_z), dtype=np.float64)
        direction = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        geomgroup = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        geomid = np.empty((1,), dtype=np.int32)
        # mj_ray queries the current model forward state for dynamic/static
        # geom transforms. Reset qpos has just been cleared, so refresh it
        # before querying terrain.
        mujoco.mj_forward(self.model, data)
        distance = mujoco.mj_ray(
            self.model,
            data,
            point,
            direction,
            geomgroup,
            1,
            self.base_body_id,
            geomid,
            None,
        )
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, data)
        if geomid[0] >= 0:
            hit_body = int(self.model.geom_bodyid[geomid[0]])
            if hit_body not in self.terrain_contact_body_ids:
                raise RuntimeError(
                    f"terrain reset ray hit non-terrain body {hit_body}"
                )
        if not np.isfinite(distance) or distance < 0.0:
            if fallback_height is not None:
                return float(fallback_height)
            # The imported terrain mesh is tessellated and has occasional
            # narrow gaps between triangles. Route reset points are sampled
            # on the route itself, so use the nearest route segment as a
            # continuous fallback instead of dropping the robot to z=0.
            best_distance = np.inf
            best_height = 0.0
            for start, end in zip(self.waypoints[:-1], self.waypoints[1:]):
                segment = end[:2] - start[:2]
                length_sq = float(np.dot(segment, segment))
                fraction = 0.0 if length_sq <= 1.0e-12 else float(
                    np.clip(np.dot(position_xy - start[:2], segment) / length_sq, 0.0, 1.0)
                )
                nearest = start[:2] + fraction * segment
                candidate_distance = float(np.linalg.norm(position_xy - nearest))
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_height = float(start[2] + fraction * (end[2] - start[2]))
            return best_height if best_distance <= 1.5 else 0.0
        terrain_z = float(ray_start_z - distance)
        return terrain_z if -1.0 <= terrain_z <= 10.0 else 0.0

    def reset(self) -> S10RawState:
        self._reset_indices(np.arange(self.num_envs, dtype=np.int64))
        self._onnx_inference_calls = 0
        return self._observe()

    def set_initial_episode_lengths(self, lengths: np.ndarray) -> S10RawState:
        """Synchronize runner phase randomization with backend-owned clocks."""

        values = np.asarray(lengths, dtype=np.int64)
        if values.shape != (self.num_envs,):
            raise ValueError(
                f"initial episode lengths must have shape {(self.num_envs,)}, got {values.shape}"
            )
        if np.any(values < 0):
            raise ValueError("initial episode lengths must be non-negative")
        maximum = np.minimum(
            self.episode_step_limits - 1,
            np.full(self.num_envs, self.single_episode_length - 1, dtype=np.int64),
        )
        values = np.minimum(values, maximum)
        self.episode_steps[:] = values
        self.target_steps[:] = values
        return self._observe()

    def set_environment_terrain_tile(
        self,
        env_index: int,
        *,
        row: int,
        col: int,
    ) -> S10RawState:
        """Select one generated tile and reset an environment for inspection."""

        if self.task_mode != "random_goal_sru" or self.terrain_atlas is None:
            raise RuntimeError("terrain tile selection requires random_goal_sru mode")
        if not 0 <= env_index < self.num_envs:
            raise ValueError(f"env_index must be in [0, {self.num_envs - 1}]")
        if not 0 <= row < self.terrain_atlas.config.num_rows:
            raise ValueError(f"row must be in [0, {self.terrain_atlas.config.num_rows - 1}]")
        if not 0 <= col < self.terrain_atlas.config.num_cols:
            raise ValueError(f"col must be in [0, {self.terrain_atlas.config.num_cols - 1}]")
        if self.learned_controller is not None:
            self.learned_controller.reset(
                self.num_envs, np.asarray((env_index,), dtype=np.int64)
            )
        self.terrain_levels[env_index] = row
        self.terrain_types[env_index] = col
        self.environment_tile_indices[env_index] = (
            row * self.terrain_atlas.config.num_cols + col
        )
        self._reset_one(env_index)
        self._onnx_inference_calls = 0
        return self._observe()

    def set_environment_terrain_tiles(self, tile_indices: np.ndarray) -> None:
        """Assign and reset a batch of random-goal environments to atlas tiles."""

        if self.task_mode != "random_goal_sru" or self.terrain_atlas is None:
            raise RuntimeError("terrain tile assignment requires random_goal_sru mode")
        indices = np.asarray(tile_indices, dtype=np.int64)
        if indices.shape != (self.num_envs,):
            raise ValueError(
                f"tile_indices must have shape {(self.num_envs,)}, got {indices.shape}"
            )
        total_tiles = self.terrain_atlas.config.num_rows * self.terrain_atlas.config.num_cols
        if (indices < 0).any() or (indices >= total_tiles).any():
            raise ValueError(f"tile_indices must be in [0, {total_tiles})")
        self.environment_tile_indices[:] = indices
        self.terrain_levels[:] = indices // self.terrain_atlas.config.num_cols
        self.terrain_types[:] = indices % self.terrain_atlas.config.num_cols
        self._reset_indices(np.arange(self.num_envs, dtype=np.int64))
        self._onnx_inference_calls = 0

    def reset_route_segment(
        self,
        start_index: int,
        *,
        env_index: int = 0,
        goal_index: int | None = None,
        fraction: float = 0.5,
        yaw: float | None = None,
        context_final_index: int | None = None,
    ) -> S10RawState:
        """Reset one environment at a route segment midpoint for diagnostics.

        This deliberately bypasses the training reset sampler. It makes a
        route-wise termination experiment reproducible without changing the
        random reset distribution used by training.
        """

        if not 0 <= env_index < self.num_envs:
            raise ValueError(f"env_index must be in [0, {self.num_envs - 1}]")
        if self.learned_controller is not None:
            self.learned_controller.reset(
                self.num_envs, np.asarray((env_index,), dtype=np.int64)
            )
        self._reset_one(
            env_index,
            route_start=start_index,
            route_goal=goal_index,
            route_fraction=fraction,
            route_yaw=yaw,
            route_context_final=context_final_index,
        )
        return self._observe()

    def reset_route_sequence(
        self,
        start_index: int,
        final_index: int,
        *,
        env_index: int = 0,
        fraction: float = 0.1,
        yaw: float | None = None,
    ) -> S10RawState:
        """Reset one environment for continuous multi-waypoint evaluation."""

        if not 0 <= env_index < self.num_envs:
            raise ValueError(f"env_index must be in [0, {self.num_envs - 1}]")
        if self.learned_controller is not None:
            self.learned_controller.reset(
                self.num_envs, np.asarray((env_index,), dtype=np.int64)
            )
        self._reset_one(
            env_index,
            route_start=start_index,
            route_goal=final_index,
            route_fraction=fraction,
            route_yaw=yaw,
            route_sequential=True,
        )
        return self._observe()

    def reset_route_sequences(
        self,
        start_indices: np.ndarray,
        final_indices: np.ndarray,
        *,
        fractions: np.ndarray | None = None,
        yaws: np.ndarray | None = None,
    ) -> S10RawState:
        """Reset every environment for continuous multi-waypoint evaluation."""

        starts = np.asarray(start_indices, dtype=np.int64)
        finals = np.asarray(final_indices, dtype=np.int64)
        if starts.shape != (self.num_envs,) or finals.shape != (self.num_envs,):
            raise ValueError("start_indices and final_indices must have shape (num_envs,)")
        route_fractions = (
            np.zeros(self.num_envs, dtype=np.float64)
            if fractions is None
            else np.asarray(fractions, dtype=np.float64)
        )
        route_yaws = None if yaws is None else np.asarray(yaws, dtype=np.float64)
        if route_fractions.shape != (self.num_envs,):
            raise ValueError("fractions must have shape (num_envs,)")
        if route_yaws is not None and route_yaws.shape != (self.num_envs,):
            raise ValueError("yaws must have shape (num_envs,)")
        indices = np.arange(self.num_envs, dtype=np.int64)
        if self.learned_controller is not None:
            self.learned_controller.reset(self.num_envs, indices)
        for index in indices:
            self._reset_one(
                int(index),
                settle=False,
                route_start=int(starts[index]),
                route_goal=int(finals[index]),
                route_fraction=float(route_fractions[index]),
                route_yaw=(
                    None if route_yaws is None else float(route_yaws[index])
                ),
                route_sequential=True,
            )
        if self.reset_settle_physics_steps:
            chunks = tuple(
                chunk
                for chunk in np.array_split(indices, min(self.physics_workers, self.num_envs))
                if len(chunk)
            )
            if self._physics_executor is None:
                self._settle_reset_chunk(chunks[0])
            else:
                futures = [
                    self._physics_executor.submit(self._settle_reset_chunk, chunk)
                    for chunk in chunks
                ]
                for future in futures:
                    future.result()
        for index in indices:
            self._finalize_reset_state(int(index))
        self._onnx_inference_calls = 0
        return self._observe()

    def reset_route_segments(
        self,
        start_indices: np.ndarray,
        fractions: np.ndarray,
        *,
        goal_indices: np.ndarray | None = None,
        yaws: np.ndarray | None = None,
    ) -> S10RawState:
        """Reset every environment to explicit route candidates for diagnostics."""

        starts = np.asarray(start_indices, dtype=np.int64)
        fractions = np.asarray(fractions, dtype=np.float64)
        if starts.shape != (self.num_envs,) or fractions.shape != (self.num_envs,):
            raise ValueError("start_indices and fractions must have shape (num_envs,)")
        if goal_indices is None:
            goals = np.asarray(
                [
                    next_route_waypoint(int(start), len(self.waypoints) - 1)
                    for start in starts
                ],
                dtype=np.int64,
            )
        else:
            goals = np.asarray(goal_indices, dtype=np.int64)
            if goals.shape != (self.num_envs,):
                raise ValueError("goal_indices must have shape (num_envs,)")
        if yaws is None:
            route_yaws: np.ndarray | None = None
        else:
            route_yaws = np.asarray(yaws, dtype=np.float64)
            if route_yaws.shape != (self.num_envs,):
                raise ValueError("yaws must have shape (num_envs,)")
        for index in range(self.num_envs):
            self._reset_one(
                index,
                settle=False,
                route_start=int(starts[index]),
                route_goal=int(goals[index]),
                route_fraction=float(fractions[index]),
                route_yaw=(None if route_yaws is None else float(route_yaws[index])),
            )
        if self.reset_settle_physics_steps:
            chunks = tuple(
                chunk
                for chunk in np.array_split(
                    np.arange(self.num_envs), min(self.physics_workers, self.num_envs)
                )
                if len(chunk)
            )
            if self._physics_executor is None:
                self._settle_reset_chunk(chunks[0])
            else:
                futures = [
                    self._physics_executor.submit(self._settle_reset_chunk, chunk)
                    for chunk in chunks
                ]
                for future in futures:
                    future.result()
        if self.learned_controller is not None:
            self.learned_controller.reset(self.num_envs)
        self._previous_goal_distance_xy[:] = np.asarray(
            [self._distance_to_goal(index, flat=True) for index in range(self.num_envs)],
            dtype=np.float64,
        )
        self._onnx_inference_calls = 0
        return self._observe()

    def _goal_point(self, index: int) -> np.ndarray:
        if self.task_mode == "random_goal_sru":
            return self.random_goal_positions[index]
        return self.waypoints[min(self.goal_waypoint_indices[index], len(self.waypoints) - 1)]

    def _distance_to_goal(self, index: int, *, flat: bool = False) -> float:
        pos = self.data[index].qpos[:3]
        delta = self._goal_point(index) - pos
        return float(np.linalg.norm(delta[:2] if flat else delta))

    def _goal_height_error(self, index: int) -> float:
        """Return terrain-level height error rather than base-to-ground error."""

        if self.task_mode == "random_goal_sru":
            # The original SRU at-goal termination is XY-only. Goal z remains
            # part of the soft XYZ reward and actor observation.
            return 0.0
        base_terrain_z = float(self.data[index].qpos[2]) - STANDING_BASE_CLEARANCE
        return abs(base_terrain_z - float(self._goal_point(index)[2]))

    def _route_distance_to_goal(self, index: int) -> float:
        distance_xy = self._distance_to_goal(index, flat=True)
        height_error = self._goal_height_error(index)
        return float(np.hypot(distance_xy, height_error))

    def _illegal_contact_force(
        self, data: mujoco.MjData, env_index: int | None = None
    ) -> float:
        """Return the largest selected-body contact force in this step.

        IsaacLab's contact sensor observes base and hip bodies and takes the
        maximum norm over its history. MuJoCo exposes per-contact forces, so
        the high-level step takes the maximum over all physics substeps.
        """

        max_force = 0.0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            diagnostic_force_norm: float | None = None
            if self.record_raw_contact_diagnostics and env_index is not None:
                diagnostic_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(
                    self.model, data, contact_index, diagnostic_force
                )
                diagnostic_force_norm = float(
                    np.linalg.norm(diagnostic_force[:3])
                )
                body_a_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_a
                ) or f"body_{body_a}"
                body_b_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_b
                ) or f"body_{body_b}"
                raw_pair_name = "<->".join(sorted((body_a_name, body_b_name)))
                raw_pair_peaks = self._raw_contact_force_peaks_by_pair[env_index]
                raw_pair_peaks[raw_pair_name] = max(
                    raw_pair_peaks.get(raw_pair_name, 0.0),
                    diagnostic_force_norm,
                )
            selected = body_a in self.contact_body_ids or body_b in self.contact_body_ids
            if not selected:
                continue
            robot_self_contact = body_a in self.robot_body_ids and body_b in self.robot_body_ids
            if not self.include_self_contacts and robot_self_contact:
                continue
            if self.ignore_world_body_contacts and (body_a == 0 or body_b == 0):
                # The S10 MJCF standing pose has a persistent base/floor
                # collision, unlike the IsaacLab asset. Keep this known
                # representation artifact out of illegal-contact termination;
                # contacts against the imported terrain body remain active.
                continue
            if diagnostic_force_norm is None:
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, data, contact_index, force)
                force_norm = float(np.linalg.norm(force[:3]))
            else:
                force_norm = diagnostic_force_norm
            max_force = max(max_force, force_norm)
            if self.record_contact_diagnostics and env_index is not None:
                body_peaks = self._contact_force_peaks_by_body[env_index]
                pair_peaks = self._contact_force_peaks_by_pair[env_index]
                selected_bodies = (
                    {body_a, body_b} & self.contact_body_ids
                )
                for body_id in selected_bodies:
                    body_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    ) or f"body_{body_id}"
                    body_peaks[body_name] = max(
                        body_peaks.get(body_name, 0.0), force_norm
                    )
                body_a_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_a
                ) or f"body_{body_a}"
                body_b_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_b
                ) or f"body_{body_b}"
                pair_name = "<->".join(sorted((body_a_name, body_b_name)))
                pair_peaks[pair_name] = max(
                    pair_peaks.get(pair_name, 0.0), force_norm
                )
        return max_force

    def _apply_low_level(self, data: mujoco.MjData, cmd: np.ndarray) -> None:
        if self.low_level == "none":
            data.ctrl[:] = 0.0
            return
        q = data.qpos[7:7 + DOF]
        dq = data.qvel[6:6 + DOF]
        target = self.joint_target.copy()
        # Match the official runner's split action: leg joints are position
        # targets (kp=80,kd=2), wheel joints are velocity targets (kp=0,kd=.6).
        # This is a stable backend baseline, not the learned ONNX policy.
        wheel_speed = 8.0 * float(cmd[0])
        yaw_speed = 3.0 * float(cmd[2])
        wheel_target = wheel_speed + np.asarray((-yaw_speed, yaw_speed, -yaw_speed, yaw_speed))
        torque = self.kp * (target - q) - self.kd * dq
        wheel_indices = np.arange(3, DOF, 4)
        torque[wheel_indices] = self.kd[wheel_indices] * (wheel_target - dq[wheel_indices])
        data.ctrl[:] = np.clip(np.nan_to_num(torque), self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def _apply_learned_low_level(
        self, data: mujoco.MjData, target: np.ndarray, env_index: int
    ) -> None:
        q = data.qpos[7:7 + DOF]
        dq = data.qvel[6:6 + DOF]
        leg_indices = np.asarray((0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14), dtype=np.int64)
        wheel_indices = np.asarray((3, 7, 11, 15), dtype=np.int64)
        # The C++ policy and state machine use raw robot joint coordinates;
        # the two ROS wire conversions cancel before MuJoCo sees the command.
        raw_leg_target = target[leg_indices]
        raw_wheel_velocity = target[wheel_indices]
        torque = np.zeros(DOF, dtype=np.float64)
        kp = self.kp * self.learned_controller.kp_scale[env_index]
        kd = self.kd * self.learned_controller.kd_scale[env_index]
        torque[leg_indices] = kp[leg_indices] * (raw_leg_target - q[leg_indices]) - kd[leg_indices] * dq[leg_indices]
        torque[wheel_indices] = kd[wheel_indices] * (raw_wheel_velocity - dq[wheel_indices])
        data.ctrl[:] = np.clip(np.nan_to_num(torque), self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def _advance_chunk(
        self,
        indices: np.ndarray,
        commands: np.ndarray,
        held_targets: np.ndarray,
        physics_steps: int,
    ) -> None:
        for index_value in indices:
            index = int(index_value)
            data = self.data[index]
            # IsaacLab's joint_acc is a velocity finite difference at its 5 ms
            # physics rate, not the simulator's instantaneous acceleration.
            acceleration_steps = max(1, int(round(0.005 / self.dt)))
            acceleration_steps = min(acceleration_steps, physics_steps)
            start_velocity = data.qvel[ROOT_QVEL:ROOT_QVEL + DOF].copy()
            for physics_index in range(physics_steps):
                if physics_index == physics_steps - acceleration_steps:
                    start_velocity = data.qvel[ROOT_QVEL:ROOT_QVEL + DOF].copy()
                if self.low_level in {"official_onnx", "pim_him"}:
                    self._apply_learned_low_level(data, held_targets[index], index)
                else:
                    self._apply_low_level(data, commands[index])
                mujoco.mj_step(self.model, data)
                contact_force = self._illegal_contact_force(data, index)
                self._last_illegal_contact_force[index] = contact_force
                self._max_illegal_contact_force[index] = max(
                    self._max_illegal_contact_force[index], contact_force
                )
                if self.record_imu_history:
                    self._imu_capture_substeps[index] += 1
                    if self._imu_capture_substeps[index] % self.imu_decimation == 0:
                        self._record_imu_sample(index, contact_force)
            self._reward_joint_acc[index] = (
                data.qvel[ROOT_QVEL:ROOT_QVEL + DOF] - start_velocity
            ) / (acceleration_steps * self.dt)

    def _advance_segment(
        self, commands: np.ndarray, held_targets: np.ndarray, physics_steps: int
    ) -> None:
        if self._physics_executor is None:
            self._advance_chunk(self._physics_chunks[0], commands, held_targets, physics_steps)
            return
        futures = [
            self._physics_executor.submit(
                self._advance_chunk, chunk, commands, held_targets, physics_steps
            )
            for chunk in self._physics_chunks
        ]
        for future in futures:
            future.result()

    def _advance_all(self, commands: np.ndarray) -> None:
        self._max_illegal_contact_force.fill(0.0)
        if self.record_contact_diagnostics:
            for body_peaks, pair_peaks, raw_pair_peaks in zip(
                self._contact_force_peaks_by_body,
                self._contact_force_peaks_by_pair,
                self._raw_contact_force_peaks_by_pair,
            ):
                body_peaks.clear()
                pair_peaks.clear()
                raw_pair_peaks.clear()
        if self.record_imu_history:
            self._imu_capture_substeps.fill(0)
            for index, history in enumerate(self._imu_histories):
                history.clear()
                self._record_imu_sample(index, self._last_illegal_contact_force[index])
        held_targets = np.tile(self.joint_target[None, :], (self.num_envs, 1))
        if self.low_level in {"official_onnx", "pim_him"}:
            for start in range(0, self.physics_steps, self.low_level_decimation):
                held_targets, _ = self.learned_controller.infer(self.data, commands)
                self._onnx_inference_calls += self.num_envs
                segment_steps = min(
                    self.low_level_decimation, self.physics_steps - start
                )
                self._advance_segment(commands, held_targets, segment_steps)
        else:
            self._advance_segment(commands, held_targets, self.physics_steps)

    def _record_imu_sample(self, index: int, contact_force: float) -> None:
        data = self.data[index]
        self._imu_histories[index].append(
            (
                float(data.time),
                np.asarray(data.sensordata[:4], dtype=np.float64).copy(),
                np.asarray(data.sensordata[4:7], dtype=np.float64).copy(),
                np.asarray(data.sensordata[7:10], dtype=np.float64).copy(),
                np.asarray(data.qvel[ROOT_QVEL + np.asarray((3, 7, 11, 15))], dtype=np.float64).copy(),
                np.asarray(data.ctrl[np.asarray((3, 7, 11, 15))], dtype=np.float64).copy(),
                float(contact_force),
            )
        )

    def imu_history(self, env_index: int = 0) -> dict[str, np.ndarray]:
        """Return the most recent high-level interval's onboard sensor samples."""

        if not self.record_imu_history:
            raise RuntimeError("IMU history recording is disabled for this backend")
        if not 0 <= env_index < self.num_envs:
            raise IndexError(env_index)
        history = self._imu_histories[env_index]
        if not history:
            return {
                "time": np.empty(0, dtype=np.float64),
                "orientation_wxyz": np.empty((0, 4), dtype=np.float64),
                "accelerometer": np.empty((0, 3), dtype=np.float64),
                "gyro": np.empty((0, 3), dtype=np.float64),
                "wheel_qvel": np.empty((0, 4), dtype=np.float64),
                "wheel_torque": np.empty((0, 4), dtype=np.float64),
                "illegal_contact_force": np.empty(0, dtype=np.float64),
            }
        return {
            "time": np.asarray([sample[0] for sample in history], dtype=np.float64),
            "orientation_wxyz": np.stack([sample[1] for sample in history]),
            "accelerometer": np.stack([sample[2] for sample in history]),
            "gyro": np.stack([sample[3] for sample in history]),
            "wheel_qvel": np.stack([sample[4] for sample in history]),
            "wheel_torque": np.stack([sample[5] for sample in history]),
            "illegal_contact_force": np.asarray([sample[6] for sample in history], dtype=np.float64),
        }

    def _body_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lin = np.zeros((self.num_envs, 3), dtype=np.float32)
        ang = np.zeros((self.num_envs, 3), dtype=np.float32)
        gravity = np.zeros((self.num_envs, 3), dtype=np.float32)
        pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        for i, data in enumerate(self.data):
            rot = quat_wxyz_to_rotmat(data.qpos[3:7])
            lin[i] = (rot.T @ data.qvel[:3]).astype(np.float32)
            ang[i] = data.sensordata[7:10].astype(np.float32)
            gravity[i] = (rot.T @ np.asarray((0.0, 0.0, -1.0))).astype(np.float32)
            pose[i] = data.qpos[:7].astype(np.float32)
        return lin, ang, gravity, pose

    def _tilt_degrees(self) -> np.ndarray:
        tilt = np.zeros(self.num_envs, dtype=np.float32)
        for i, data in enumerate(self.data):
            rot = quat_wxyz_to_rotmat(data.qpos[3:7])
            gravity = rot.T @ np.asarray((0.0, 0.0, -1.0))
            roll = np.arctan2(gravity[1], -gravity[2])
            pitch = np.arctan2(gravity[0], -gravity[2])
            tilt[i] = np.rad2deg(max(abs(roll), abs(pitch)))
        return tilt

    def _goal_body(self, poses: np.ndarray) -> np.ndarray:
        goals = np.zeros((self.num_envs, 4), dtype=np.float32)
        for i in range(self.num_envs):
            rot = quat_wxyz_to_rotmat(poses[i, 3:7])
            target = self._goal_point(i)
            delta = rot.T @ (target - poses[i, :3])
            distance = max(float(np.linalg.norm(delta)), 1.0e-6)
            goals[i, :3] = (delta / distance).astype(np.float32)
            goals[i, 3] = np.log1p(distance)
        return goals

    def _capture_sensor_one(
        self, index: int
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, np.ndarray | None]:
        data = self.data[index]
        lidar = (
            self.lidar_samplers[index].capture(data)
            if self.use_lidar and self.sensor_backend == "cpu" else None
        )
        height = self.height_scanners[index].raw_scan(data) if self.use_height else None
        return lidar, height

    def _observe(self) -> S10RawState:
        lin, ang, gravity, poses = self._body_state()
        if self._sensor_executor is None:
            sensor_captures = [self._capture_sensor_one(i) for i in range(self.num_envs)]
        else:
            sensor_captures = list(
                self._sensor_executor.map(self._capture_sensor_one, range(self.num_envs))
            )
        if self.use_lidar:
            if self.sensor_backend == "warp":
                root_qpos = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
                front, rear, front_z_native, rear_z_native = (
                    self.warp_lidar.capture_with_world_z(root_qpos)
                )
            else:
                scans = [capture[0] for capture in sensor_captures]
                front = np.stack([item[0] for item in scans])
                rear = np.stack([item[1] for item in scans])
            if self.record_lidar_for_visualization:
                if isinstance(front, torch.Tensor):
                    self._visual_lidar_scans = (
                        front[0].detach().cpu().numpy().copy(),
                        rear[0].detach().cpu().numpy().copy(),
                    )
                else:
                    self._visual_lidar_scans = (
                        np.asarray(front[0]).copy(),
                        np.asarray(rear[0]).copy(),
                    )
            front_d, _ = native_to_90(front)
            rear_d, _ = native_to_90(rear)
            if self.sensor_backend == "warp":
                front_z = gather_aux_at_min_distance(front_z_native, front)
                rear_z = gather_aux_at_min_distance(rear_z_native, rear)
            else:
                front_z = np.stack([
                    gather_aux_at_min_distance(world_z_native(scan[0], pose, front=True, sensor_dirs=self.sensor_dirs), scan[0]).squeeze(0).numpy()
                    for scan, pose in zip(scans, poses)
                ])
                rear_z = np.stack([
                    gather_aux_at_min_distance(world_z_native(scan[1], pose, front=False, sensor_dirs=self.sensor_dirs), scan[1]).squeeze(0).numpy()
                    for scan, pose in zip(scans, poses)
                ])
            lidar_latent = self.lidar_encoder.encode_maps(front_d, rear_d, front_z, rear_z)
        else:
            lidar_latent = torch.zeros((self.num_envs, 64, 5, 8), device=self.device)

        height_latent = None
        if self.use_height:
            raw_height = np.stack([capture[1] for capture in sensor_captures])
            height_latent = self.height_encoder.encode(raw_height)

        return S10RawState(
            base_lin_vel=torch.as_tensor(lin, device=self.device),
            base_ang_vel=torch.as_tensor(ang, device=self.device),
            projected_gravity=torch.as_tensor(gravity, device=self.device),
            last_action=self.last_action.clone(),
            goal_body=torch.as_tensor(self._goal_body(poses), device=self.device),
            lidar_latent=lidar_latent,
            height_latent=height_latent,
            time_normalized=torch.as_tensor(
                np.minimum(
                    self.target_steps / max(self.single_episode_length, 1),
                    1.0,
                )[:, None],
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def lidar_hit_points(
        self,
        env_index: int = 0,
        *,
        vertical_stride: int = 12,
        horizontal_stride: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a sparse world-frame point cloud for GUI visualization."""

        if self._visual_lidar_scans is None:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty
        if not 0 <= env_index < self.num_envs:
            raise IndexError(env_index)
        root_qpos = np.asarray(self.data[env_index].qpos[:7], dtype=np.float64)
        root_rot = quat_wxyz_to_rotmat(root_qpos[3:7])
        directions = self.sensor_dirs[::vertical_stride, ::horizontal_stride]
        points: list[np.ndarray] = []
        for scan, sensor_pos, sensor_quat in (
            (self._visual_lidar_scans[0], S10_FRONT_POS, S10_FRONT_ROT_WXYZ),
            (self._visual_lidar_scans[1], S10_REAR_POS, S10_REAR_ROT_WXYZ),
        ):
            sampled = np.asarray(scan)[::vertical_stride, ::horizontal_stride]
            valid = (sampled > MIN_RANGE_M) & (sampled < INVALID_RANGE_THRESHOLD_M)
            sensor_origin = root_qpos[:3] + root_rot @ sensor_pos
            world_dirs = np.einsum(
                "ij,hwj->hwi",
                root_rot @ quat_wxyz_to_rotmat(sensor_quat),
                directions,
            )
            hit_points = sensor_origin + sampled[..., None] * world_dirs
            points.append(np.asarray(hit_points[valid], dtype=np.float64))
        return points[0], points[1]

    def _reward_done(
        self, commands: np.ndarray, policy_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate SRU rewards and advance continuous evaluation routes."""
        if policy_actions is None:
            policy_actions = np.column_stack((commands[:, 0] / 1.5, commands[:, 2]))
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        components = {
            "joint_acc_l2": np.zeros(self.num_envs, dtype=np.float32),
            "lateral_movement": np.zeros(self.num_envs, dtype=np.float32),
            "rot_movement": np.zeros(self.num_envs, dtype=np.float32),
            "action_rate_l1": np.zeros(self.num_envs, dtype=np.float32),
            "goal_progress": np.zeros(self.num_envs, dtype=np.float32),
            "episode_termination": np.zeros(self.num_envs, dtype=np.float32),
            "reach_goal_xy_soft": np.zeros(self.num_envs, dtype=np.float32),
            "reach_goal_xy_tight": np.zeros(self.num_envs, dtype=np.float32),
        }
        termination_components = {
            DONE_NONE: np.zeros(self.num_envs, dtype=np.float32),
            DONE_NONFINITE: np.zeros(self.num_envs, dtype=np.float32),
            DONE_BASE_CONTACT: np.zeros(self.num_envs, dtype=np.float32),
            DONE_LARGE_ANGLE: np.zeros(self.num_envs, dtype=np.float32),
            DONE_TERRAIN_FALL: np.zeros(self.num_envs, dtype=np.float32),
            DONE_COMPLETE: np.zeros(self.num_envs, dtype=np.float32),
            DONE_TIMEOUT: np.zeros(self.num_envs, dtype=np.float32),
        }
        reward_dt = self.physics_steps * self.dt
        cfg = self.reward_config
        self.waypoints_reached_this_step.fill(0)
        self.reached_waypoint_indices_this_step.fill(-1)
        for i, data in enumerate(self.data):
            rot = quat_wxyz_to_rotmat(data.qpos[3:7])
            gravity = rot.T @ np.asarray((0.0, 0.0, -1.0))
            lin_vel_body = rot.T @ data.qvel[:3]
            ang_vel_body = rot.T @ data.qvel[3:6]
            current_action = policy_actions[i]
            action_rate = float(np.abs(current_action - self._previous_high_level_action[i]).sum())
            self._previous_high_level_action[i] = current_action

            distance_xy = self._distance_to_goal(i, flat=True)
            distance_xyz = self._distance_to_goal(i)
            height_error = self._goal_height_error(i)
            self._target_distance_xy_this_step[i] = distance_xy
            self._target_height_error_this_step[i] = height_error
            components["goal_progress"][i] = compute_goal_progress_reward(
                self._previous_goal_distance_xy[i],
                distance_xy,
                cfg.goal_progress,
            )
            self._previous_goal_distance_xy[i] = distance_xy
            is_sequential_route = self.waypoint_task_types[i] != TASK_SINGLE
            is_final_route_target = (
                is_sequential_route
                and self.goal_waypoint_indices[i] == self.final_waypoint_indices[i]
            )
            entered_goal = (
                distance_xy < cfg.goal_distance_threshold
                and height_error < cfg.goal_height_threshold
            )
            if is_sequential_route:
                # Continuous evaluation targets switch on first entry so pose,
                # velocity and recurrent policy state are preserved.
                arrived = bool(entered_goal)
                goal_complete = bool(entered_goal)
            else:
                if not self._goal_was_reached[i] and entered_goal:
                    self._goal_was_reached[i] = True
                if self._goal_was_reached[i]:
                    self._goal_hold_steps[i] += 1
                arrived = bool(self._goal_was_reached[i])
                goal_complete = (
                    self._goal_hold_steps[i] > self.required_goal_hold_steps
                )
            target_step_limit = self.single_episode_length
            timeout = (
                self.target_steps[i] >= target_step_limit
                or self.episode_steps[i] >= self.episode_step_limits[i]
            )
            roll = float(np.arctan2(gravity[1], -gravity[2]))
            pitch = float(np.arctan2(gravity[0], -gravity[2]))
            large_angle = abs(roll) > np.deg2rad(40.0) or abs(pitch) > np.deg2rad(40.0)
            if self._max_illegal_contact_force[i] > self.contact_threshold:
                self._illegal_contact_steps[i] += 1
            else:
                self._illegal_contact_steps[i] = 0
            base_contact = self._illegal_contact_steps[i] >= self.contact_persistence_steps
            terrain_fall = float(data.qpos[2]) < self.terrain_fall_height
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                self.last_done_reason[i] = DONE_NONFINITE
            elif base_contact:
                self.last_done_reason[i] = DONE_BASE_CONTACT
            elif large_angle:
                self.last_done_reason[i] = DONE_LARGE_ANGLE
            elif terrain_fall:
                self.last_done_reason[i] = DONE_TERRAIN_FALL
            elif timeout:
                self.last_done_reason[i] = DONE_TIMEOUT
            elif goal_complete:
                self.waypoints_reached_this_step[i] = 1
                self.reached_waypoint_indices_this_step[i] = self.goal_waypoint_indices[i]
                if is_sequential_route and not is_final_route_target:
                    self.goal_waypoint_indices[i] = next_route_waypoint(
                        int(self.goal_waypoint_indices[i]),
                        int(self.final_waypoint_indices[i]),
                    )
                    self.target_steps[i] = 0
                    self._goal_was_reached[i] = False
                    self._goal_hold_steps[i] = 0
                    self._previous_goal_distance_xy[i] = self._distance_to_goal(i, flat=True)
                    self.last_done_reason[i] = DONE_NONE
                else:
                    self.last_done_reason[i] = DONE_COMPLETE
            else:
                self.last_done_reason[i] = DONE_NONE
            dones[i] = self.last_done_reason[i] != DONE_NONE

            components["joint_acc_l2"][i] = cfg.joint_acc_l2 * float(np.square(self._reward_joint_acc[i]).sum()) * reward_dt
            components["lateral_movement"][i] = cfg.lateral_movement * abs(float(lin_vel_body[1])) * reward_dt
            components["rot_movement"][i] = cfg.rot_movement * float(np.linalg.norm(ang_vel_body)) * reward_dt
            components["action_rate_l1"][i] = cfg.action_rate_l1 * action_rate * reward_dt
            if self.last_done_reason[i] in {
                DONE_BASE_CONTACT,
                DONE_LARGE_ANGLE,
                DONE_NONFINITE,
            }:
                components["episode_termination"][i] = cfg.episode_termination * reward_dt

            # Match the original SRU reward mask: both goal kernels remain
            # enabled after first entry and throughout the final four seconds
            # of the active target budget. The two pre-window random samples
            # are independent in the source task.
            timeup_mask = self.target_steps[i] > (
                target_step_limit - self.required_goal_hold_steps
            )
            random_soft_mask = bool(
                self.rng.random() < cfg.random_goal_reward_probability
            )
            random_tight_mask = bool(
                self.rng.random() < cfg.random_goal_reward_probability
            )
            soft = (
                1.0
                / (1.0 + (distance_xyz / cfg.soft_sigmoid) ** 2)
                / cfg.soft_time_scale
            )
            tight = (
                1.0
                / (1.0 + (distance_xy / cfg.tight_sigmoid) ** 2)
                / cfg.tight_time_scale
            )
            if arrived or timeup_mask or random_soft_mask:
                components["reach_goal_xy_soft"][i] = (
                    cfg.reach_goal_xy_soft
                    * soft
                    * reward_dt
                )
            if arrived or timeup_mask or random_tight_mask:
                components["reach_goal_xy_tight"][i] = (
                    cfg.reach_goal_xy_tight
                    * tight
                    * reward_dt
                )
            rewards[i] = sum(value[i] for value in components.values())
            termination_components[self.last_done_reason[i]][i] = 1.0
        self.last_reward_components = components
        self.last_termination_components = termination_components
        return rewards, dones

    def set_policy_actions(self, actions: torch.Tensor) -> None:
        actions = torch.as_tensor(actions, dtype=torch.float32).detach().cpu().numpy()
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"policy actions must have shape {(self.num_envs, 2)}, got {actions.shape}")
        self._pending_high_level_action = actions.astype(np.float64, copy=True)

    def step(self, cmd_vel: torch.Tensor) -> tuple[S10RawState, torch.Tensor, torch.Tensor, dict[str, Any]]:
        commands = torch.as_tensor(cmd_vel, dtype=torch.float32).detach().cpu().numpy().reshape(self.num_envs, 3)
        if self.command_noise > 0.0:
            commands = commands + self.rng.normal(0.0, self.command_noise, commands.shape)
        commands[:, 0] = np.clip(commands[:, 0], -1.0, 1.0)
        commands[:, 1] = 0.0
        commands[:, 2] = np.clip(commands[:, 2], -1.0, 1.0)
        if self._pending_high_level_action is None:
            policy_actions = np.column_stack((commands[:, 0] / 1.5, commands[:, 2]))
        else:
            policy_actions = self._pending_high_level_action
            self._pending_high_level_action = None
        self._advance_all(commands)
        self.episode_steps += 1
        self.target_steps += 1
        rewards, dones = self._reward_done(commands, policy_actions)
        terminal_reasons = tuple(str(reason) if done else DONE_NONE for reason, done in zip(self.last_done_reason, dones))
        terminal_base_position = np.asarray(
            [data.qpos[:3].copy() for data in self.data], dtype=np.float64
        )
        terminal_base_z = np.asarray([data.qpos[2] for data in self.data], dtype=np.float64)
        terminal_tilt_deg = self._tilt_degrees().astype(np.float64)
        terminal_goal_distance = self._target_distance_xy_this_step.copy()
        terminal_goal_height_error = self._target_height_error_this_step.copy()
        terminal_episode_steps = self.episode_steps.copy()
        terminal_target_steps = self.target_steps.copy()
        terminal_start_waypoints = self.start_waypoint_indices.copy()
        terminal_goal_waypoints = self.goal_waypoint_indices.copy()
        terminal_final_waypoints = self.final_waypoint_indices.copy()
        terminal_task_types = self.waypoint_task_types.copy()
        terminal_waypoints_reached = self.waypoints_reached_this_step.copy()
        reached_waypoints = self.reached_waypoint_indices_this_step.copy()
        terminal_max_illegal_contact_force = self._max_illegal_contact_force.copy()
        terminal_last_illegal_contact_force = self._last_illegal_contact_force.copy()
        terminal_illegal_contact_steps = self._illegal_contact_steps.copy()
        terminal_contact_force_peaks_by_body = tuple(
            dict(peaks) for peaks in self._contact_force_peaks_by_body
        )
        terminal_contact_force_peaks_by_pair = tuple(
            dict(peaks) for peaks in self._contact_force_peaks_by_pair
        )
        terminal_raw_contact_force_peaks_by_pair = tuple(
            dict(peaks) for peaks in self._raw_contact_force_peaks_by_pair
        )
        terminal_timeouts = np.asarray(
            [
                done and reason == DONE_TIMEOUT
                for reason, done in zip(self.last_done_reason, dones)
            ],
            dtype=bool,
        )
        if self.segment_sampler is not None:
            single_done = dones & (terminal_task_types == TASK_SINGLE)
            self.segment_sampler.update(
                terminal_start_waypoints[single_done],
                np.asarray(
                    [
                        float(reason == DONE_COMPLETE)
                        for reason, selected in zip(self.last_done_reason, single_done)
                        if selected
                    ],
                    dtype=np.float64,
                ),
            )
        self.last_cmd[:] = commands
        self.last_action.copy_(torch.as_tensor(policy_actions, dtype=torch.float32, device=self.device))
        done_indices = np.flatnonzero(dones)
        self._reset_indices(done_indices)
        state = self._observe()
        info = {
            "backend": "native_mujoco",
            "task_mode": self.task_mode,
            "low_level": self.low_level,
            "num_envs": self.num_envs,
            "termination_config": {
                "contact_threshold": self.contact_threshold,
                "contact_persistence_steps": self.contact_persistence_steps,
                "contact_force_statistic": "max_selected_contact_over_policy_step",
                "large_angle_deg": 40.0,
                "terrain_fall_height": self.terrain_fall_height,
                "goal_distance_threshold": self.reward_config.goal_distance_threshold,
                "goal_height_threshold": self.reward_config.goal_height_threshold,
                "goal_completion": "original_sru_latched_hold_single_only",
                "sequential_route_completion": "first_entry_continuous",
                "goal_reward_distances": {
                    "soft": "base_goal_xyz",
                    "tight": "base_goal_xy",
                },
                "goal_required_steps": self.required_goal_hold_steps,
            },
            "timing_config": {
                "mujoco_dt": self.dt,
                "high_level_hz": self.action_spec.policy_hz,
                "high_level_physics_steps": self.physics_steps,
                "low_level_hz": self.action_spec.low_level_hz,
                "low_level_decimation": self.low_level_decimation,
                "sensor_workers": self.sensor_workers,
                "physics_workers": self.physics_workers,
                "lidar_horizontal_samples": self.lidar_horizontal_samples,
                "sensor_backend": self.sensor_backend,
                "lidar_visualization": self.record_lidar_for_visualization,
                "low_level_inference_calls_this_step": (
                    self.physics_steps // self.low_level_decimation * self.num_envs
                    if self.low_level in {"official_onnx", "pim_him"} else 0
                ),
                # Retained for old log consumers and checkpoint diagnostics.
                "onnx_inference_calls_this_step": (
                    self.physics_steps // self.low_level_decimation * self.num_envs
                    if self.low_level == "official_onnx" else 0
                ),
            },
            "start_waypoint_indices": terminal_start_waypoints,
            "goal_waypoint_indices": terminal_goal_waypoints,
            "final_waypoint_indices": terminal_final_waypoints,
            "waypoint_task_types": terminal_task_types,
            "waypoint_task_names": tuple(
                TASK_NAMES[int(task_type)] for task_type in terminal_task_types
            ),
            "waypoints_reached_this_step": terminal_waypoints_reached,
            "reached_waypoint_indices": reached_waypoints,
            "terrain_levels": (
                None if self.terrain_levels is None else self.terrain_levels.copy()
            ),
            "terrain_types": (
                None if self.terrain_types is None else self.terrain_types.copy()
            ),
            "terrain_tile_indices": (
                None
                if self.environment_tile_indices is None
                else self.environment_tile_indices.copy()
            ),
            "done_count": int(dones.sum()),
            "mean_forward_speed": float(np.mean([data.qvel[0] for data in self.data])),
            "mean_reward": float(rewards.mean()),
            "done_reason": terminal_reasons,
            "time_outs": terminal_timeouts,
            "terminal_base_position": terminal_base_position,
            "terminal_base_z": terminal_base_z,
            "terminal_tilt_deg": terminal_tilt_deg,
            "terminal_goal_distance_xy": terminal_goal_distance,
            "terminal_goal_height_error": terminal_goal_height_error,
            "terminal_episode_steps": terminal_episode_steps,
            "terminal_target_steps": terminal_target_steps,
            "max_illegal_contact_force": terminal_max_illegal_contact_force,
            "last_illegal_contact_force": terminal_last_illegal_contact_force,
            "illegal_contact_steps": terminal_illegal_contact_steps,
            "contact_force_peaks_by_body": terminal_contact_force_peaks_by_body,
            "contact_force_peaks_by_pair": terminal_contact_force_peaks_by_pair,
            "raw_contact_force_peaks_by_pair": terminal_raw_contact_force_peaks_by_pair,
            "termination_components": {
                name: float(value.mean()) for name, value in self.last_termination_components.items()
            },
            "segment_sampling": (
                self.segment_sampler.summary()
                if self.segment_sampler is not None else None
            ),
            "entry_state_sampling": {
                "enabled": self.entry_state_bank is not None,
                "probability": self.entry_state_probability,
                "applied_total": self.entry_state_samples,
                "missing_segment_fallbacks": self.entry_state_fallbacks,
                "available_segments": (
                    []
                    if self.entry_state_bank is None
                    else list(self.entry_state_bank.available_segments)
                ),
            },
            "reward_components": {name: float(value.mean()) for name, value in self.last_reward_components.items()},
        }
        return state, torch.as_tensor(rewards, device=self.device), torch.as_tensor(dones, device=self.device), info

    def _training_protocol(self) -> dict[str, Any]:
        random_goal_task = self.task_mode == "random_goal_sru"
        protocol = {
                "goal_completion": "original_sru_latched_hold_single_only",
                "goal_reward_distances": {
                    "soft": "base_goal_xyz",
                    "tight": "base_goal_xy",
                },
                "goal_distance_threshold": self.reward_config.goal_distance_threshold,
                "goal_height_threshold": self.reward_config.goal_height_threshold,
                "required_goal_hold_steps": self.required_goal_hold_steps,
                "skipped_waypoints": sorted(SKIPPED_ROUTE_WAYPOINTS),
                "spawn_edge_semantics": (
                    "same_tile_independent_goal_and_spawn_masks"
                    if random_goal_task else "next_enabled_waypoint"
                ),
                "reset_distribution": (
                    "sru_spawn_mask_random_yaw"
                    if random_goal_task else "v7_safe_fraction_waypoint_start_mixture"
                ),
                "timeout_semantics": (
                    "single_random_goal" if random_goal_task else "per_active_waypoint"
                ),
                "critic_time_semantics": (
                    "single_random_goal" if random_goal_task else "per_active_waypoint"
                ),
                "task_mode": self.task_mode,
                "reset_mode": self.reset_mode,
                "randomize_waypoint_yaw": self.randomize_waypoint_yaw,
                "training_spawn_mode": self.training_spawn_mode,
                "waypoint_start_probability": self.waypoint_start_probability,
                "waypoint_yaw_jitter_deg": self.waypoint_yaw_jitter_deg,
                "safe_spawn_sha256": self.safe_spawn_sha256,
                "simulation_assets_sha256": self.simulation_assets_sha256,
                "low_level_checkpoint_sha256": self.low_level_checkpoint_sha256,
                "low_level_profile": self.low_level_profile.name,
                "low_level_profile_contract": asdict(self.low_level_profile),
                "low_level_ready_after_reset": self.low_level_ready_after_reset,
                "lidar_encoder_sha256": self.lidar_encoder_sha256,
                "height_encoder_sha256": self.height_encoder_sha256,
                "num_envs": self.num_envs,
                "low_level": self.low_level,
                "use_lidar": self.use_lidar,
                "use_height": self.use_height,
                "sensor_backend": self.sensor_backend,
                "lidar_horizontal_samples": self.lidar_horizontal_samples,
                "terrain_seed": self.terrain_seed,
                "surface_seed": self.surface_seed,
                "terrain_profile": self.terrain_profile,
                "terrain_type_counts": (
                    None if self.terrain_atlas is None else self.terrain_atlas.type_counts
                ),
                "grass_fraction": self.grass_fraction,
                "gravel_fraction": self.gravel_fraction,
                "command_noise": self.command_noise,
                "reset_position_noise": self.reset_position_noise,
                "reset_yaw_noise": self.reset_yaw_noise,
                "reset_velocity_probability": self.reset_velocity_probability,
                "reset_forward_speed_min": self.reset_forward_speed_min,
                "reset_forward_speed_max": self.reset_forward_speed_max,
                "reset_lateral_speed_max": self.reset_lateral_speed_max,
                "reset_yaw_rate_max": self.reset_yaw_rate_max,
                "entry_state_bank_sha256": self.entry_state_bank_sha256,
                "entry_state_probability": self.entry_state_probability,
                "goal_min_gap": self.goal_min_gap,
                "goal_max_gap": self.goal_max_gap,
                "max_episode_length": self.max_episode_length,
                "contact_threshold": self.contact_threshold,
                "terrain_fall_height": self.terrain_fall_height,
                "include_self_contacts": self.include_self_contacts,
                "ignore_world_body_contacts": self.ignore_world_body_contacts,
                "reset_settle_physics_steps": self.reset_settle_physics_steps,
                "contact_persistence_steps": self.contact_persistence_steps,
                "contact_force_statistic": "max_selected_contact_over_policy_step",
                "reward_config": asdict(self.reward_config),
                "action_config": asdict(self.action_spec),
                "waypoint_route": (
                    None
                    if self.waypoint_route_path is None
                    else self.waypoint_route_path.name
                ),
                "waypoint_sha256": self.waypoint_sha256(),
                "training_task": self.task_mode,
                "single_episode_length": self.single_episode_length,
                "adaptive_segment_sampling": self.adaptive_segment_sampling,
                "adaptive_uniform_mix": self.adaptive_uniform_mix,
                "adaptive_ema_alpha": self.adaptive_ema_alpha,
                "adaptive_difficulty_power": self.adaptive_difficulty_power,
                "adaptive_warmup_attempts": self.adaptive_warmup_attempts,
                "adaptive_max_probability": self.adaptive_max_probability,
                "adaptive_sampling_strategy": self.adaptive_sampling_strategy,
                "adaptive_progress_fast_alpha": self.adaptive_progress_fast_alpha,
                "adaptive_progress_slow_alpha": self.adaptive_progress_slow_alpha,
                "adaptive_progress_min_mastery": self.adaptive_progress_min_mastery,
                "adaptive_progress_max_mastery": self.adaptive_progress_max_mastery,
                "adaptive_progress_epsilon": self.adaptive_progress_epsilon,
        }
        return protocol

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "version": 16,
            "training_protocol": self._training_protocol(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "segment_sampler": (
                self.segment_sampler.state_dict()
                if self.segment_sampler is not None else None
            ),
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", 1))
        if version < 16:
            raise ValueError(
                "checkpoint predates the complete low-level profile contract; "
                "initialize weights into a fresh run instead of resuming it"
            )
        protocol = state.get("training_protocol")
        if protocol != self._training_protocol():
            raise ValueError(
                "checkpoint curriculum protocol does not match the current backend"
            )
        sampler_state = state.get("segment_sampler")
        if sampler_state is not None:
            if self.segment_sampler is None:
                raise ValueError("checkpoint has segment sampler state but backend does not")
            self.segment_sampler.load_state_dict(sampler_state)
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("checkpoint has no valid backend RNG state")
        self.rng.bit_generator.state = copy.deepcopy(rng_state)

    def close(self) -> None:
        if self._sensor_executor is not None:
            self._sensor_executor.shutdown(wait=True)
            self._sensor_executor = None
        if self._physics_executor is not None:
            self._physics_executor.shutdown(wait=True)
            self._physics_executor = None
        self.data.clear()
