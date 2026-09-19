"""Train the migrated SRU policy on the native MuJoCo backend."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure this entry point cannot accidentally import a different installed
# rsl_rl package when launched from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sru_training"))

from rsl_rl.runners import OnPolicyRunner

from sru_training.s10_mujoco_backend import (
    S10NativeMujocoBackend,
    S10RewardConfig,
)
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_policy_config import (
    S10ObservationSpec,
    mdpo_config,
    ppo_config,
    ppo_mdpo_stage1_control_config,
)


def _expand_config_args(argv: list[str]) -> list[str]:
    """Prepend YAML arguments so explicit CLI values remain authoritative."""
    boolean_optional_arguments = {
        "adaptive_segment_sampling",
        "allow_existing_log_dir",
        "init_at_random_episode_length",
        "low_level_ready_after_reset",
        "randomize_action_scale",
        "randomize_waypoint_yaw",
        "restore_curriculum_on_transfer",
    }
    config_path = None
    remaining = []
    iterator = iter(range(len(argv)))
    consumed = set()
    for index in iterator:
        value = argv[index]
        if value == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a YAML path")
            config_path = Path(argv[index + 1]).expanduser().resolve()
            consumed.update((index, index + 1))
            next(iterator, None)
        elif value.startswith("--config="):
            config_path = Path(value.split("=", 1)[1]).expanduser().resolve()
            consumed.add(index)
    remaining = [value for index, value in enumerate(argv) if index not in consumed]
    if config_path is None:
        return remaining
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arguments = payload.get("arguments") if isinstance(payload, dict) else None
    if not isinstance(arguments, dict):
        raise ValueError(f"training config must contain an 'arguments' mapping: {config_path}")
    expanded = []
    for key, value in arguments.items():
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                expanded.append(option)
            elif str(key) in boolean_optional_arguments:
                expanded.append("--no-" + str(key).replace("_", "-"))
        elif value is not None:
            expanded.extend((option, str(value)))
    print(f"[S10 training] config={config_path}", flush=True)
    return expanded + remaining


def main(argv: list[str] | None = None) -> None:
    argv = _expand_config_args(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        help="YAML protocol file; explicit CLI options override its arguments.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--num-steps-per-env",
        type=int,
        default=None,
        help="Rollout horizon per environment (default: selected algorithm's SRU setting).",
    )
    parser.add_argument(
        "--algorithm",
        choices=("mdpo", "ppo"),
        required=True,
        help="Training algorithm; choose explicitly so PPO and dual-policy MDPO runs cannot be confused.",
    )
    parser.add_argument(
        "--ppo-preset",
        choices=("sru", "mdpo_stage1_control"),
        default="sru",
        help=(
            "PPO configuration preset. mdpo_stage1_control keeps single-policy PPO but matches "
            "the successful MDPO stage-1 reward shift, discount, optimizer, schedule, and rollout."
        ),
    )
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the algorithm learning rate (recommended for checkpoint fine-tuning).",
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("fixed", "adaptive", "linear", "cosine", "exponential"),
        help="Override the algorithm learning-rate schedule.",
    )
    parser.add_argument(
        "--schedule-horizon",
        type=int,
        help=(
            "Absolute iteration horizon used by learning-rate decay. This lets a "
            "short validation run retain the original 15000-iteration SRU schedule."
        ),
    )
    parser.add_argument(
        "--action-std",
        type=float,
        help="Override action standard deviation before training starts.",
    )
    parser.add_argument(
        "--policy-dropout",
        type=float,
        help="Override SRU actor/critic consistent dropout probability.",
    )
    parser.add_argument(
        "--verify-rollout-contract",
        action="store_true",
        help="Once, compare rollout action means with recurrent PPO replay.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Strictly initialize model weights while starting a fresh optimizer and iteration count.",
    )
    parser.add_argument(
        "--init-actor-checkpoint",
        type=Path,
        help=(
            "Initialize actor weights while leaving critics and optimizers fresh. "
            "MDPO preserves both checkpoint actors when policy 2 is available. "
            "The critic, optimizer, and iteration count start fresh; saved curriculum state is restored "
            "when adaptive segment sampling is enabled."
        ),
    )
    parser.add_argument(
        "--init-task-transfer-checkpoint",
        type=Path,
        help=(
            "Initialize actors and critic feature/recurrent layers from a checkpoint, "
            "but reset scalar value heads and optimizers for a changed task protocol."
        ),
    )
    parser.add_argument(
        "--init-policy2-task-transfer-checkpoint",
        type=Path,
        help=(
            "MDPO only: initialize policy 2 from this checkpoint while policy 1 uses "
            "--init-task-transfer-checkpoint. Value heads and optimizers remain fresh."
        ),
    )
    parser.add_argument(
        "--init-policy1-state-key",
        choices=("model_state_dict", "model_state_dict_2"),
        default="model_state_dict",
        help="State key used for policy 1 during task-transfer initialization.",
    )
    parser.add_argument(
        "--init-policy2-state-key",
        choices=("model_state_dict", "model_state_dict_2"),
        default=None,
        help=(
            "State key used for policy 2 during task transfer. By default a single "
            "checkpoint uses model_state_dict_2, while a second checkpoint uses "
            "model_state_dict for backward compatibility."
        ),
    )
    parser.add_argument(
        "--recover-actor-checkpoint",
        type=Path,
        help=(
            "Recover actor weights and iteration while keeping the newly initialized critic and optimizer. "
            "Use this after correcting invalid value targets in an existing run."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--low-level",
        choices=("official_onnx", "pim_him"),
        default="official_onnx",
        help="Locomotion backend. The factory ONNX remains the default.",
    )
    parser.add_argument(
        "--low-level-checkpoint",
        type=Path,
        help="Low-level ONNX or PIM-HIM .pt checkpoint used by every environment.",
    )
    parser.add_argument(
        "--low-level-profile",
        choices=("legacy", "official_20260828"),
        default="legacy",
        help="Runner parameters that must match the selected low-level ONNX.",
    )
    parser.add_argument(
        "--low-level-ready-after-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat a settled simulator episode reset as an already completed "
            "low-level startup transition. Hardware/GUI startup keeps the ramp."
        ),
    )
    parser.add_argument(
        "--lidar-encoder-checkpoint",
        type=Path,
        help=(
            "LiDAR encoder used to construct actor observations. Maintained configs "
            "always provide the frozen random-terrain encoder explicitly."
        ),
    )
    parser.add_argument("--log-dir", default="logs/s10_mujoco_sru")
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--contact-threshold", type=float, default=500.0)
    parser.add_argument("--contact-persistence-steps", type=int, default=1)
    parser.add_argument(
        "--goal-progress",
        type=float,
        default=0.0,
        help=(
            "Dense reward coefficient for reducing XY goal distance per high-level step. "
            "Zero preserves the original sparse SRU reward."
        ),
    )
    parser.add_argument("--reset-mode", choices=("fixed", "random_waypoint"), default="fixed")
    parser.add_argument(
        "--randomize-waypoint-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample calibrated route-relative start headings.",
    )
    parser.add_argument(
        "--training-spawn-mode",
        choices=("waypoint_start", "safe_fraction"),
        default="waypoint_start",
        help=(
            "waypoint_start resets at the real waypoint with only a calibrated nearest "
            "fallback; safe_fraction preserves the obsolete line-interior distribution."
        ),
    )
    parser.add_argument(
        "--waypoint-yaw-jitter-deg",
        type=float,
        default=10.0,
        help="Uniform route-relative yaw jitter for waypoint_start resets.",
    )
    parser.add_argument(
        "--reset-velocity-probability",
        type=float,
        default=0.0,
        help="Probability of applying a moving root-state reset after MuJoCo settling.",
    )
    parser.add_argument("--reset-forward-speed-min", type=float, default=0.0)
    parser.add_argument("--reset-forward-speed-max", type=float, default=0.0)
    parser.add_argument("--reset-lateral-speed-max", type=float, default=0.0)
    parser.add_argument("--reset-yaw-rate-max", type=float, default=0.0)
    parser.add_argument(
        "--randomize-action-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy experiment switch. Upstream MX disables action-scale and "
            "bias randomization; only enable this explicitly for an old protocol."
        ),
    )
    parser.add_argument(
        "--entry-state-bank",
        type=Path,
        help="Compressed bank of real continuous waypoint-entry states.",
    )
    parser.add_argument(
        "--entry-state-probability",
        type=float,
        default=0.0,
        help="Probability of restoring a matching real entry state at reset.",
    )
    parser.add_argument(
        "--task-mode",
        choices=("waypoint", "random_goal_sru"),
        default="random_goal_sru",
        help="Maintained training configs use the SRU random-goal task.",
    )
    parser.add_argument("--terrain-seed", type=int)
    parser.add_argument(
        "--terrain-profile",
        choices=(
            "legacy_full",
            "stage1_flat",
            "stage2_low_density_obstacles",
            "stage3_reduced_height",
            "stage4_full_no_pits",
            "stage5_lower_density_stairs",
        ),
        default="legacy_full",
        help="Named terrain curriculum profile for random_goal_sru.",
    )
    parser.add_argument(
        "--surface-seed",
        type=int,
        help="Independent seed for flat-ground grass/gravel assignment; defaults to terrain seed.",
    )
    parser.add_argument("--grass-fraction", type=float, default=0.25)
    parser.add_argument("--gravel-fraction", type=float, default=0.25)
    parser.add_argument("--terrain-rows", type=int, default=6)
    parser.add_argument("--terrain-cols", type=int, default=30)
    parser.add_argument(
        "--waypoint-route",
        type=Path,
        help="Ordered waypoint route used by adjacent training and replay.",
    )
    parser.add_argument(
        "--safe-spawn-candidates",
        type=Path,
        default=None,
        help="Safe reset calibration matching the selected waypoint route.",
    )
    parser.add_argument(
        "--waypoint-start-probability",
        type=float,
        default=None,
        help="Probability of strict waypoint reset; omitted derives from training-spawn-mode.",
    )
    parser.add_argument(
        "--single-episode-length",
        type=int,
        default=300,
        help="Timeout for one adjacent-goal episode.",
    )
    parser.add_argument(
        "--adaptive-segment-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bias resets toward route segments with lower online success rates.",
    )
    parser.add_argument("--adaptive-uniform-mix", type=float, default=0.6)
    parser.add_argument("--adaptive-ema-alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-difficulty-power", type=float, default=1.0)
    parser.add_argument("--adaptive-warmup-attempts", type=int, default=5)
    parser.add_argument(
        "--adaptive-max-probability",
        type=float,
        default=1.0,
        help="Maximum probability assigned to one template inside an adaptive bucket.",
    )
    parser.add_argument(
        "--adaptive-sampling-strategy",
        choices=("difficulty", "learning_progress"),
        default="difficulty",
    )
    parser.add_argument("--adaptive-progress-fast-alpha", type=float, default=0.10)
    parser.add_argument("--adaptive-progress-slow-alpha", type=float, default=0.01)
    parser.add_argument("--adaptive-progress-min-mastery", type=float, default=0.05)
    parser.add_argument("--adaptive-progress-max-mastery", type=float, default=0.95)
    parser.add_argument("--adaptive-progress-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--distill-coef",
        type=float,
        default=None,
        help="Override the MDPO mutual-distillation target coefficient.",
    )
    parser.add_argument(
        "--distill-warmup-iterations",
        type=int,
        default=0,
        help="MDPO iterations with mutual distillation fully disabled.",
    )
    parser.add_argument(
        "--distill-ramp-iterations",
        type=int,
        default=0,
        help="Iterations used to linearly ramp mutual distillation after warmup.",
    )
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument(
        "--init-at-random-episode-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Randomize the initial episode phase as in the upstream IsaacLab "
            "training entry point; backend and VecEnv clocks are synchronized."
        ),
    )
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-height", action="store_true")
    parser.add_argument(
        "--sensor-workers",
        type=int,
        default=None,
        help="Sensor raycast worker threads (default: min(num_envs, 8)).",
    )
    parser.add_argument(
        "--physics-workers",
        type=int,
        default=None,
        help="MuJoCo environment worker threads (default: min(num_envs, 8)).",
    )
    parser.add_argument(
        "--lidar-horizontal-samples",
        type=int,
        choices=(90, 180, 270, 450, 900),
        default=900,
        help="LiDAR rays per row/view; 900 is exact and practical with the Warp backend.",
    )
    parser.add_argument(
        "--sensor-backend",
        choices=("warp", "cpu"),
        default="warp",
        help="LiDAR raycast backend; warp uses the static terrain CUDA BVH.",
    )
    parser.add_argument(
        "--allow-existing-log-dir",
        action=argparse.BooleanOptionalAction,
        help="Allow writing TensorBoard events into a directory containing an earlier run.",
    )
    parser.add_argument(
        "--restore-curriculum-on-transfer",
        action=argparse.BooleanOptionalAction,
        help=(
            "Restore sampler statistics with --init-task-transfer-checkpoint. Disabled by "
            "default because changed task semantics invalidate old curriculum outcomes."
        ),
    )
    args = parser.parse_args(argv)
    if args.algorithm == "mdpo" and (args.num_envs < 2 or args.num_envs % 2):
        raise ValueError("--algorithm mdpo requires an even --num-envs of at least 2")
    if args.schedule_horizon is not None and args.schedule_horizon < args.iterations:
        raise ValueError("--schedule-horizon must be at least --iterations")
    if args.terrain_rows < 1 or args.terrain_cols < 1:
        raise ValueError("terrain rows and columns must be positive")
    if args.task_mode == "random_goal_sru":
        if args.terrain_cols != 30:
            raise ValueError("the equivalent SRU task requires --terrain-cols 30")
        if args.reset_mode != "fixed":
            raise ValueError("random_goal_sru uses terrain masks and requires --reset-mode fixed")
        if args.waypoint_route is not None:
            raise ValueError("--waypoint-route is incompatible with random_goal_sru")
        if args.entry_state_bank is not None or args.entry_state_probability != 0.0:
            raise ValueError("entry-state resets are incompatible with random_goal_sru")
        if args.adaptive_segment_sampling:
            raise ValueError("waypoint adaptive sampling is incompatible with random_goal_sru")
    if args.algorithm != "ppo" and args.ppo_preset != "sru":
        raise ValueError("--ppo-preset is only meaningful with --algorithm ppo")
    if args.init_policy2_task_transfer_checkpoint is not None and (
        args.algorithm != "mdpo" or args.init_task_transfer_checkpoint is None
    ):
        raise ValueError(
            "--init-policy2-task-transfer-checkpoint requires MDPO and "
            "--init-task-transfer-checkpoint"
        )
    if (
        args.init_policy1_state_key != "model_state_dict"
        or args.init_policy2_state_key is not None
    ) and (args.algorithm != "mdpo" or args.init_task_transfer_checkpoint is None):
        raise ValueError(
            "explicit task-transfer state keys require MDPO and "
            "--init-task-transfer-checkpoint"
        )
    if args.algorithm != "mdpo" and (
        args.distill_coef is not None
        or args.distill_warmup_iterations
        or args.distill_ramp_iterations
    ):
        raise ValueError("distillation options are MDPO-only")
    checkpoint_modes = sum(
        value is not None
        for value in (
            args.resume,
            args.init_checkpoint,
            args.init_actor_checkpoint,
            args.init_task_transfer_checkpoint,
            args.recover_actor_checkpoint,
        )
    )
    if checkpoint_modes > 1:
        raise ValueError(
            "--resume and all checkpoint initialization/recovery modes are mutually exclusive"
        )
    if args.num_steps_per_env is not None and args.num_steps_per_env < 2:
        raise ValueError("--num-steps-per-env must be at least 2 for recurrent SRU rollouts")
    if args.save_interval < 1:
        raise ValueError("--save-interval must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.action_std is not None and args.action_std <= 0.0:
        raise ValueError("--action-std must be positive")
    if args.policy_dropout is not None and not 0.0 <= args.policy_dropout < 1.0:
        raise ValueError("--policy-dropout must be in [0, 1)")
    if args.contact_threshold < 0.0:
        raise ValueError("--contact-threshold must be non-negative")
    if args.contact_persistence_steps < 1:
        raise ValueError("--contact-persistence-steps must be positive")
    if not 0.0 <= args.entry_state_probability <= 1.0:
        raise ValueError("--entry-state-probability must be in [0, 1]")
    if args.entry_state_probability > 0.0 and args.entry_state_bank is None:
        raise ValueError("--entry-state-probability requires --entry-state-bank")
    if args.single_episode_length < 1:
        raise ValueError("--single-episode-length must be positive")
    if args.restore_curriculum_on_transfer and args.init_task_transfer_checkpoint is None:
        raise ValueError(
            "--restore-curriculum-on-transfer requires --init-task-transfer-checkpoint"
        )
    if args.resume is not None and any(
        value is not None
        for value in (args.learning_rate, args.learning_rate_schedule, args.action_std)
    ):
        raise ValueError(
            "--resume is an exact continuation; learning-rate and action-std overrides are not allowed"
        )
    if args.sensor_workers is not None and args.sensor_workers < 1:
        raise ValueError("--sensor-workers must be positive")
    if args.physics_workers is not None and args.physics_workers < 1:
        raise ValueError("--physics-workers must be positive")
    if not 0.0 <= args.waypoint_yaw_jitter_deg <= 180.0:
        raise ValueError("--waypoint-yaw-jitter-deg must be in [0, 180]")
    if not 0.0 <= args.adaptive_uniform_mix <= 1.0:
        raise ValueError("--adaptive-uniform-mix must be in [0, 1]")
    if not 0.0 < args.adaptive_ema_alpha <= 1.0:
        raise ValueError("--adaptive-ema-alpha must be in (0, 1]")
    if args.adaptive_difficulty_power <= 0.0:
        raise ValueError("--adaptive-difficulty-power must be positive")
    if args.adaptive_warmup_attempts < 0:
        raise ValueError("--adaptive-warmup-attempts must be non-negative")
    if not 0.0 < args.adaptive_max_probability <= 1.0:
        raise ValueError("--adaptive-max-probability must be in (0, 1]")
    if args.waypoint_start_probability is not None and not 0.0 <= args.waypoint_start_probability <= 1.0:
        raise ValueError("--waypoint-start-probability must be in [0, 1]")
    if not 0.0 < args.adaptive_progress_slow_alpha < args.adaptive_progress_fast_alpha <= 1.0:
        raise ValueError("adaptive progress EMA alphas require 0 < slow < fast <= 1")
    if not 0.0 <= args.adaptive_progress_min_mastery < args.adaptive_progress_max_mastery <= 1.0:
        raise ValueError("adaptive progress mastery band must lie inside [0, 1]")
    if args.adaptive_progress_epsilon < 0.0:
        raise ValueError("--adaptive-progress-epsilon must be non-negative")
    if args.goal_progress < 0.0:
        raise ValueError("--goal-progress must be non-negative")
    if args.distill_coef is not None and args.distill_coef < 0.0:
        raise ValueError("--distill-coef must be non-negative")
    if args.distill_warmup_iterations < 0 or args.distill_ramp_iterations < 0:
        raise ValueError("distillation warmup and ramp iterations must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.algorithm == "mdpo":
        cfg = mdpo_config(smoke=False)
    elif args.ppo_preset == "mdpo_stage1_control":
        cfg = ppo_mdpo_stage1_control_config(smoke=False)
    else:
        cfg = ppo_config(smoke=False)
    num_steps_per_env = (
        int(cfg["num_steps_per_env"])
        if args.num_steps_per_env is None
        else args.num_steps_per_env
    )
    log_dir = Path(args.log_dir).expanduser().resolve()
    existing_run_files = list(log_dir.glob("events.out.tfevents.*")) + list(log_dir.glob("model_*.pt"))
    if existing_run_files and args.resume is None and not args.allow_existing_log_dir:
        raise FileExistsError(
            f"Log directory already contains a training run: {log_dir}. "
            "Choose a new --log-dir to keep TensorBoard curves and checkpoints unambiguous."
        )
    rollout_transitions = args.num_envs * num_steps_per_env
    sensor_workers = (
        0
        if args.no_lidar and args.no_height
        else args.sensor_workers or min(args.num_envs, 8)
    )
    physics_workers = args.physics_workers or min(args.num_envs, 8)
    print(
        "[S10 training] "
        f"algorithm={args.algorithm} num_envs={args.num_envs} steps_per_env={num_steps_per_env} "
        f"transitions_per_iteration={rollout_transitions} sensor_workers={sensor_workers} "
        f"physics_workers={physics_workers}; MuJoCo and height raycasts use CPU thread pools, "
        f"lidar_horizontal_samples={args.lidar_horizontal_samples}; "
        f"sensor_backend={args.sensor_backend}; "
        f"training_task={args.task_mode}; "
        f"terrain_profile={args.terrain_profile}; "
        f"surface_seed={args.surface_seed} grass={args.grass_fraction} gravel={args.gravel_fraction}; "
        "goal_gap=1:1; "
        f"training_spawn_mode={args.training_spawn_mode}; "
        f"waypoint_start_probability={args.waypoint_start_probability}; "
        f"waypoint_yaw_jitter_deg={args.waypoint_yaw_jitter_deg}; "
        f"reset_velocity={args.reset_velocity_probability} "
        f"forward_speed={args.reset_forward_speed_min}:{args.reset_forward_speed_max} "
        f"lateral_speed_max={args.reset_lateral_speed_max} "
        f"yaw_rate_max={args.reset_yaw_rate_max}; "
        f"entry_state_bank={args.entry_state_bank} "
        f"entry_state_probability={args.entry_state_probability}; "
        f"waypoint_route={args.waypoint_route} "
        f"safe_spawn_candidates={args.safe_spawn_candidates}; "
        f"contact={args.contact_threshold}N/{args.contact_persistence_steps}step; "
        f"goal_progress={args.goal_progress}; "
        "actor_observation=current_goal_only; "
        f"adaptive_segment_sampling={args.adaptive_segment_sampling}; "
        f"adaptive_strategy={args.adaptive_sampling_strategy} "
        f"uniform_mix={args.adaptive_uniform_mix} "
        f"max_probability={args.adaptive_max_probability}; "
        f"low_level={args.low_level}; learned low-level controllers use batch inference.",
        flush=True,
    )

    backend = S10NativeMujocoBackend(
        num_envs=args.num_envs,
        device=args.device,
        low_level=args.low_level,
        low_level_checkpoint=args.low_level_checkpoint,
        low_level_profile=args.low_level_profile,
        low_level_ready_after_reset=args.low_level_ready_after_reset,
        lidar_encoder_checkpoint=args.lidar_encoder_checkpoint,
        task_mode=args.task_mode,
        terrain_seed=args.terrain_seed,
        surface_seed=args.surface_seed,
        terrain_profile=args.terrain_profile,
        grass_fraction=args.grass_fraction,
        gravel_fraction=args.gravel_fraction,
        terrain_rows=args.terrain_rows,
        terrain_cols=args.terrain_cols,
        waypoint_route=args.waypoint_route,
        max_episode_length=args.max_episode_length,
        reset_mode=args.reset_mode,
        randomize_waypoint_yaw=args.randomize_waypoint_yaw,
        training_spawn_mode=args.training_spawn_mode,
        waypoint_start_probability=args.waypoint_start_probability,
        waypoint_yaw_jitter_deg=args.waypoint_yaw_jitter_deg,
        reset_velocity_probability=args.reset_velocity_probability,
        reset_forward_speed_min=args.reset_forward_speed_min,
        reset_forward_speed_max=args.reset_forward_speed_max,
        reset_lateral_speed_max=args.reset_lateral_speed_max,
        reset_yaw_rate_max=args.reset_yaw_rate_max,
        entry_state_bank=args.entry_state_bank,
        entry_state_probability=args.entry_state_probability,
        safe_spawn_candidates=args.safe_spawn_candidates,
        contact_threshold=args.contact_threshold,
        contact_persistence_steps=args.contact_persistence_steps,
        goal_min_gap=1,
        goal_max_gap=1,
        single_episode_length=args.single_episode_length,
        adaptive_segment_sampling=args.adaptive_segment_sampling,
        adaptive_uniform_mix=args.adaptive_uniform_mix,
        adaptive_ema_alpha=args.adaptive_ema_alpha,
        adaptive_difficulty_power=args.adaptive_difficulty_power,
        adaptive_warmup_attempts=args.adaptive_warmup_attempts,
        adaptive_max_probability=args.adaptive_max_probability,
        adaptive_sampling_strategy=args.adaptive_sampling_strategy,
        adaptive_progress_fast_alpha=args.adaptive_progress_fast_alpha,
        adaptive_progress_slow_alpha=args.adaptive_progress_slow_alpha,
        adaptive_progress_min_mastery=args.adaptive_progress_min_mastery,
        adaptive_progress_max_mastery=args.adaptive_progress_max_mastery,
        adaptive_progress_epsilon=args.adaptive_progress_epsilon,
        reward_config=S10RewardConfig(goal_progress=args.goal_progress),
        seed=args.seed,
        use_lidar=not args.no_lidar,
        use_height=not args.no_height,
        sensor_workers=args.sensor_workers,
        physics_workers=args.physics_workers,
        lidar_horizontal_samples=args.lidar_horizontal_samples,
        sensor_backend=args.sensor_backend,
    )
    obs_spec = S10ObservationSpec()
    env = S10MujocoVecEnv(
        backend,
        obs_spec=obs_spec,
        randomize_action_scale=args.randomize_action_scale,
        device=args.device,
    )
    cfg["seed"] = args.seed
    cfg["num_steps_per_env"] = num_steps_per_env
    cfg["save_interval"] = args.save_interval
    if args.policy_dropout is not None:
        cfg["policy"]["dropout"] = args.policy_dropout
    if args.schedule_horizon is not None:
        cfg["schedule_max_iterations"] = args.schedule_horizon
    if args.learning_rate is not None:
        cfg["algorithm"]["learning_rate"] = args.learning_rate
    if args.learning_rate_schedule is not None:
        cfg["algorithm"]["schedule"] = args.learning_rate_schedule
    cfg["algorithm"]["verify_rollout_contract"] = args.verify_rollout_contract
    if args.distill_coef is not None:
        cfg["algorithm"]["distill_coef"] = args.distill_coef
    if args.algorithm == "mdpo":
        cfg["algorithm"]["distill_warmup_iterations"] = (
            args.distill_warmup_iterations
        )
        cfg["algorithm"]["distill_ramp_iterations"] = args.distill_ramp_iterations
    # Recurrent rollout storage uses contiguous environment/time batches. Do
    # not request more mini-batches than the available environment dimension;
    # this also makes tiny smoke runs exercise the same updater safely.
    environments_per_policy = args.num_envs // 2 if args.algorithm == "mdpo" else args.num_envs
    cfg["algorithm"]["num_mini_batches"] = min(
        cfg["algorithm"]["num_mini_batches"], max(1, environments_per_policy)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=args.device)
    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        resume_infos = runner.load(str(checkpoint), load_optimizer=True)
        if isinstance(resume_infos, dict) and resume_infos.get("env_training_state"):
            env.load_training_state(resume_infos["env_training_state"])
            env.reset()
            print("[S10 training] restored environment curriculum state", flush=True)
        elif args.adaptive_segment_sampling:
            print(
                "[S10 training] checkpoint has no curriculum state; adaptive sampler starts with warmup",
                flush=True,
            )
        print(f"[S10 training] resumed checkpoint={checkpoint}", flush=True)
    elif args.init_actor_checkpoint is not None:
        checkpoint = args.init_actor_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Actor initialization checkpoint does not exist: {checkpoint}")
        initialization_infos = runner.initialize_actor_from_checkpoint(str(checkpoint))
        if (
            args.adaptive_segment_sampling
            and isinstance(initialization_infos, dict)
            and initialization_infos.get("env_training_state")
        ):
            env.load_training_state(initialization_infos["env_training_state"])
            env.reset()
            print(
                "[S10 training] restored environment curriculum state for actor-initialized run",
                flush=True,
            )
        elif args.adaptive_segment_sampling:
            print(
                "[S10 training] checkpoint has no curriculum state; adaptive sampler starts with warmup",
                flush=True,
            )
        print(
            f"[S10 training] initialized checkpoint actor(s) from checkpoint={checkpoint}; "
            "critics, optimizers, and iteration start fresh",
            flush=True,
        )
    elif args.init_task_transfer_checkpoint is not None:
        checkpoint = args.init_task_transfer_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Task-transfer checkpoint does not exist: {checkpoint}")
        policy_2_checkpoint = (
            None
            if args.init_policy2_task_transfer_checkpoint is None
            else args.init_policy2_task_transfer_checkpoint.expanduser().resolve()
        )
        if policy_2_checkpoint is not None and not policy_2_checkpoint.is_file():
            raise FileNotFoundError(
                f"Policy-2 task-transfer checkpoint does not exist: {policy_2_checkpoint}"
            )
        (initialization_infos, reset_heads) = runner.initialize_task_transfer_from_checkpoint(
            str(checkpoint),
            None if policy_2_checkpoint is None else str(policy_2_checkpoint),
            policy_1_state_key=args.init_policy1_state_key,
            policy_2_state_key=args.init_policy2_state_key,
        )
        if (
            args.restore_curriculum_on_transfer
            and args.adaptive_segment_sampling
            and isinstance(initialization_infos, dict)
            and initialization_infos.get("env_training_state")
        ):
            env.load_training_state(initialization_infos["env_training_state"])
            env.reset()
            print(
                "[S10 training] restored compatible curriculum state for task transfer",
                flush=True,
            )
        elif args.adaptive_segment_sampling:
            print(
                "[S10 training] task transfer starts with fresh curriculum state",
                flush=True,
            )
        print(
            f"[S10 training] task-transfer initialized from checkpoint={checkpoint}; "
            f"policy_1_state_key={args.init_policy1_state_key}; "
            f"policy_2_checkpoint={policy_2_checkpoint}; "
            f"policy_2_state_key={args.init_policy2_state_key}; "
            f"reset_value_heads={reset_heads}; optimizers and iteration start fresh",
            flush=True,
        )
    elif args.recover_actor_checkpoint is not None:
        checkpoint = args.recover_actor_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Actor recovery checkpoint does not exist: {checkpoint}")
        recovery_infos = runner.recover_actor_from_checkpoint(str(checkpoint))
        if isinstance(recovery_infos, dict) and recovery_infos.get("env_training_state"):
            env.load_training_state(recovery_infos["env_training_state"])
            env.reset()
            print("[S10 training] restored environment curriculum state", flush=True)
        elif args.adaptive_segment_sampling:
            print(
                "[S10 training] checkpoint has no curriculum state; adaptive sampler starts with warmup",
                flush=True,
            )
        print(
            f"[S10 training] recovered actor and iteration from checkpoint={checkpoint}; "
            "critic and optimizer start fresh",
            flush=True,
        )
    elif args.init_checkpoint is not None:
        checkpoint = args.init_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint}")
        runner.initialize_from_checkpoint(str(checkpoint))
        print(
            f"[S10 training] initialized model weights from checkpoint={checkpoint}; "
            "optimizer and iteration start fresh",
            flush=True,
        )
    if args.action_std is not None:
        previous_stds = runner.set_action_std(args.action_std)
        print(
            f"[S10 training] action std override: {previous_stds} -> {args.action_std:.4f}",
            flush=True,
        )
    print(
        "[S10 training] optimizer settings: "
        f"learning_rate={runner.alg.learning_rate:.6g} "
        f"schedule={runner.alg.schedule} "
        f"schedule_horizon={cfg.get('schedule_max_iterations', args.iterations)} "
        f"reward_shift={cfg.get('reward_shifting_value', 0.0):.4f} "
        f"distill={getattr(runner.alg, 'distill_coef', 0.0):.6g}/"
        f"{getattr(runner.alg, 'distill_warmup_iterations', 0)}/"
        f"{getattr(runner.alg, 'distill_ramp_iterations', 0)} "
        f"optimizer_groups={len((runner.alg.optimizer_1 if args.algorithm == 'mdpo' else runner.alg.optimizer).param_groups)}",
        flush=True,
    )
    try:
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=args.init_at_random_episode_length,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
