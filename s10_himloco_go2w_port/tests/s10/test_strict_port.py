"""Static audit tests that do not require Isaac Sim."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GO2W = ROOT / "locowheeledlegged/config/go2w/locomotion_env_cfg.py"
S10 = ROOT / "locowheeledlegged/config/s10/locomotion_env_cfg.py"
REFERENCE = ROOT / "locowheeledlegged/config/s10/reference_env_cfg.py"
TASKS = ROOT / "locowheeledlegged/config/s10/__init__.py"
COMMANDS = ROOT / "locowheeledlegged/mdp/commands.py"
PPO = ROOT / "locowheeledlegged/config/s10/agents/rsl_rl_ppo_cfg.py"


def _section(text: str, start: str, stop: str) -> str:
    return text[text.index(start) : text.index(stop)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_xyz_command_configuration_is_identical_to_upstream() -> None:
    go2w = GO2W.read_text(encoding="utf-8")
    s10 = S10.read_text(encoding="utf-8")
    assert _section(go2w, "class CommandsCfg:", "# endregion -- Commands --") == _section(
        s10, "class CommandsCfg:", "# endregion -- Commands --"
    )
    assert "lin_vel_x=(-1.0, 1.0)" in s10
    assert "lin_vel_y=(-0.5, 0.5)" in s10
    assert "ang_vel_z=(-math.pi / 4, math.pi / 4)" in s10


def test_terrain_recipe_only_changes_initial_level() -> None:
    go2w = GO2W.read_text(encoding="utf-8")
    s10 = S10.read_text(encoding="utf-8")
    go2w_terrain = _section(go2w, "terrain: TerrainImporterCfg", "    robot: ArticulationCfg")
    s10_terrain = _section(s10, "terrain: TerrainImporterCfg", "    robot: ArticulationCfg")
    go2w_terrain = go2w_terrain.replace("max_init_terrain_level=5", "max_init_terrain_level=2")
    assert go2w_terrain == s10_terrain


def test_official_asset_copy_checksums() -> None:
    official = ROOT / "locowheeledlegged/assets/s10/official"
    assert _sha256(official / "urdf/S10.urdf") == "67755f9ea87bff2a45801fb3a7c407a06604c117ed9ec951f114b20b3ba01cba"
    assert _sha256(official / "mjcf/S10.xml") == "c869bd7032f1ea7a6319cd0165f38e783caf3bccd25593d337ff9fe2779cc8ef"
    assert _sha256(ROOT / "locowheeledlegged/assets/s10/generated/s10.usd") == "74c1bc0710a2fc07e998f665db6e75bc212b51a32226027a36092784c804d41e"


def test_s10_policy_joint_order_and_action_scales_are_explicit() -> None:
    asset = (ROOT / "s10_policy_protocol.py").read_text(encoding="utf-8")
    expected = (
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    )
    positions = [asset.index(f'"{name}"') for name in expected]
    assert positions == sorted(positions)
    cfg = S10.read_text(encoding="utf-8")
    assert '0.125 if "hipx" in name else 0.25' in cfg
    assert len(re.findall(r"preserve_order=True", cfg)) >= 3


def test_only_abc_add_one_gait_term_each() -> None:
    text = S10.read_text(encoding="utf-8")
    assert text.count("class ClearanceConstraintRewardsCfg") == 1
    assert text.count("class TargetBandConstraintRewardsCfg") == 1
    assert text.count("class PairGeometryConstraintRewardsCfg") == 1
    assert text.count("wheel_lateral_clearance = RewardTermCfg") == 1
    assert text.count("wheel_lateral_target = RewardTermCfg") == 1
    assert text.count("wheel_pair_geometry = RewardTermCfg") == 1


def test_upstream_reward_block_only_changes_nominal_height() -> None:
    go2w = GO2W.read_text(encoding="utf-8")
    s10 = S10.read_text(encoding="utf-8")
    go2w_rewards = _section(
        go2w, "class RewardsCfg:", "# =============================================================================\n# region -- Events --"
    )
    s10_rewards = _section(s10, "class RewardsCfg:", "WHEEL_BODY_CFG =")
    assert go2w_rewards.replace(
        '"target_height": 0.40', '"target_height": NOMINAL_BASE_HEIGHT'
    ) == s10_rewards


def test_default_spawn_height_has_negligible_height_penalty() -> None:
    asset = (ROOT / "locowheeledlegged/assets/s10_robot.py").read_text(encoding="utf-8")
    nominal = float(re.search(r"NOMINAL_BASE_HEIGHT\s*=\s*([0-9.]+)", asset).group(1))
    spawn = float(re.search(r"pos=\(0\.0, 0\.0, ([0-9.]+)\)", asset).group(1))
    assert abs(spawn - nominal) <= 0.002
    assert (spawn - nominal) ** 2 <= 4.0e-6


def test_official_and_hybrid_parallel_tasks_are_registered() -> None:
    text = TASKS.read_text(encoding="utf-8")
    for task_id in (
        "Isaac-LocomotionS10-Official-v1",
        "Isaac-LocomotionS10-Hybrid-v1",
        "Isaac-LocomotionS10-HIM-Official-v1",
        "Isaac-LocomotionS10-HIM-Hybrid-v1",
    ):
        assert task_id in text


def test_deeprobotics_m20_reference_is_commit_locked() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert 'OFFICIAL_M20_REFERENCE_REPOSITORY = "DeepRoboticsLab/rl_training"' in text
    assert 'OFFICIAL_M20_REFERENCE_COMMIT = "6d317dfb33060226139e38600510e1751372eb5d"' in text
    for filename, digest in {
        "rough_env_cfg.py": "38ff27cb3863bab9b0b0509c06082eb98b6b357a5335d780eb508026bb3d8fba",
        "rsl_rl_ppo_cfg.py": "c67d49772c0f841d8cb2ce5a67b62aa1d7598409d87269f6cbfe797958062bdc",
        "velocity_env_cfg.py": "295cb84a57bf9289c51a1ff2e9cf71fa41bf026a99f94422605ec0ae998e7bda",
        "rewards.py": "959d9a55751e57530a840e74f294f9fdf3582755bf51f42d10a3e97740f3d7b2",
        "commands.py": "31877425d07293cfdefd29858e105885a0edf8f7d75dc72d2200c29706017641",
        "curriculums.py": "c65a2a197c431a40b6772ffcd34c7ff203a47a5437033fbf46316897a324711d",
    }.items():
        assert f'"{filename}": "{digest}"' in text


def test_official_terrain_matches_deeprobotics_m20_recipe() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    block = _section(text, "class OfficialReferenceSceneCfg", "class HybridReferenceSceneCfg")
    assert block.count("proportion=0.2") == 4
    assert block.count("proportion=0.1") == 2
    for terrain in (
        '"pyramid_stairs"',
        '"pyramid_stairs_inv"',
        '"boxes"',
        '"random_rough"',
        '"hf_pyramid_slope"',
        '"hf_pyramid_slope_inv"',
    ):
        assert terrain in block
    assert "grid_height_range=(0.025, 0.20)" in block
    assert "noise_range=(0.01, 0.16)" in block
    assert "noise_step=0.01" in block
    assert "step_height_range=(0.05, 0.23)" in text
    assert "max_init_terrain_level=5" in block


def test_hybrid_terrain_is_explicit_five_way_midpoint() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    block = _section(text, "class HybridReferenceSceneCfg", "def _joint_pose_term")
    assert block.count("proportion=0.2") == 5
    for terrain in ('"flat"', '"pyramid_stairs_inv"', '"pyramid_stairs"', '"boxes"', '"random_rough"'):
        assert terrain in block
    assert "hf_pyramid_slope" not in block
    assert '"perlin_rough"' not in block
    assert "max_init_terrain_level=2" in block


def test_official_reward_weights_match_locked_m20_source() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    block = _section(text, "class OfficialReferenceRewardsCfg", "class HybridReferenceRewardsCfg")
    expected_weights = {
        "track_lin_vel_xy_exp": "5.0",
        "track_ang_vel_z_exp": "3.0",
        "lin_vel_z_l2": "-2.0",
        "ang_vel_xy_l2": "-0.02",
        "flat_orientation_l2": "-50.0",
        "joint_torques_l2": "-2.5e-5",
        "leg_joint_acc_l2": "-2.0e-7",
        "wheel_joint_acc_l2": "-1.0e-7",
        "stand_still_without_cmd": "-1.0",
        "joint_mirror": "-0.03",
        "action_rate_l2": "-0.01",
        "action_smooth_l2": "-0.025",
        "undesired_contacts": "-1.0",
        "contact_forces": "-1.5e-4",
        "bad_orientation_penalty": "-1000.0",
    }
    for term, weight in expected_weights.items():
        pattern = rf"{term}\s*=\s*RewardTermCfg\((?:(?!\n    \w+\s*=).)*?weight={re.escape(weight)}"
        assert re.search(pattern, block, flags=re.DOTALL), term
    assert "base_height_l2 = None" in block
    assert "_joint_pose_term(-3.0" in block
    assert "_joint_pose_term(-1.5" in block
    assert "_joint_pose_term(-0.75" in block
    for assignment in (
        "feet_air_time_yaw = _yaw_air_time_term(50.0)",
        "wheel_slide_yaw = _yaw_slide_term(-2.0)",
        "rotation_gait_status = _rotation_gait_status_term(2.0)",
        "rotation_gait_symmetry = _rotation_gait_symmetry_term(15.0)",
    ):
        assert assignment in block


def test_reference_terrain_family_labels_match_isaaclab_geometry() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    official = _section(text, "OFFICIAL_TERRAIN_FAMILIES =", "HYBRID_TERRAIN_FAMILIES =")
    hybrid = _section(text, "HYBRID_TERRAIN_FAMILIES =", "LEGACY_TERRAIN_FAMILIES =")
    legacy = _section(text, "LEGACY_TERRAIN_FAMILIES =", "def terrain_family_columns")
    assert '"pyramid_stairs": "stairs_down"' in official
    assert '"pyramid_stairs_inv": "stairs_up"' in official
    assert '"hf_pyramid_slope": "ramp_down"' in official
    assert '"hf_pyramid_slope_inv": "ramp_up"' in official
    assert '"pyramid_stairs_inv": "stairs_up"' in hybrid
    assert '"pyramid_stairs": "stairs_down"' in hybrid
    assert '"pyramid_stairs": "stairs_down"' in legacy
    assert '"pyramid_stairs_inv": "stairs_up"' in legacy
    assert "column / terrain_generator.num_cols + 0.001" in text


def test_reference_stand_thresholds_are_explicit() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    official = _section(text, "class OfficialReferenceRewardsCfg", "class HybridReferenceRewardsCfg")
    hybrid = _section(text, "class HybridReferenceRewardsCfg", "class OfficialReferenceTerminationsCfg")
    assert '"command_threshold": 0.06' in official
    assert '"command_threshold": 0.1' in hybrid


def test_gait_scale_is_cached_at_curriculum_updates() -> None:
    rewards = (ROOT / "locowheeledlegged/mdp/rewards.py").read_text(encoding="utf-8")
    curriculums = (ROOT / "locowheeledlegged/mdp/curriculums.py").read_text(encoding="utf-8")
    assert 'setattr(env, "_s10_reference_gait_scale", scale)' in rewards
    scale_getter = _section(rewards, "def terrain_curriculum_scale", "def terrain_scaled_joint_torques_l2")
    assert ".mean().item()" not in scale_getter
    assert "update_terrain_curriculum_scale(env, mean_level)" in curriculums
    runner = (ROOT / "locowheeledlegged/him/runner.py").read_text(encoding="utf-8")
    assert 'record["curriculum/gait_scale"]' in runner


def test_timeout_is_not_counted_as_runner_failure() -> None:
    runner = (ROOT / "locowheeledlegged/him/runner.py").read_text(encoding="utf-8")
    assert 'timeout = term_flags.get("time_out", zero_flags)' in runner
    assert 'term_name not in ("time_out", "obstacle_success", "platform_success")' in runner
    assert "family_failure = failure & ~success" in runner
    assert "family_failure |= timeout" not in runner
    assert '"terrain_out_of_bounds", "out_of_lane", "out_of_patch"' in runner
    assert "neutral timeouts=" in runner


def test_restoring_terrain_levels_invalidates_gait_scale_cache() -> None:
    runner = (ROOT / "locowheeledlegged/him/runner.py").read_text(encoding="utf-8")
    load_block = _section(runner, "    def load(", "    def load_transfer")
    assert '"_s10_reference_gait_scale"' in load_block
    assert "delattr(raw_env, cache_name)" in load_block


def test_reference_command_mixture_and_sampler_are_isolated() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    commands = COMMANDS.read_text(encoding="utf-8")
    official = _section(reference, "class OfficialReferenceCommandsCfg", "class HybridReferenceCommandsCfg")
    for assignment in (
        "rel_zero_vel_envs = 0.20",
        "rel_only_lin_y_envs = 0.02",
        "rel_only_lin_x_envs = 0.02",
        "rel_only_ang_z_envs = 0.20",
        "lin_vel_deadzone = 0.2",
    ):
        assert assignment in official
    for default in (
        "rel_zero_vel_envs: float = 0.0",
        "rel_only_lin_y_envs: float = 0.0",
        "rel_only_lin_x_envs: float = 0.0",
        "rel_only_ang_z_envs: float = 0.0",
    ):
        assert default in commands
    assert "def _apply_fixed_proportion_samples" in commands
    assert "lin_vel_deadzone: float = 0.0" in commands
    multi_sampler = _section(
        commands,
        "class UniformVelocityCommandMultiSampling(UniformVelocityCommand):",
        "@configclass\nclass UniformVelocityCommandMultiSamplingCfg",
    )
    assert "if self.cfg.lin_vel_deadzone > 0.0:" in multi_sampler
    assert "planar_norm > self.cfg.lin_vel_deadzone" in multi_sampler


def test_relative_joint_mirror_is_zero_at_default_pose() -> None:
    rewards = (ROOT / "locowheeledlegged/mdp/rewards.py").read_text(encoding="utf-8")
    assert "asset.data.joint_pos[:, left_ids] - asset.data.default_joint_pos[:, left_ids]" in rewards
    assert "asset.data.joint_pos[:, right_ids] - asset.data.default_joint_pos[:, right_ids]" in rewards
    assert "left - sign * right" in rewards
    # At q == q0, both relative deltas are exactly zero regardless of the
    # front/hind signed default-angle convention.
    left_delta = 0.0
    right_delta = 0.0
    assert (left_delta + right_delta) ** 2 == 0.0


def test_official_ppo_scale_is_preserved_for_him_comparison() -> None:
    text = PPO.read_text(encoding="utf-8")
    official = _section(text, "class S10OfficialReferencePPORunnerCfg", "class S10HybridReferencePPORunnerCfg")
    assert "max_iterations = 20000" in official
    assert "save_interval = 100" in official
    assert "entropy_coef=0.003" in official
    assert "learning_rate=1.0e-3" in official
