"""Official-M20 reference and conservative hybrid tasks for S10.

Both tasks retain S10's asset, joint order, action interface, low-pass action
terms, observations, command ranges, and domain randomization.  They isolate
the reward/terrain/termination choices that differ from Deep Robotics' M20
task, without changing the deployable 57-D proprioceptive protocol.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp.rewards as isaaclab_rewards
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import locowheeledlegged.mdp as mdp
import locowheeledlegged.mdp.rewards as custom_rewards
from locowheeledlegged.assets.s10_robot import NOMINAL_BASE_HEIGHT

from .locomotion_env_cfg import (
    BASE_LINK_NAME,
    CommandsCfg,
    FOOT_LINK_NAME,
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    LocomotionEnvCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


HIPX_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "hipx" in name]
HIPY_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "hipy" in name]
KNEE_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "knee" in name]
DIAGONAL_A_WHEELS = ("fl_wheel", "hr_wheel")
DIAGONAL_B_WHEELS = ("fr_wheel", "hl_wheel")
OFFICIAL_M20_REFERENCE_REPOSITORY = "DeepRoboticsLab/rl_training"
OFFICIAL_M20_REFERENCE_COMMIT = "6d317dfb33060226139e38600510e1751372eb5d"
# Lock every official source file used by this port.  This prevents a future
# checkout of the same repository from silently changing the reference recipe.
OFFICIAL_M20_SOURCE_SHA256 = {
    "rough_env_cfg.py": "38ff27cb3863bab9b0b0509c06082eb98b6b357a5335d780eb508026bb3d8fba",
    "rsl_rl_ppo_cfg.py": "c67d49772c0f841d8cb2ce5a67b62aa1d7598409d87269f6cbfe797958062bdc",
    "velocity_env_cfg.py": "295cb84a57bf9289c51a1ff2e9cf71fa41bf026a99f94422605ec0ae998e7bda",
    "rewards.py": "959d9a55751e57530a840e74f294f9fdf3582755bf51f42d10a3e97740f3d7b2",
    "commands.py": "31877425d07293cfdefd29858e105885a0edf8f7d75dc72d2200c29706017641",
    "curriculums.py": "c65a2a197c431a40b6772ffcd34c7ff203a47a5437033fbf46316897a324711d",
}
MIRROR_JOINTS = (
    ("fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"),
    ("fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"),
)


def _terrain_importer(sub_terrains: dict, *, max_init_terrain_level: int) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains=sub_terrains,
            seed=1,
        ),
        max_init_terrain_level=max_init_terrain_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


def _stairs(*, inverted: bool, proportion: float):
    terrain_type = (
        terrain_gen.MeshInvertedPyramidStairsTerrainCfg
        if inverted
        else terrain_gen.MeshPyramidStairsTerrainCfg
    )
    return terrain_type(
        proportion=proportion,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    )


@configclass
class OfficialReferenceSceneCfg(SceneCfg):
    """The official M20 terrain recipe, including its initial level cap of five."""

    terrain = _terrain_importer(
        {
            "pyramid_stairs": _stairs(inverted=False, proportion=0.2),
            "pyramid_stairs_inv": _stairs(inverted=True, proportion=0.2),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2,
                grid_width=0.45,
                grid_height_range=(0.025, 0.20),
                platform_width=2.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2,
                noise_range=(0.01, 0.16),
                noise_step=0.01,
                border_width=0.25,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
        },
        max_init_terrain_level=5,
    )


@configclass
class HybridReferenceSceneCfg(SceneCfg):
    """Competition-oriented mix: flat replaces slopes and Perlin is removed."""

    terrain = _terrain_importer(
        {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
            "pyramid_stairs_inv": _stairs(inverted=True, proportion=0.2),
            "pyramid_stairs": _stairs(inverted=False, proportion=0.2),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2,
                grid_width=0.45,
                grid_height_range=(0.025, 0.20),
                platform_width=2.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2,
                noise_range=(0.01, 0.10),
                noise_step=0.01,
                border_width=0.25,
            ),
        },
        max_init_terrain_level=2,
    )


def _joint_pose_term(weight: float, joint_names: list[str], *, relax_on_high_terrain: bool):
    params = {
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
        "stand_still_scale": 5.0,
        "velocity_threshold": 0.5,
        "command_threshold": 0.1,
    }
    if relax_on_high_terrain:
        params.update(
            sensor_cfg=SceneEntityCfg("height_scanner_base"),
            terrain_height_threshold=0.06,
            high_terrain_penalty_scale=0.1,
            ang_cmd_threshold=0.1,
            y_cmd_threshold=0.1,
            xy_norm_max=0.1,
            xz_norm_max=0.1,
        )
    return RewardTermCfg(
        func=custom_rewards.joint_pos_penalty_with_terrain_relaxation,
        weight=weight,
        params=params,
    )


def _yaw_air_time_term(weight: float) -> RewardTermCfg:
    return RewardTermCfg(
        func=custom_rewards.feet_air_time_yaw_with_clearance,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "threshold": 0.2,
            "command_threshold": 0.1,
            "clearance_threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_NAME]),
        },
    )


def _yaw_slide_term(weight: float) -> RewardTermCfg:
    return RewardTermCfg(
        func=custom_rewards.wheel_slide_yaw_command,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "linear_command_threshold": 0.1,
            "yaw_command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_NAME]),
        },
    )


def _rotation_gait_status_term(weight: float) -> RewardTermCfg:
    return RewardTermCfg(
        func=custom_rewards.rotation_gait_status,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot"),
            "group_a_body_names": DIAGONAL_A_WHEELS,
            "group_b_body_names": DIAGONAL_B_WHEELS,
            "target_height": 0.05,
            "linear_command_threshold": 0.5,
            "yaw_command_threshold": 0.05,
        },
    )


def _rotation_gait_symmetry_term(weight: float) -> RewardTermCfg:
    return RewardTermCfg(
        func=custom_rewards.RotationGaitSymmetry,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "group_a_body_names": DIAGONAL_A_WHEELS,
            "group_b_body_names": DIAGONAL_B_WHEELS,
            "target_duty": 0.5,
            "std": 0.2,
            "linear_command_threshold": 0.5,
            "yaw_command_threshold": 0.05,
            "window_s": 5.0,
        },
    )


@configclass
class OfficialReferenceCommandsCfg(CommandsCfg):
    """Current S10 command range with the official M20 special-case mixture."""

    def __post_init__(self):
        self.base_velocity.rel_standing_envs = 0.0
        self.base_velocity.initial_zero_command_steps = 0
        self.base_velocity.bang_bang_envs = 0.0
        self.base_velocity.lin_vel_deadzone = 0.2
        self.base_velocity.rel_zero_vel_envs = 0.20
        self.base_velocity.rel_only_lin_y_envs = 0.02
        self.base_velocity.rel_only_lin_x_envs = 0.02
        self.base_velocity.rel_only_ang_z_envs = 0.20


@configclass
class HybridReferenceCommandsCfg(CommandsCfg):
    """Midpoint between the existing S10 sampler and the official mixture."""

    def __post_init__(self):
        self.base_velocity.rel_standing_envs = 0.05
        self.base_velocity.initial_zero_command_steps = 25
        self.base_velocity.bang_bang_envs = 0.025
        self.base_velocity.lin_vel_deadzone = 0.1
        self.base_velocity.rel_zero_vel_envs = 0.10
        self.base_velocity.rel_only_lin_y_envs = 0.01
        self.base_velocity.rel_only_lin_x_envs = 0.01
        self.base_velocity.rel_only_ang_z_envs = 0.10


@configclass
class OfficialReferenceRewardsCfg(RewardsCfg):
    """Core M20 reward recipe adapted only where S10 conventions require it."""

    track_lin_vel_x_exp = None
    track_lin_vel_y_exp = None
    track_lin_vel_xy_exp = RewardTermCfg(
        func=isaaclab_rewards.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=isaaclab_rewards.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    lin_vel_z_l2 = RewardTermCfg(func=isaaclab_rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=isaaclab_rewards.ang_vel_xy_l2, weight=-0.02)
    flat_orientation_l2 = RewardTermCfg(func=isaaclab_rewards.flat_orientation_l2, weight=-50.0)
    base_height_l2 = None

    joint_torques_l2 = RewardTermCfg(
        func=custom_rewards.terrain_scaled_joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    leg_joint_acc_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    wheel_joint_acc_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    hip_deviation_l2 = None
    joint_deviation_l2 = None
    hipx_joint_pos_penalty = _joint_pose_term(-3.0, HIPX_JOINT_NAMES, relax_on_high_terrain=False)
    hipy_joint_pos_penalty = _joint_pose_term(-1.5, HIPY_JOINT_NAMES, relax_on_high_terrain=True)
    knee_joint_pos_penalty = _joint_pose_term(-0.75, KNEE_JOINT_NAMES, relax_on_high_terrain=True)
    stand_still_without_cmd = RewardTermCfg(
        func=custom_rewards.stand_still_without_cmd,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            # The function is commented out in the locked upstream commit,
            # but its source declares 0.06 as the intended M20 threshold.
            "command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_mirror = RewardTermCfg(
        func=custom_rewards.joint_mirror_relative_l2,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": MIRROR_JOINTS,
            "opposite_sign": True,
        },
    )
    action_rate_l2 = RewardTermCfg(func=custom_rewards.terrain_scaled_action_rate_l2, weight=-0.01)
    action_smooth_l2 = RewardTermCfg(func=custom_rewards.terrain_scaled_action_smooth_l2, weight=-0.025)
    undesired_contacts = RewardTermCfg(
        func=isaaclab_rewards.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 1.0,
        },
    )
    contact_forces = RewardTermCfg(
        func=custom_rewards.contact_forces_over_limit,
        weight=-1.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "threshold": 100.0,
        },
    )
    bad_orientation_penalty = RewardTermCfg(
        func=custom_rewards.bad_orientation_binary,
        weight=-1000.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_air_time_yaw = _yaw_air_time_term(50.0)
    wheel_slide_yaw = _yaw_slide_term(-2.0)
    rotation_gait_status = _rotation_gait_status_term(2.0)
    rotation_gait_symmetry = _rotation_gait_symmetry_term(15.0)


@configclass
class HybridReferenceRewardsCfg(OfficialReferenceRewardsCfg):
    """Conservative midpoint between the old S10 terms and official M20."""

    track_lin_vel_xy_exp = RewardTermCfg(
        func=isaaclab_rewards.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.35)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=isaaclab_rewards.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.35)},
    )
    flat_orientation_l2 = RewardTermCfg(func=isaaclab_rewards.flat_orientation_l2, weight=-10.0)
    base_height_l2 = RewardTermCfg(
        func=custom_rewards.base_height_above_terrain_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": NOMINAL_BASE_HEIGHT,
        },
    )
    joint_torques_l2 = RewardTermCfg(
        func=custom_rewards.terrain_scaled_joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    wheel_joint_acc_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_acc_l2,
        weight=-5.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    hipx_joint_pos_penalty = _joint_pose_term(-1.0, HIPX_JOINT_NAMES, relax_on_high_terrain=False)
    hipy_joint_pos_penalty = _joint_pose_term(-0.5, HIPY_JOINT_NAMES, relax_on_high_terrain=True)
    knee_joint_pos_penalty = _joint_pose_term(-0.25, KNEE_JOINT_NAMES, relax_on_high_terrain=True)
    stand_still_without_cmd = RewardTermCfg(
        func=custom_rewards.stand_still_without_cmd,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_mirror = RewardTermCfg(
        func=custom_rewards.joint_mirror_relative_l2,
        weight=-0.015,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": MIRROR_JOINTS,
            "opposite_sign": True,
        },
    )
    action_rate_l2 = RewardTermCfg(func=custom_rewards.terrain_scaled_action_rate_l2, weight=-0.0075)
    action_smooth_l2 = RewardTermCfg(func=custom_rewards.terrain_scaled_action_smooth_l2, weight=-0.01)
    contact_forces = RewardTermCfg(
        func=custom_rewards.contact_forces_over_limit,
        weight=-7.5e-5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "threshold": 100.0,
        },
    )
    bad_orientation_penalty = RewardTermCfg(
        func=custom_rewards.bad_orientation_binary,
        weight=-500.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_air_time_yaw = _yaw_air_time_term(25.0)
    wheel_slide_yaw = _yaw_slide_term(-1.0)
    rotation_gait_status = _rotation_gait_status_term(1.0)
    rotation_gait_symmetry = _rotation_gait_symmetry_term(7.5)


@configclass
class OfficialReferenceTerminationsCfg(TerminationsCfg):
    base_contact = None
    hip_contact = None
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation_components,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class HybridReferenceTerminationsCfg(TerminationsCfg):
    hip_contact = None
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation_components,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class OfficialReferenceEnvCfg(LocomotionEnvCfg):
    scene: OfficialReferenceSceneCfg = OfficialReferenceSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: OfficialReferenceCommandsCfg = OfficialReferenceCommandsCfg()
    rewards: OfficialReferenceRewardsCfg = OfficialReferenceRewardsCfg()
    terminations: OfficialReferenceTerminationsCfg = OfficialReferenceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The official M20 task trains the configured command range directly.
        self.curriculum.command_x_levels = None
        self.curriculum.command_y_levels = None
        self.curriculum.command_z_levels = None


@configclass
class HybridReferenceEnvCfg(LocomotionEnvCfg):
    scene: HybridReferenceSceneCfg = HybridReferenceSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: HybridReferenceCommandsCfg = HybridReferenceCommandsCfg()
    rewards: HybridReferenceRewardsCfg = HybridReferenceRewardsCfg()
    terminations: HybridReferenceTerminationsCfg = HybridReferenceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Hybrid uses the official combined XY tracking term.  The legacy S10
        # per-axis curricula reference removed reward names, while the official
        # M20 task trains the configured command range directly.
        self.curriculum.command_x_levels = None
        self.curriculum.command_y_levels = None
        self.curriculum.command_z_levels = None


OFFICIAL_TERRAIN_FAMILIES = {
    # Robots spawn at the center.  A regular pyramid is high at the center,
    # so moving away from spawn is downstairs/downhill.  The inverted form is
    # low at the center, so moving away is upstairs/uphill.
    "pyramid_stairs": "stairs_down",
    "pyramid_stairs_inv": "stairs_up",
    "boxes": "boxes",
    "random_rough": "random_rough",
    "hf_pyramid_slope": "ramp_down",
    "hf_pyramid_slope_inv": "ramp_up",
}

HYBRID_TERRAIN_FAMILIES = {
    "flat": "flat",
    "pyramid_stairs_inv": "stairs_up",
    "pyramid_stairs": "stairs_down",
    "boxes": "boxes",
    "random_rough": "random_rough",
}

LEGACY_TERRAIN_FAMILIES = {
    "flat": "flat",
    "boxes": "boxes",
    "perlin_rough": "perlin_rough",
    "random_rough": "random_rough",
    "pyramid_stairs": "stairs_down",
    "pyramid_stairs_inv": "stairs_up",
}


def terrain_family_columns(terrain_generator: TerrainGeneratorCfg, family_by_subterrain: dict[str, str]):
    """Return family column ranges using Isaac Lab's exact curriculum allocation rule.

    ``TerrainImporter.terrain_types`` stores atlas column indices, not indices
    into ``sub_terrains``.  Computing ranges from the active generator also
    keeps reduced-column smoke runs correctly labelled.
    """

    names = tuple(terrain_generator.sub_terrains)
    missing = set(names) - set(family_by_subterrain)
    if missing:
        raise ValueError(f"missing terrain family labels for: {sorted(missing)}")

    proportions = [float(terrain_generator.sub_terrains[name].proportion) for name in names]
    total = sum(proportions)
    if total <= 0.0:
        raise ValueError("terrain proportions must sum to a positive value")
    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    family_columns: dict[str, list[int]] = {}
    for column in range(terrain_generator.num_cols):
        sample = column / terrain_generator.num_cols + 0.001
        sub_index = next(
            (index for index, upper in enumerate(cumulative) if sample < upper),
            len(cumulative) - 1,
        )
        family = family_by_subterrain[names[sub_index]]
        family_columns.setdefault(family, []).append(column)

    ranges: dict[str, tuple[int, int]] = {}
    for family, columns in family_columns.items():
        start, stop = columns[0], columns[-1] + 1
        if columns != list(range(start, stop)):
            raise ValueError(f"terrain family {family!r} occupies non-contiguous columns: {columns}")
        ranges[family] = (start, stop)
    return ranges
