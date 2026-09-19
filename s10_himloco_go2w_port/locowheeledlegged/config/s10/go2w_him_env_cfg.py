"""Faithful IsaacLab port of TrackinBIT/HIMLoco-for-Go2W for S10.

The algorithmic reference is locked to commit
``011693738c61603c3f22f2bce755098dd36fa7eb``. Robot geometry, nominal pose,
actuator limits and gains come from the official S10 assets; the task itself
follows the Go2W implementation.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import locowheeledlegged.mdp as mdp
from locowheeledlegged.assets.s10_robot import (
    HIP_JOINT_NAMES,
    LEG_JOINT_NAMES,
    NOMINAL_BASE_HEIGHT,
    POLICY_JOINT_NAMES,
    S10_CFG,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)
from locowheeledlegged.mdp import s10_observations
from locowheeledlegged.terrains import HfGo2WRoughPyramidSlopeTerrainCfg


REFERENCE_REPOSITORY = "TrackinBIT/HIMLoco-for-Go2W"
REFERENCE_COMMIT = "011693738c61603c3f22f2bce755098dd36fa7eb"
BASE_LINK = "base_link"
PENALIZED_CONTACT_BODIES = [BASE_LINK, ".*_hipy", ".*_knee"]


def _robot_cfg() -> ArticulationCfg:
    cfg = S10_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    # Go2W samples a 0--3 physics-tick delay. Keep S10 gains/limits because
    # these are part of the actual robot protocol, not the task algorithm.
    cfg.actuators["legs"].min_delay = 0
    cfg.actuators["legs"].max_delay = 3
    cfg.actuators["wheels"].min_delay = 0
    cfg.actuators["wheels"].max_delay = 3
    return cfg


@configclass
class Go2WSceneCfg(InteractiveSceneCfg):
    replicate_physics = False
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=25.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                # The reference's 10% smooth-slope bucket is split evenly by sign.
                "smooth_slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0,
                ),
                "smooth_slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0,
                ),
                "rough_slope_down": HfGo2WRoughPyramidSlopeTerrainCfg(
                    proportion=0.10,
                    slope_range=(0.0, 0.4),
                    noise_range=(0.01, 0.08),
                    noise_step=0.005,
                    downsampled_scale=0.2,
                    platform_width=3.0,
                ),
                # IsaacLab's native mesh stairs implement the same 0.30 m
                # concentric-pyramid geometry without expensive height-field
                # cooking for every patch.
                "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.35,
                    step_height_range=(0.05, 0.23),
                    step_width=0.30,
                    platform_width=3.0,
                    border_width=0.0,
                ),
                "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.20,
                    step_height_range=(0.05, 0.23),
                    step_width=0.30,
                    platform_width=3.0,
                    border_width=0.0,
                ),
                "discrete": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.25,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(1.0, 2.0),
                    obstacle_height_range=(0.05, 0.25),
                    num_obstacles=20,
                    platform_width=3.0,
                ),
            },
            seed=1,
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = _robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class Go2WCommandsCfg:
    base_velocity = mdp.Go2WVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        high_speed_fraction=0.2,
        ranges=mdp.Go2WVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


def _proprio_term(cache_role: str) -> ObservationTermCfg:
    return ObservationTermCfg(
        func=s10_observations.go2w_proprio_reference,
        scale=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True
            ),
            "cache_role": cache_role,
            "add_noise": True,
        },
    )


@configclass
class Go2WObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        proprio = _proprio_term("producer")

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        # The reference critic reuses the noisy current proprioceptive frame.
        proprio = _proprio_term("consumer")
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0)
        disturbance = ObservationTermCfg(func=mdp.privileged_disturbance, scale=1.0)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan_reference,
            scale=5.0,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        wheel_contact_forces = ObservationTermCfg(
            func=mdp.normalized_wheel_contact_forces,
            scale=1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=list(WHEEL_BODY_NAMES), preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WActionsCfg:
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEG_JOINT_NAMES),
        # S10's official runner uses 0.125 for hip-x and 0.25 for the other
        # leg joints. This is a robot protocol adaptation, not a task change.
        scale={name: (0.125 if "hipx" in name else 0.25) for name in LEG_JOINT_NAMES},
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_JOINT_NAMES),
        scale=5.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class Go2WRewardsCfg:
    tracking_lin_vel = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    tracking_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5)
    base_height = RewardTermCfg(
        # IsaacGym height fields always return a finite sample.  IsaacLab's
        # ray caster returns +inf when a ray leaves the terrain mesh, so the
        # native reward can become -inf near patch edges.  This adapter keeps
        # the same clearance objective while averaging finite hits only.
        func=mdp.base_height_above_terrain_l2,
        weight=-10.0,
        params={
            "target_height": NOMINAL_BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    hip_default = RewardTermCfg(
        func=mdp.joint_deviation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HIP_JOINT_NAMES))},
    )
    stand_still = RewardTermCfg(
        func=mdp.stand_still_without_cmd,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES)),
        },
    )
    collision = RewardTermCfg(
        func=mdp.collision_count,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PENALIZED_CONTACT_BODIES),
        },
    )
    feet_stumble = RewardTermCfg(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=list(WHEEL_BODY_NAMES), preserve_order=True
            )
        },
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES))},
    )
    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES))},
    )
    dof_acc = RewardTermCfg(
        func=mdp.dof_acc_reference,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(POLICY_JOINT_NAMES), preserve_order=True
            ),
            "wheel_count": 4,
        },
    )
    run_still = RewardTermCfg(
        func=mdp.run_still,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES)),
        },
    )


@configclass
class Go2WEventsCfg:
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK]),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    randomize_material = EventTermCfg(
        func=mdp.randomize_go2w_friction,
        # The reference reassigns friction on every episode reset.
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.25, 1.25),
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_go2w_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "kp_range": (0.9, 1.1),
            "kd_range": (0.9, 1.1),
            "motor_strength_range": (0.9, 1.1),
        },
    )
    disturbance = EventTermCfg(
        func=mdp.pulsed_disturbance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "force_range": (-30.0, 30.0),
            "interval_steps": 8,
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK]),
        },
    )
    push_robot = EventTermCfg(
        func=mdp.overwrite_root_velocity_push,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        is_global_time=True,
        params={"max_vel_xy": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Go2WTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK]),
            "threshold": 1.0,
        },
    )


@configclass
class Go2WCurriculumCfg:
    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_reference)
    command_range = CurriculumTermCfg(
        func=mdp.command_range_reference,
        params={"reward_term_name": "tracking_lin_vel", "max_curriculum": 1.5},
    )


@configclass
class Go2WHIMEnvCfg(ManagerBasedRLEnvCfg):
    # Legacy reference behavior leaves this false.  Deployment-oriented
    # fine-tuning can enable the measured wheel-speed signal without changing
    # the 57-D observation shape.
    observe_wheel_velocity: bool = False
    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=4096, env_spacing=3.0)
    viewer = ViewerCfg(
        eye=(5.0, 5.0, 4.0), lookat=(0.0, 0.0, 0.0), origin_type="asset_root", asset_name="robot"
    )
    commands: Go2WCommandsCfg = Go2WCommandsCfg()
    observations: Go2WObservationsCfg = Go2WObservationsCfg()
    actions: Go2WActionsCfg = Go2WActionsCfg()
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()
    events: Go2WEventsCfg = Go2WEventsCfg()
    terminations: Go2WTerminationsCfg = Go2WTerminationsCfg()
    curriculum: Go2WCurriculumCfg = Go2WCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.scene.height_scanner.update_period = self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
