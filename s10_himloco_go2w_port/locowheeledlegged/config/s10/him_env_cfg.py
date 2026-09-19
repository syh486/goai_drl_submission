"""Parallel HIMLoco observation packaging for the strict S10 task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import locowheeledlegged.mdp as mdp
import locowheeledlegged.mdp.s10_observations as s10_observations

from .locomotion_env_cfg import (
    JOINT_NAMES,
    NOMINAL_BASE_HEIGHT,
    ClearanceConstraintRewardsCfg,
    LocomotionEnvCfg,
    PairGeometryConstraintRewardsCfg,
    TargetBandConstraintRewardsCfg,
    WHEEL_JOINT_NAMES,
)
from .reference_env_cfg import (
    HybridReferenceEnvCfg,
    HybridReferenceRewardsCfg,
    OfficialReferenceEnvCfg,
    OfficialReferenceRewardsCfg,
)


@configclass
class HIMObservationsCfg:
    """Same upstream terms, grouped as six complete frames for HIMLoco."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            scale=1.0,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObservationTermCfg(
            func=s10_observations.joint_pos_rel_without_wheel_policy_order,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
        )
        joint_vel = ObservationTermCfg(
            func=s10_observations.joint_vel_rel_policy_order,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
        )
        last_action = ObservationTermCfg(func=mdp.last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            scale=1.0,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, scale=1.0)
        joint_pos = ObservationTermCfg(
            func=s10_observations.joint_pos_rel_without_wheel_policy_order,
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
        )
        joint_vel = ObservationTermCfg(
            func=s10_observations.joint_vel_rel_policy_order,
            scale=0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
        )
        last_action = ObservationTermCfg(func=mdp.last_action, scale=1.0)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0)
        height_scan = ObservationTermCfg(
            func=mdp.custom_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": NOMINAL_BASE_HEIGHT},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HIMLocomotionAEnvCfg(LocomotionEnvCfg):
    observations: HIMObservationsCfg = HIMObservationsCfg()
    rewards: ClearanceConstraintRewardsCfg = ClearanceConstraintRewardsCfg()


@configclass
class HIMLocomotionBEnvCfg(LocomotionEnvCfg):
    observations: HIMObservationsCfg = HIMObservationsCfg()
    rewards: TargetBandConstraintRewardsCfg = TargetBandConstraintRewardsCfg()


@configclass
class HIMLocomotionCEnvCfg(LocomotionEnvCfg):
    observations: HIMObservationsCfg = HIMObservationsCfg()
    rewards: PairGeometryConstraintRewardsCfg = PairGeometryConstraintRewardsCfg()


@configclass
class HIMOfficialReferenceEnvCfg(OfficialReferenceEnvCfg):
    observations: HIMObservationsCfg = HIMObservationsCfg()
    rewards: OfficialReferenceRewardsCfg = OfficialReferenceRewardsCfg()


@configclass
class HIMHybridReferenceEnvCfg(HybridReferenceEnvCfg):
    observations: HIMObservationsCfg = HIMObservationsCfg()
    rewards: HybridReferenceRewardsCfg = HybridReferenceRewardsCfg()
