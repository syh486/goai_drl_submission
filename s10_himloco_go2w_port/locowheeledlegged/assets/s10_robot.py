"""S10 articulation configuration derived from the official controller assets."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from .s10.delayed_explicit import DelayedExplicitPDActuatorCfg
from .s10.delayed_implicit import DelayedImplicitActuatorCfg
from s10_policy_protocol import (
    DEFAULT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    POLICY_TO_ROBOT_INDICES,
    ROBOT_JOINT_NAMES,
    ROBOT_TO_POLICY_INDICES,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)


S10_ASSET_DIR = Path(__file__).resolve().parent / "s10"
S10_USD_PATH = S10_ASSET_DIR / "generated" / "s10.usd"

HIP_JOINT_NAMES = tuple(name for name in LEG_JOINT_NAMES if "hipx" in name)

# Default-pose clearance measured with the official S10 URDF/USD asset.
NOMINAL_BASE_HEIGHT = 0.423

if not S10_USD_PATH.is_file():
    raise FileNotFoundError(f"Missing generated S10 USD asset: {S10_USD_PATH}")


S10_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(S10_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.424),
        joint_pos=DEFAULT_JOINT_POSITIONS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=list(LEG_JOINT_NAMES),
            effort_limit_sim=50.0,
            velocity_limit_sim=25.76,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
            # Resolve by name inside the actuator.  IsaacLab's actuator order is
            # grouped by joint type, not the policy's per-leg order.
            reset_position_target=DEFAULT_JOINT_POSITIONS,
        ),
        "wheels": DelayedImplicitActuatorCfg(
            joint_names_expr=list(WHEEL_JOINT_NAMES),
            effort_limit_sim=14.0,
            velocity_limit_sim=65.5,
            stiffness=0.0,
            damping=0.8,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
    },
)

S10_PLAY_CFG = S10_CFG.copy()


def s10_explicit_pd_cfg() -> ArticulationCfg:
    """Return the S10 asset with deployment-style explicit torque control.

    Geometry, rigid-body physics, limits and gains are unchanged.  Only the
    actuator integration path changes from PhysX implicit drives to explicit
    clipped PD torques, matching IsaacGym and MuJoCo/real deployment semantics.
    """

    cfg = S10_CFG.copy()
    cfg.actuators = {
        "legs": DelayedExplicitPDActuatorCfg(
            joint_names_expr=list(LEG_JOINT_NAMES),
            effort_limit=50.0,
            effort_limit_sim=50.0,
            velocity_limit=25.76,
            velocity_limit_sim=25.76,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
            reset_position_target=DEFAULT_JOINT_POSITIONS,
        ),
        "wheels": DelayedExplicitPDActuatorCfg(
            joint_names_expr=list(WHEEL_JOINT_NAMES),
            effort_limit=14.0,
            effort_limit_sim=14.0,
            velocity_limit=65.5,
            velocity_limit_sim=65.5,
            stiffness=0.0,
            damping=0.8,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
    }
    return cfg
