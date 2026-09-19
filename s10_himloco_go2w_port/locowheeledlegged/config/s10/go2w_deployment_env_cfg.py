"""Deployment-oriented fine-tuning variant of the strict Go2W S10 task.

This variant changes only two simulator-facing protocol choices whose legacy
implementation is incompatible with the official S10 deployment loop:

* observe measured wheel velocity in the existing four 57-D observation slots;
* apply explicit clipped PD torques instead of PhysX implicit joint drives.

Reward, terrain, command, curriculum, randomization and network dimensions are
inherited unchanged from the strict reference task.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from locowheeledlegged.assets.s10_robot import s10_explicit_pd_cfg

from .go2w_him_env_cfg import Go2WHIMEnvCfg, Go2WSceneCfg


def _deployment_robot_cfg() -> ArticulationCfg:
    cfg = s10_explicit_pd_cfg()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    for actuator in cfg.actuators.values():
        actuator.min_delay = 0
        actuator.max_delay = 3
    return cfg


@configclass
class Go2WDeploymentSceneCfg(Go2WSceneCfg):
    robot: ArticulationCfg = _deployment_robot_cfg()


@configclass
class Go2WDeploymentHIMEnvCfg(Go2WHIMEnvCfg):
    observe_wheel_velocity: bool = True
    scene: Go2WDeploymentSceneCfg = Go2WDeploymentSceneCfg(num_envs=4096, env_spacing=3.0)

    def __post_init__(self):
        super().__post_init__()
        # S10 uses twice the Go2W leg gains (80 vs 40 N m/rad).  A 5 ms
        # explicit torque loop is too coarse and is unstable even for the
        # mature checkpoint.  At 2 ms the same checkpoint is stable while the
        # 20 ms policy period remains unchanged.
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        # Preserve the reference's physical 0--15 ms action-latency support.
        # Eight 2 ms ticks gives a conservative 0--16 ms deployment envelope.
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 8
            actuator.resample_every_n_physics_steps = self.decimation
