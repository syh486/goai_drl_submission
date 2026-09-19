"""Gym registrations for the strict S10 port of LocoWheeledLegged."""

import gymnasium as gym

from . import go2w_deployment_env_cfg, go2w_him_env_cfg, him_env_cfg, locomotion_env_cfg, reference_env_cfg
from .agents import rsl_rl_ppo_cfg


_PPO_TASKS = {
    "Isaac-LocomotionS10-A-v1": locomotion_env_cfg.LocomotionAEnvCfg,
    "Isaac-LocomotionS10-B-v1": locomotion_env_cfg.LocomotionBEnvCfg,
    "Isaac-LocomotionS10-C-v1": locomotion_env_cfg.LocomotionCEnvCfg,
    "Isaac-LocomotionS10-Play-v1": locomotion_env_cfg.LocomotionPlayEnvCfg,
    "Isaac-LocomotionS10-Official-v1": reference_env_cfg.OfficialReferenceEnvCfg,
    "Isaac-LocomotionS10-Hybrid-v1": reference_env_cfg.HybridReferenceEnvCfg,
}

_HIM_TASKS = {
    "Isaac-S10-Go2W-HIM-v1": go2w_him_env_cfg.Go2WHIMEnvCfg,
    "Isaac-S10-Go2W-Deployment-HIM-v1": go2w_deployment_env_cfg.Go2WDeploymentHIMEnvCfg,
    "Isaac-LocomotionS10-HIM-A-v1": him_env_cfg.HIMLocomotionAEnvCfg,
    "Isaac-LocomotionS10-HIM-B-v1": him_env_cfg.HIMLocomotionBEnvCfg,
    "Isaac-LocomotionS10-HIM-C-v1": him_env_cfg.HIMLocomotionCEnvCfg,
    "Isaac-LocomotionS10-HIM-Official-v1": him_env_cfg.HIMOfficialReferenceEnvCfg,
    "Isaac-LocomotionS10-HIM-Hybrid-v1": him_env_cfg.HIMHybridReferenceEnvCfg,
}

for task_id, env_cfg in {**_PPO_TASKS, **_HIM_TASKS}.items():
    if "Official" in task_id:
        runner_cfg = rsl_rl_ppo_cfg.S10OfficialReferencePPORunnerCfg
    elif "Hybrid" in task_id:
        runner_cfg = rsl_rl_ppo_cfg.S10HybridReferencePPORunnerCfg
    else:
        runner_cfg = rsl_rl_ppo_cfg.S10LocomotionPPORunnerCfg
    entry_point = (
        "locowheeledlegged.envs:Go2WPositiveRewardEnv"
        if task_id in ("Isaac-S10-Go2W-HIM-v1", "Isaac-S10-Go2W-Deployment-HIM-v1")
        else "isaaclab.envs:ManagerBasedRLEnv"
    )
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )
