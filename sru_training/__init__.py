"""MuJoCo-side migration layer for the original SRU navigation stack."""

import sys
from pathlib import Path

# The package initializer runs before ``python -m sru_training.*`` executes
# the selected module. Put the copied training stack first before any adapter
# imports can resolve an unrelated installed ``rsl_rl`` package.
_LOCAL_ROOT = str(Path(__file__).resolve().parent)
if _LOCAL_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_ROOT)

from .s10_mujoco_env import (
    S10MujocoBackend,
    S10MujocoVecEnv,
    S10ObservationAdapter,
    S10RawState,
    policy_action_to_cmd_vel,
)
from .s10_policy_config import (
    S10ActionSpec,
    S10ObservationSpec,
    mdpo_config,
    ppo_config,
    ppo_mdpo_stage1_control_config,
)

__all__ = [
    "S10ActionSpec",
    "S10MujocoBackend",
    "S10MujocoVecEnv",
    "S10ObservationAdapter",
    "S10ObservationSpec",
    "S10RawState",
    "mdpo_config",
    "policy_action_to_cmd_vel",
    "ppo_config",
    "ppo_mdpo_stage1_control_config",
]
