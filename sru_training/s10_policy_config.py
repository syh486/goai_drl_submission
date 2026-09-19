"""SRU policy and runner configuration for the S10 MuJoCo bridge.

This module intentionally contains model/training-framework settings only.  It
does not define a navigation reward, terrain sampler, or episode generator.
Those are properties of the MuJoCo task and must be supplied by a backend.
"""

from __future__ import annotations

from dataclasses import dataclass


NEXT_WAYPOINT_OBS_DIM = 5


@dataclass(frozen=True)
class S10ObservationSpec:
    """Observation contract between MuJoCo and the migrated SRU policy.

    The order follows the old IsaacLab navigation policy where possible:
    proprioception/goal first, encoded LiDAR last. The critic is ordered as
    ``[proprioception, time, height latent, LiDAR latent]`` because the SRU
    network locates the three trailing blocks by positional slicing. It keeps
    the old asymmetric-observation shape by reserving a 64x7x7 height block.
    A backend may provide the privileged height feature; when it does not, the
    adapter fills that block with zeros and reports this fact in ``extras``.
    """

    latent_shape: tuple[int, int, int] = (64, 5, 8)
    height_shape: tuple[int, int, int] = (64, 7, 7)
    num_cameras: int = 1
    proprio_dim: int = 15
    goal_dim: int = 4
    next_waypoint_dim: int = 0

    @property
    def latent_dim(self) -> int:
        c, h, w = self.latent_shape
        return c * h * w * self.num_cameras

    @property
    def height_dim(self) -> int:
        c, h, w = self.height_shape
        return c * h * w

    @property
    def actor_obs_dim(self) -> int:
        return self.proprio_dim + self.next_waypoint_dim + self.latent_dim

    @property
    def critic_obs_dim(self) -> int:
        return (
            self.proprio_dim
            + self.next_waypoint_dim
            + self.height_dim
            + self.latent_dim
            + 1
        )

    def validate(self) -> None:
        if self.num_cameras not in (1, 2):
            raise ValueError(f"ActorCriticSRU supports one or two cameras, got {self.num_cameras}")
        if self.proprio_dim != 3 + 3 + 3 + 2 + self.goal_dim:
            raise ValueError("proprio_dim must match base_lin_vel + base_ang_vel + gravity + last_action + goal")
        if self.next_waypoint_dim not in (0, NEXT_WAYPOINT_OBS_DIM):
            raise ValueError(
                f"next_waypoint_dim must be 0 or {NEXT_WAYPOINT_OBS_DIM}"
            )


def observation_spec_for_route_context(enabled: bool) -> S10ObservationSpec:
    """Build the v4 or v5 route-observation contract."""

    return S10ObservationSpec(
        next_waypoint_dim=NEXT_WAYPOINT_OBS_DIM if enabled else 0
    )


def observation_spec_from_policy_state(state_dict: dict) -> S10ObservationSpec:
    """Infer the observation contract from an SRU checkpoint.

    The visual encoder contributes 64 features before the actor recurrent
    cell. The remaining recurrent inputs are proprioception and optional route
    context, so old and new checkpoints can coexist without metadata.
    """

    key = "memory_a.rnn.cells.0.transform_gate.weight"
    if key not in state_dict:
        raise KeyError(f"policy state has no {key}")
    recurrent_input = int(state_dict[key].shape[1])
    base_recurrent_input = S10ObservationSpec().proprio_dim + 64
    next_waypoint_dim = recurrent_input - base_recurrent_input
    spec = S10ObservationSpec(next_waypoint_dim=next_waypoint_dim)
    spec.validate()
    return spec


@dataclass(frozen=True)
class S10ActionSpec:
    """Mapping from SRU's two actions to the official ``cmd_vel`` contract."""

    # IsaacLab's original high-level action scale was [1.5, 1.0].
    policy_scale_vx: float = 1.5
    policy_scale_yaw: float = 1.0
    max_vx: float = 1.0
    max_vy: float = 0.0
    max_yaw: float = 1.0
    policy_hz: float = 5.0
    low_level_hz: float = 50.0
    mujoco_dt: float = 0.001

    @property
    def physics_steps_per_policy_step(self) -> int:
        steps = round(1.0 / (self.policy_hz * self.mujoco_dt))
        if steps <= 0:
            raise ValueError("policy_hz and mujoco_dt must produce a positive hold interval")
        return steps

    @property
    def low_level_decimation(self) -> int:
        """Physics steps between low-level policy evaluations.

        IsaacLab uses 200 Hz physics and a four-step low-level decimation.
        The official MuJoCo scene runs at 1 kHz, so physical-time equivalence
        requires 20 MuJoCo steps per 50 Hz ONNX evaluation.
        """

        steps = round(1.0 / (self.low_level_hz * self.mujoco_dt))
        if steps <= 0:
            raise ValueError("low_level_hz and mujoco_dt must produce a positive decimation")
        return steps


def ppo_config(*, smoke: bool = False) -> dict:
    """Return the MX PPO configuration accepted by OnPolicyRunner."""

    obs = S10ObservationSpec()
    obs.validate()
    if smoke:
        steps, epochs, minibatches = 4, 1, 2
        hidden = [64, 32, 16]
    else:
        # MXNavPPORunnerCfg uses 16 recurrent transitions per environment.
        # A 24-step rollout changes the recurrent minibatch and GAE horizon.
        steps, epochs, minibatches = 16, 5, 4
        hidden = [512, 256, 128]
    return {
        "seed": 42,
        "num_steps_per_env": steps,
        "save_interval": 500,
        "empirical_normalization": False,
        "logger": "tensorboard",
        # This is part of the upstream MX PPO runner protocol, not an MDPO
        # implementation detail.
        "reward_shifting_value": 0.05,
        "policy": {
            "class_name": "ActorCriticSRU",
            "init_noise_std": 1.0,
            "actor_hidden_dims": hidden,
            "critic_hidden_dims": hidden.copy(),
            "activation": "elu",
            "rnn_hidden_size": 512 if not smoke else 32,
            "rnn_type": "lstm_sru",
            "num_cameras": obs.num_cameras,
            "image_input_dims": obs.latent_shape,
            "height_input_dims": obs.height_shape,
            "dropout": 0.2,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 0.02,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "value_clip_param": 0.2,
            "entropy_coef": 0.00375,
            "num_learning_epochs": epochs,
            "num_mini_batches": minibatches,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.995,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def ppo_mdpo_stage1_control_config(*, smoke: bool = False) -> dict:
    """Return single-policy PPO with the successful MDPO stage-1 settings.

    The policy update remains PPO. Reward shifting, discounting, optimizer,
    learning-rate schedule, and rollout length match the local MDPO baseline so
    the controlled experiment primarily removes the second policy and mutual
    distillation.
    """

    cfg = ppo_config(smoke=smoke)
    cfg["num_steps_per_env"] = 4 if smoke else 16
    cfg["reward_shifting_value"] = 0.05
    cfg["algorithm"].update(
        {
            "gamma": 0.999,
            "schedule": "exponential",
            "use_muon": True,
            "min_learning_rate": 1.0e-7,
        }
    )
    return cfg


def mdpo_config(*, smoke: bool = False) -> dict:
    """Return the original MX MDPO configuration."""

    cfg = ppo_config(smoke=smoke)
    cfg["num_steps_per_env"] = 4 if smoke else 16
    cfg["reward_shifting_value"] = 0.05
    cfg["algorithm"].update(
        {
            "class_name": "MDPO",
            "value_loss_coef": 0.02,
            "entropy_coef": 0.00375,
            "schedule": "exponential",
            "gamma": 0.999,
        }
    )
    return cfg
