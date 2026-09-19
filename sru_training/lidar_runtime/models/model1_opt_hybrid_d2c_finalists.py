from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .model1_opt_hybrid import Model1OptHybrid
from .model1_opt_hybrid_crossview import _CrossViewMixer
from .model1_opt_hybrid_d2c_variants import (
    LidarVqD2CGeoLateBackbone,
    Model1OptHybridD2CAdditiveResGateView,
    Model1OptHybridD2CGeoLateConfig,
    Model1OptHybridD2CGeoLateView,
)


def _mix_continuous_latents(
    front_encoded: dict[str, torch.Tensor],
    rear_encoded: dict[str, torch.Tensor],
    *,
    quantized_channels: int,
    mixer: _CrossViewMixer,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    front_discrete = front_encoded["lidar_latent"][:, :quantized_channels]
    rear_discrete = rear_encoded["lidar_latent"][:, :quantized_channels]
    front_cont = front_encoded["lidar_latent"][:, quantized_channels:]
    rear_cont = rear_encoded["lidar_latent"][:, quantized_channels:]

    mixed_front_cont = mixer(front_cont, rear_cont)
    mixed_rear_cont = mixer(rear_cont, front_cont)

    front_encoded["lidar_latent"] = torch.cat((front_discrete, mixed_front_cont), dim=1)
    rear_encoded["lidar_latent"] = torch.cat((rear_discrete, mixed_rear_cont), dim=1)
    front_encoded["pre_quant_latent_continuous"] = mixed_front_cont
    rear_encoded["pre_quant_latent_continuous"] = mixed_rear_cont
    if "quantized_lookup_discrete" in front_encoded:
        front_encoded["quantized_lookup"] = torch.cat((front_encoded["quantized_lookup_discrete"], mixed_front_cont), dim=1)
    if "quantized_lookup_discrete" in rear_encoded:
        rear_encoded["quantized_lookup"] = torch.cat((rear_encoded["quantized_lookup_discrete"], mixed_rear_cont), dim=1)
    return front_encoded, rear_encoded


@dataclass(frozen=True)
class Model1OptHybridD2CGeoLateAdditiveResGateConfig(Model1OptHybridD2CGeoLateConfig):
    residual_abs_max_m: float = 3.0
    residual_gate_bias_init: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.residual_gate_bias_init < 1.0):
            raise ValueError("residual_gate_bias_init must be in (0, 1).")


class Model1OptHybridD2CGeoLateAdditiveResGateView(Model1OptHybridD2CAdditiveResGateView):
    def __init__(self, config: Model1OptHybridD2CGeoLateAdditiveResGateConfig | None = None):
        resolved_config = config or Model1OptHybridD2CGeoLateAdditiveResGateConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CGeoLateBackbone(self.config)


class Model1OptHybridD2CGeoLateAdditiveResGate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoLateAdditiveResGateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoLateAdditiveResGateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoLateAdditiveResGateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoLateAdditiveResGateView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CGeoLateCrossViewContConfig(Model1OptHybridD2CGeoLateConfig):
    crossview_interaction_width_scale_num: int = 1
    crossview_interaction_width_scale_den: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.crossview_interaction_width_scale_num <= 0 or self.crossview_interaction_width_scale_den <= 0:
            raise ValueError("crossview interaction width scale numerator/denominator must be positive.")


class Model1OptHybridD2CGeoLateCrossViewContView(Model1OptHybridD2CGeoLateView):
    def __init__(self, config: Model1OptHybridD2CGeoLateCrossViewContConfig | None = None):
        super().__init__(config=config or Model1OptHybridD2CGeoLateCrossViewContConfig())


class Model1OptHybridD2CGeoLateCrossViewCont(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoLateCrossViewContConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoLateCrossViewContConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoLateCrossViewContView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoLateCrossViewContView(config=self.config)

        inter_width = max(
            self.config.continuous_channels,
            int(
                round(
                    self.config.continuous_channels
                    * self.config.crossview_interaction_width_scale_num
                    / self.config.crossview_interaction_width_scale_den
                )
            ),
        )
        self.crossview_cont_mixer = _CrossViewMixer(self.config.continuous_channels, inter_width)

    def extract_latents(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input = split["front_input"]
            rear_input = split["rear_input"]
        if front_input is None or rear_input is None:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")

        front_encoded = self.front_model.encode(front_input)
        rear_encoded = self.rear_model.encode(rear_input)
        front_encoded, rear_encoded = _mix_continuous_latents(
            front_encoded,
            rear_encoded,
            quantized_channels=self.config.quantized_channels,
            mixer=self.crossview_cont_mixer,
        )
        fused_latent = torch.cat((front_encoded["lidar_latent"], rear_encoded["lidar_latent"]), dim=1)
        return {"front": front_encoded, "rear": rear_encoded, "fused_latent": fused_latent}


@dataclass(frozen=True)
class Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig(Model1OptHybridD2CGeoLateAdditiveResGateConfig):
    crossview_interaction_width_scale_num: int = 1
    crossview_interaction_width_scale_den: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.crossview_interaction_width_scale_num <= 0 or self.crossview_interaction_width_scale_den <= 0:
            raise ValueError("crossview interaction width scale numerator/denominator must be positive.")


class Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateView(Model1OptHybridD2CGeoLateAdditiveResGateView):
    def __init__(self, config: Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig | None = None):
        super().__init__(config=config or Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig())


class Model1OptHybridD2CGeoLateCrossViewContAdditiveResGate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoLateCrossViewContAdditiveResGateView(config=self.config)

        inter_width = max(
            self.config.continuous_channels,
            int(
                round(
                    self.config.continuous_channels
                    * self.config.crossview_interaction_width_scale_num
                    / self.config.crossview_interaction_width_scale_den
                )
            ),
        )
        self.crossview_cont_mixer = _CrossViewMixer(self.config.continuous_channels, inter_width)

    def extract_latents(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input = split["front_input"]
            rear_input = split["rear_input"]
        if front_input is None or rear_input is None:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")

        front_encoded = self.front_model.encode(front_input)
        rear_encoded = self.rear_model.encode(rear_input)
        front_encoded, rear_encoded = _mix_continuous_latents(
            front_encoded,
            rear_encoded,
            quantized_channels=self.config.quantized_channels,
            mixer=self.crossview_cont_mixer,
        )
        fused_latent = torch.cat((front_encoded["lidar_latent"], rear_encoded["lidar_latent"]), dim=1)
        return {"front": front_encoded, "rear": rear_encoded, "fused_latent": fused_latent}
