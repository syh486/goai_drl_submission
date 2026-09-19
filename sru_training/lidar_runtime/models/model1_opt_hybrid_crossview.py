from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView


@dataclass(frozen=True)
class Model1OptHybridCrossViewConfig(Model1OptHybridConfig):
    quantized_channels: int = 8
    continuous_channels: int = 24
    continuous_budget: float = 0.15
    dead_code_threshold: float = 0.1
    crossview_interaction_width_scale_num: int = 1
    crossview_interaction_width_scale_den: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.crossview_interaction_width_scale_num <= 0 or self.crossview_interaction_width_scale_den <= 0:
            raise ValueError("crossview interaction width scale numerator/denominator must be positive.")


class Model1OptHybridCrossViewView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridCrossViewConfig | None = None):
        super().__init__(config=config or Model1OptHybridCrossViewConfig())
        self.config = config or Model1OptHybridCrossViewConfig()

    def encode_backbone_only(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone.encode_backbone(noisy_dz)

    def encode_from_latent_features(
        self,
        latent_features: torch.Tensor,
        backbone_encoded: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        discrete_latent, continuous_latent = self._split_latent(latent_features)
        quantized = self.quantizer(discrete_latent)

        continuous_penalty_raw = self._continuous_penalty(continuous_latent)
        continuous_penalty = float(self.config.continuous_penalty_weight) * continuous_penalty_raw
        total_regularizer = quantized["vq_loss"] + continuous_penalty

        lidar_latent = torch.cat((quantized["quantized"], continuous_latent), dim=1)
        quantized_lookup = torch.cat((quantized["quantized_lookup"], continuous_latent), dim=1)

        return {
            **backbone_encoded,
            "latent_features": latent_features,
            "lidar_latent": lidar_latent,
            "pre_quant_latent": latent_features,
            "pre_quant_latent_discrete": discrete_latent,
            "pre_quant_latent_continuous": continuous_latent,
            "quantized_lookup": quantized_lookup,
            "quantized_lookup_discrete": quantized["quantized_lookup"],
            "encoding_indices": quantized["encoding_indices"],
            "vq_loss": total_regularizer,
            "vq_loss_discrete": quantized["vq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "continuous_penalty": continuous_penalty,
            "continuous_penalty_raw": continuous_penalty_raw,
            "perplexity": quantized["perplexity"],
        }


class _CrossViewMixer(nn.Module):
    def __init__(self, latent_channels: int, inter_width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * latent_channels, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, latent_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(latent_channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, self_features: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat((self_features, other_features), dim=1))
        return self.act(self_features + delta)


class Model1OptHybridCrossView(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridCrossViewConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridCrossViewConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridCrossViewView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridCrossViewView(config=self.config)

        inter_width = max(
            self.config.lidar_latent_channels,
            int(
                round(
                    self.config.lidar_latent_channels
                    * self.config.crossview_interaction_width_scale_num
                    / self.config.crossview_interaction_width_scale_den
                )
            ),
        )
        self.crossview_mixer = _CrossViewMixer(self.config.lidar_latent_channels, inter_width)

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

        front_backbone = self.front_model.encode_backbone_only(front_input)
        rear_backbone = self.rear_model.encode_backbone_only(rear_input)
        front_latent_features = self.crossview_mixer(front_backbone["latent_features"], rear_backbone["latent_features"])
        rear_latent_features = self.crossview_mixer(rear_backbone["latent_features"], front_backbone["latent_features"])

        front_encoded = self.front_model.encode_from_latent_features(front_latent_features, front_backbone)
        rear_encoded = self.rear_model.encode_from_latent_features(rear_latent_features, rear_backbone)
        fused_latent = torch.cat((front_encoded["lidar_latent"], rear_encoded["lidar_latent"]), dim=1)
        return {"front": front_encoded, "rear": rear_encoded, "fused_latent": fused_latent}
