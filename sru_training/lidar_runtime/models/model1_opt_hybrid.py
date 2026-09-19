from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_lidar_common import EmaVectorQuantizer, LidarVqViewBackbone


@dataclass(frozen=True)
class Model1OptHybridConfig:
    raw_height: int = 96
    raw_width: int = 90
    padded_height: int = 96
    padded_width: int = 96
    input_channels_per_view: int = 2
    target_channels_per_view: int = 1
    lidar_latent_channels: int = 32
    latent_height: int = 5
    latent_width: int = 8
    encoder_widths: tuple[int, int, int, int] = (32, 64, 96, 128)
    distance_scale: float = 10.0
    codebook_size: int = 512
    commitment_cost: float = 0.25
    quantized_channels: int = 24
    continuous_channels: int = 8
    continuous_budget: float = 0.25
    continuous_penalty_weight: float = 1.0e-2
    ema_decay: float = 0.99
    ema_epsilon: float = 1.0e-5
    dead_code_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.quantized_channels <= 0:
            raise ValueError("quantized_channels must be positive.")
        if self.continuous_channels < 0:
            raise ValueError("continuous_channels must be non-negative.")
        if self.quantized_channels + self.continuous_channels != self.lidar_latent_channels:
            raise ValueError(
                "quantized_channels + continuous_channels must match lidar_latent_channels, "
                f"got {self.quantized_channels} + {self.continuous_channels} != {self.lidar_latent_channels}."
            )
        if self.continuous_budget < 0.0:
            raise ValueError("continuous_budget must be non-negative.")
        if self.continuous_penalty_weight < 0.0:
            raise ValueError("continuous_penalty_weight must be non-negative.")

    @property
    def horizontal_pad(self) -> int:
        pad = self.padded_width - self.raw_width
        if pad < 0 or pad % 2 != 0:
            raise ValueError(
                f"Expected an even non-negative width pad, got padded_width={self.padded_width}, raw_width={self.raw_width}."
            )
        return pad // 2


class Model1OptHybridView(nn.Module):
    def __init__(self, config: Model1OptHybridConfig | None = None):
        super().__init__()
        self.config = config or Model1OptHybridConfig()
        self.backbone = LidarVqViewBackbone(self.config)
        self.quantizer = EmaVectorQuantizer(
            codebook_size=self.config.codebook_size,
            embedding_dim=self.config.quantized_channels,
            commitment_cost=self.config.commitment_cost,
            decay=self.config.ema_decay,
            epsilon=self.config.ema_epsilon,
            dead_code_threshold=self.config.dead_code_threshold,
        )

    def _split_latent(self, latent_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_features.shape[1] != self.config.lidar_latent_channels:
            raise ValueError(
                f"Expected latent channels {self.config.lidar_latent_channels}, got {latent_features.shape[1]}"
            )
        return torch.split(
            latent_features,
            [self.config.quantized_channels, self.config.continuous_channels],
            dim=1,
        )

    def _continuous_penalty(self, continuous_latent: torch.Tensor) -> torch.Tensor:
        if self.config.continuous_channels == 0:
            return continuous_latent.new_zeros(())
        overflow = F.relu(continuous_latent.abs() - float(self.config.continuous_budget))
        return overflow.square().mean()

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.backbone.encode_backbone(noisy_dz)
        latent_features = encoded["latent_features"]
        discrete_latent, continuous_latent = self._split_latent(latent_features)
        quantized = self.quantizer(discrete_latent)

        continuous_penalty_raw = self._continuous_penalty(continuous_latent)
        continuous_penalty = float(self.config.continuous_penalty_weight) * continuous_penalty_raw
        total_regularizer = quantized["vq_loss"] + continuous_penalty

        lidar_latent = torch.cat((quantized["quantized"], continuous_latent), dim=1)
        quantized_lookup = torch.cat((quantized["quantized_lookup"], continuous_latent), dim=1)

        return {
            **encoded,
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

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone.decode(lidar_latent)


class Model1OptHybrid(nn.Module):
    def __init__(
        self,
        config: Model1OptHybridConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__()
        self.config = config or Model1OptHybridConfig()
        self.share_view_weights = share_view_weights
        self.front_model = Model1OptHybridView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridView(config=self.config)

    @staticmethod
    def split_full_sample(lidar_sample: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_sample.dim() != 4 or lidar_sample.shape[1] != 6:
            raise ValueError(
                f"Expected a full LiDAR sample batch with shape (B, 6, 96, 90), got {tuple(lidar_sample.shape)}"
            )
        return {
            "front_target": lidar_sample[:, 0:1],
            "rear_target": lidar_sample[:, 1:2],
            "front_input": torch.stack((lidar_sample[:, 2], lidar_sample[:, 4]), dim=1),
            "rear_input": torch.stack((lidar_sample[:, 3], lidar_sample[:, 5]), dim=1),
        }

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
        fused_latent = torch.cat((front_encoded["lidar_latent"], rear_encoded["lidar_latent"]), dim=1)
        return {"front": front_encoded, "rear": rear_encoded, "fused_latent": fused_latent}

    def forward(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
        front_target: torch.Tensor | None = None,
        rear_target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input = split["front_input"]
            rear_input = split["rear_input"]
            front_target = split["front_target"]
            rear_target = split["rear_target"]

        if front_input is None or rear_input is None:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")

        latent_outputs = self.extract_latents(front_input=front_input, rear_input=rear_input)
        front_output = {**latent_outputs["front"], **self.front_model.decode(latent_outputs["front"]["lidar_latent"])}
        rear_output = {**latent_outputs["rear"], **self.rear_model.decode(latent_outputs["rear"]["lidar_latent"])}
        return {
            "front": front_output,
            "rear": rear_output,
            "front_target": front_target,
            "rear_target": rear_target,
            "fused_latent": latent_outputs["fused_latent"],
            "aux_losses": {
                "vq": front_output["vq_loss"] + rear_output["vq_loss"],
                "front_vq": front_output["vq_loss"],
                "rear_vq": rear_output["vq_loss"],
                "front_discrete_vq": front_output["vq_loss_discrete"],
                "rear_discrete_vq": rear_output["vq_loss_discrete"],
                "front_codebook": front_output["codebook_loss"],
                "rear_codebook": rear_output["codebook_loss"],
                "front_commitment": front_output["commitment_loss"],
                "rear_commitment": rear_output["commitment_loss"],
                "front_continuous_penalty": front_output["continuous_penalty"],
                "rear_continuous_penalty": rear_output["continuous_penalty"],
                "front_continuous_penalty_raw": front_output["continuous_penalty_raw"],
                "rear_continuous_penalty_raw": rear_output["continuous_penalty_raw"],
                "front_perplexity": front_output["perplexity"],
                "rear_perplexity": rear_output["perplexity"],
            },
        }
