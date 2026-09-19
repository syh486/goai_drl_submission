from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .vq_lidar_common import LidarVqViewBackbone, VanillaVectorQuantizer


@dataclass(frozen=True)
class Model2VqVaeConfig:
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
    output_abs_max_m: float = 10.0
    codebook_size: int = 512
    commitment_cost: float = 0.25

    @property
    def horizontal_pad(self) -> int:
        pad = self.padded_width - self.raw_width
        if pad < 0 or pad % 2 != 0:
            raise ValueError(
                f"Expected an even non-negative width pad, got padded_width={self.padded_width}, raw_width={self.raw_width}."
            )
        return pad // 2


class Model2VqVaeBackbone(LidarVqViewBackbone):
    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        metric_reconstruction = float(self.config.output_abs_max_m) * torch.tanh(reconstruction_logits)
        return metric_reconstruction, metric_reconstruction


class Model2VqVaeView(nn.Module):
    def __init__(self, config: Model2VqVaeConfig | None = None):
        super().__init__()
        self.config = config or Model2VqVaeConfig()
        self.backbone = Model2VqVaeBackbone(self.config)
        self.quantizer = VanillaVectorQuantizer(
            codebook_size=self.config.codebook_size,
            embedding_dim=self.config.lidar_latent_channels,
            commitment_cost=self.config.commitment_cost,
        )

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.backbone.encode_backbone(noisy_dz)
        quantized = self.quantizer(encoded["latent_features"])
        return {
            **encoded,
            "lidar_latent": quantized["quantized"],
            "pre_quant_latent": encoded["latent_features"],
            "quantized_lookup": quantized["quantized_lookup"],
            "encoding_indices": quantized["encoding_indices"],
            "vq_loss": quantized["vq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "perplexity": quantized["perplexity"],
        }

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone.decode(lidar_latent)


class Model2VqVae(nn.Module):
    def __init__(
        self,
        config: Model2VqVaeConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__()
        self.config = config or Model2VqVaeConfig()
        self.share_view_weights = share_view_weights
        self.front_model = Model2VqVaeView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model2VqVaeView(config=self.config)

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
                "front_codebook": front_output["codebook_loss"],
                "rear_codebook": rear_output["codebook_loss"],
                "front_commitment": front_output["commitment_loss"],
                "rear_commitment": rear_output["commitment_loss"],
                "front_perplexity": front_output["perplexity"],
                "rear_perplexity": rear_output["perplexity"],
            },
        }
