from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Model1VqVaeConfig:
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

    @property
    def horizontal_pad(self) -> int:
        pad = self.padded_width - self.raw_width
        if pad < 0 or pad % 2 != 0:
            raise ValueError(
                f"Expected an even non-negative width pad, got padded_width={self.padded_width}, raw_width={self.raw_width}."
            )
        return pad // 2


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.dim() != 4:
            raise ValueError(f"Expected a 4D latent tensor, got shape {tuple(latents.shape)}")
        b, c, h, w = latents.shape
        flat = latents.permute(0, 2, 3, 1).reshape(-1, c)
        codebook = self.embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(quantized.detach(), latents)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized_st = latents + (quantized - latents).detach()
        encodings = F.one_hot(encoding_indices, self.codebook_size).to(dtype=latents.dtype)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1.0e-10))).sum())

        return {
            "quantized": quantized_st,
            "quantized_lookup": quantized,
            "encoding_indices": encoding_indices.view(b, h, w),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class Model1VqVaeView(nn.Module):
    def __init__(self, config: Model1VqVaeConfig | None = None):
        super().__init__()
        self.config = config or Model1VqVaeConfig()

        w1, w2, w3, w4 = self.config.encoder_widths
        self.encoder = nn.Sequential(
            _conv_block(self.config.input_channels_per_view, w1, stride=2),
            _conv_block(w1, w2, stride=2),
            _conv_block(w2, w3, stride=2),
            _conv_block(w3, w4, stride=1),
            _conv_block(w4, w4, stride=1),
        )
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _conv_block(w4, w4, stride=1),
            nn.Conv2d(w4, self.config.lidar_latent_channels, kernel_size=1, stride=1, padding=0),
        )
        self.quantizer = VectorQuantizer(
            codebook_size=self.config.codebook_size,
            embedding_dim=self.config.lidar_latent_channels,
            commitment_cost=self.config.commitment_cost,
        )
        self.from_latent = nn.Sequential(
            nn.Conv2d(self.config.lidar_latent_channels, w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(w4),
            nn.SiLU(inplace=True),
        )
        self.decode_block_12 = _conv_block(w4, w4, stride=1)
        self.decode_block_24 = _conv_block(w4, w3, stride=1)
        self.decode_block_48 = _conv_block(w3, w2, stride=1)
        self.decode_block_96 = _conv_block(w2, w1, stride=1)
        self.decode_head = nn.Sequential(
            _conv_block(w1, w1, stride=1),
            nn.Conv2d(w1, 1, kernel_size=3, stride=1, padding=1),
        )

    def _validate_view_input(self, noisy_dz: torch.Tensor) -> None:
        if noisy_dz.dim() != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got shape {tuple(noisy_dz.shape)}")
        if noisy_dz.shape[1] != self.config.input_channels_per_view:
            raise ValueError(
                f"Expected {self.config.input_channels_per_view} input channels per LiDAR view, got {noisy_dz.shape[1]}"
            )
        if noisy_dz.shape[2] != self.config.raw_height:
            raise ValueError(f"Expected raw height {self.config.raw_height}, got {noisy_dz.shape[2]}")
        if noisy_dz.shape[3] not in (self.config.raw_width, self.config.padded_width):
            raise ValueError(
                f"Expected width {self.config.raw_width} or {self.config.padded_width}, got {noisy_dz.shape[3]}"
            )

    def pad_input(self, noisy_dz: torch.Tensor) -> torch.Tensor:
        self._validate_view_input(noisy_dz)
        if noisy_dz.shape[-1] == self.config.padded_width:
            return noisy_dz
        pad = self.config.horizontal_pad
        return F.pad(noisy_dz, (pad, pad, 0, 0), mode="circular")

    def crop_output(self, padded_reconstruction: torch.Tensor) -> torch.Tensor:
        pad = self.config.horizontal_pad
        if pad == 0:
            return padded_reconstruction
        return padded_reconstruction[..., pad:-pad]

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        encoded = self.encoder(padded_input)
        latent_features = self.to_latent(encoded)
        quantized = self.quantizer(latent_features)
        return {
            "padded_input": padded_input,
            "encoder_features": encoded,
            "latent_features": latent_features,
            "lidar_latent": quantized["quantized"],
            "pre_quant_latent": latent_features,
            "quantized_lookup": quantized["quantized_lookup"],
            "encoding_indices": quantized["encoding_indices"],
            "vq_loss": quantized["vq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "perplexity": quantized["perplexity"],
        }

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        if lidar_latent.shape[1] != self.config.lidar_latent_channels:
            raise ValueError(
                f"Expected latent channels {self.config.lidar_latent_channels}, got {lidar_latent.shape[1]}"
            )
        expanded_latent = self.from_latent(lidar_latent)
        x = F.interpolate(expanded_latent, size=(12, 12), mode="bilinear", align_corners=False)
        x = self.decode_block_12(x)
        x = F.interpolate(x, size=(24, 24), mode="bilinear", align_corners=False)
        x = self.decode_block_24(x)
        x = F.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        x = self.decode_block_48(x)
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        x = self.decode_block_96(x)
        padded_reconstruction_logits = self.decode_head(x)
        raw_width_reconstruction_logits = self.crop_output(padded_reconstruction_logits)
        _, padded_reconstruction = self.project_metric_reconstruction(padded_reconstruction_logits)
        normalized_reconstruction, raw_width_reconstruction = self.project_metric_reconstruction(
            raw_width_reconstruction_logits
        )
        return {
            "expanded_latent": expanded_latent,
            "padded_reconstruction_logits": padded_reconstruction_logits,
            "padded_reconstruction": padded_reconstruction,
            "reconstruction_logits": raw_width_reconstruction_logits,
            "normalized_reconstruction": normalized_reconstruction,
            "reconstruction": raw_width_reconstruction,
        }

    def forward(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(noisy_dz)
        decoded = self.decode(encoded["lidar_latent"])
        return {**encoded, **decoded}


class Model1VqVae(nn.Module):
    def __init__(
        self,
        config: Model1VqVaeConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__()
        self.config = config or Model1VqVaeConfig()
        self.share_view_weights = share_view_weights

        self.front_model = Model1VqVaeView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1VqVaeView(config=self.config)

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
        return {
            "front": front_encoded,
            "rear": rear_encoded,
            "fused_latent": fused_latent,
        }

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
        front_output = self.front_model.decode(latent_outputs["front"]["lidar_latent"])
        rear_output = self.rear_model.decode(latent_outputs["rear"]["lidar_latent"])
        front_output = {**latent_outputs["front"], **front_output}
        rear_output = {**latent_outputs["rear"], **rear_output}
        fused_latent = latent_outputs["fused_latent"]

        return {
            "front": front_output,
            "rear": rear_output,
            "front_target": front_target,
            "rear_target": rear_target,
            "fused_latent": fused_latent,
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
