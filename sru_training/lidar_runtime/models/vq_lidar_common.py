from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class VanillaVectorQuantizer(nn.Module):
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
        quantized_lookup = self.embedding(encoding_indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized_lookup, latents.detach())
        commitment_loss = F.mse_loss(quantized_lookup.detach(), latents)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized_st = latents + (quantized_lookup - latents).detach()
        encodings = F.one_hot(encoding_indices, self.codebook_size).to(dtype=latents.dtype)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1.0e-10))).sum())

        return {
            "quantized": quantized_st,
            "quantized_lookup": quantized_lookup,
            "encoding_indices": encoding_indices.view(b, h, w),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class EmaVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float,
        dead_code_threshold: float,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.dead_code_threshold = float(dead_code_threshold)

        embedding = torch.empty(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(embedding, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embed_avg", embedding.clone())

    def _update_ema(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        cluster_size_batch = encodings.sum(dim=0)
        embed_sum_batch = encodings.t() @ flat

        self.cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum_batch, alpha=1.0 - self.decay)

        if self.dead_code_threshold > 0.0 and flat.shape[0] > 0:
            dead_mask = self.cluster_size < self.dead_code_threshold
            if torch.any(dead_mask):
                num_dead = int(dead_mask.sum().item())
                replace_indices = torch.randint(0, flat.shape[0], (num_dead,), device=flat.device)
                replacements = flat[replace_indices]
                self.embed_avg[dead_mask] = replacements
                self.cluster_size[dead_mask] = max(self.dead_code_threshold, 1.0)

        cluster_size = self.cluster_size.clone()
        total_count = cluster_size.sum()
        cluster_size = (cluster_size + self.epsilon) / (
            total_count + self.codebook_size * self.epsilon
        ) * total_count.clamp_min(1.0)
        normalized_embed = self.embed_avg / cluster_size.unsqueeze(1).clamp_min(self.epsilon)
        self.embedding.copy_(normalized_embed)

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.dim() != 4:
            raise ValueError(f"Expected a 4D latent tensor, got shape {tuple(latents.shape)}")
        b, c, h, w = latents.shape
        flat = latents.permute(0, 2, 3, 1).reshape(-1, c)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1).unsqueeze(0)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).to(dtype=flat.dtype)

        if self.training:
            self._update_ema(flat.detach(), encodings.detach())

        quantized_lookup = F.embedding(encoding_indices, self.embedding).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        commitment_loss = F.mse_loss(quantized_lookup.detach(), latents)
        vq_loss = self.commitment_cost * commitment_loss
        codebook_loss = latents.new_zeros(())
        quantized_st = latents + (quantized_lookup - latents).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1.0e-10))).sum())

        return {
            "quantized": quantized_st,
            "quantized_lookup": quantized_lookup,
            "encoding_indices": encoding_indices.view(b, h, w),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class LidarVqViewBackbone(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        w1, w2, w3, w4 = self.config.encoder_widths
        self.encoder = nn.Sequential(
            conv_block(self.config.input_channels_per_view, w1, stride=2),
            conv_block(w1, w2, stride=2),
            conv_block(w2, w3, stride=2),
            conv_block(w3, w4, stride=1),
            conv_block(w4, w4, stride=1),
        )
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            conv_block(w4, w4, stride=1),
            nn.Conv2d(w4, self.config.lidar_latent_channels, kernel_size=1, stride=1, padding=0),
        )
        self.from_latent = nn.Sequential(
            nn.Conv2d(self.config.lidar_latent_channels, w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(w4),
            nn.SiLU(inplace=True),
        )
        self.decode_block_12 = conv_block(w4, w4, stride=1)
        self.decode_block_24 = conv_block(w4, w3, stride=1)
        self.decode_block_48 = conv_block(w3, w2, stride=1)
        self.decode_block_96 = conv_block(w2, w1, stride=1)
        self.decode_head = nn.Sequential(
            conv_block(w1, w1, stride=1),
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

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        encoded = self.encoder(padded_input)
        latent_features = self.to_latent(encoded)
        return {
            "padded_input": padded_input,
            "encoder_features": encoded,
            "latent_features": latent_features,
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
