from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView
from .vq_lidar_common import conv_block


@dataclass(frozen=True)
class Model1OptHybridAdditiveConfig(Model1OptHybridConfig):
    quantized_channels: int = 8
    continuous_channels: int = 24
    continuous_budget: float = 0.15
    dead_code_threshold: float = 0.1
    residual_abs_max_m: float = 3.0


class _MetricDecoderBranch(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        encoder_widths: tuple[int, int, int, int],
        padded_height: int,
        padded_width: int,
        output_mode: str,
        output_scale: float,
    ):
        super().__init__()
        w1, w2, w3, w4 = encoder_widths
        self.padded_height = int(padded_height)
        self.padded_width = int(padded_width)
        self.output_mode = str(output_mode)
        self.output_scale = float(output_scale)

        self.from_latent = nn.Sequential(
            nn.Conv2d(input_channels, w4, kernel_size=1, stride=1, padding=0, bias=False),
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

    def project(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.output_mode == "sigmoid":
            normalized = torch.sigmoid(logits)
            metric = normalized * self.output_scale
            return normalized, metric
        if self.output_mode == "tanh":
            normalized = torch.tanh(logits)
            metric = normalized * self.output_scale
            return normalized, metric
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.from_latent(latent)
        x = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)
        x = self.decode_block_12(x)
        x = F.interpolate(x, size=(24, 24), mode="bilinear", align_corners=False)
        x = self.decode_block_24(x)
        x = F.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        x = self.decode_block_48(x)
        x = F.interpolate(x, size=(self.padded_height, self.padded_width), mode="bilinear", align_corners=False)
        x = self.decode_block_96(x)
        logits = self.decode_head(x)
        normalized, metric = self.project(logits)
        return {
            "expanded_latent": x,
            "padded_logits": logits,
            "padded_normalized": normalized,
            "padded_metric": metric,
        }


class Model1OptHybridAdditiveView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridAdditiveConfig | None = None):
        resolved_config = config or Model1OptHybridAdditiveConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.discrete_decoder = _MetricDecoderBranch(
            input_channels=self.config.quantized_channels,
            encoder_widths=self.config.encoder_widths,
            padded_height=self.config.padded_height,
            padded_width=self.config.padded_width,
            output_mode="sigmoid",
            output_scale=self.config.distance_scale,
        )
        self.continuous_decoder = _MetricDecoderBranch(
            input_channels=self.config.continuous_channels,
            encoder_widths=self.config.encoder_widths,
            padded_height=self.config.padded_height,
            padded_width=self.config.padded_width,
            output_mode="tanh",
            output_scale=self.config.residual_abs_max_m,
        )

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        discrete_latent, continuous_latent = self._split_latent(lidar_latent)
        discrete_out = self.discrete_decoder.decode(discrete_latent)
        continuous_out = self.continuous_decoder.decode(continuous_latent)

        padded_reconstruction = torch.clamp(
            discrete_out["padded_metric"] + continuous_out["padded_metric"],
            min=0.0,
            max=float(self.config.distance_scale),
        )
        raw_width_reconstruction = self.backbone.crop_output(padded_reconstruction)
        normalized_reconstruction = raw_width_reconstruction / float(self.config.distance_scale)

        return {
            "expanded_latent": lidar_latent,
            "padded_reconstruction_logits": padded_reconstruction,
            "padded_reconstruction": padded_reconstruction,
            "reconstruction_logits": raw_width_reconstruction,
            "normalized_reconstruction": normalized_reconstruction,
            "reconstruction": raw_width_reconstruction,
            "discrete_padded_reconstruction": discrete_out["padded_metric"],
            "continuous_padded_residual": continuous_out["padded_metric"],
            "discrete_reconstruction": self.backbone.crop_output(discrete_out["padded_metric"]),
            "continuous_residual": self.backbone.crop_output(continuous_out["padded_metric"]),
        }


class Model1OptHybridAdditive(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridAdditiveConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridAdditiveConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridAdditiveView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridAdditiveView(config=self.config)
