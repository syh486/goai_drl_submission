from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model1_vae import Model1VaeConfig
from .model2_vae import SharedVaeModel, SharedVaeView, _conv_block


@dataclass(frozen=True)
class Model1VaeResizeConvConfig(Model1VaeConfig):
    pass


def _upsample_conv_block(
    in_channels: int,
    out_channels: int,
    *,
    output_size: tuple[int, int],
) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(size=output_size, mode="bilinear", align_corners=False),
        _conv_block(in_channels, out_channels, stride=1),
    )


class Model1VaeResizeConvView(SharedVaeView):
    def __init__(self, config: Model1VaeResizeConvConfig | None = None):
        resolved_config = config or Model1VaeResizeConvConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config

        w1, w2, w3, w4 = self.config.encoder_widths
        self.decode_up_10x16 = _upsample_conv_block(w4, w4, output_size=(10, 16))
        self.decode_up_20x32 = _upsample_conv_block(w4, w3, output_size=(20, 32))
        self.decode_up_40x64 = _upsample_conv_block(w3, w2, output_size=(40, 64))
        self.decode_up_80x96 = _upsample_conv_block(w2, w1, output_size=(80, 96))
        self.decode_up_96x96 = _upsample_conv_block(w1, w1, output_size=(96, 96))
        self.decode_head = nn.Sequential(
            _conv_block(w1, w1, stride=1),
            nn.Conv2d(w1, 1, kernel_size=3, stride=1, padding=1),
        )

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        if lidar_latent.shape[1] != self.config.lidar_latent_channels:
            raise ValueError(
                f"Expected latent channels {self.config.lidar_latent_channels}, got {lidar_latent.shape[1]}"
            )

        expanded_latent = self.from_latent(lidar_latent)
        x = self.decode_up_10x16(expanded_latent)
        x = self.decode_up_20x32(x)
        x = self.decode_up_40x64(x)
        x = self.decode_up_80x96(x)
        x = self.decode_up_96x96(x)
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


class Model1VaeResizeConv(SharedVaeModel):
    def __init__(
        self,
        config: Model1VaeResizeConvConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model1VaeResizeConvConfig(),
            view_cls=Model1VaeResizeConvView,
            share_view_weights=share_view_weights,
        )
