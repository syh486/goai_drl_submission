from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SharedUnetConfig:
    raw_height: int = 96
    raw_width: int = 90
    padded_height: int = 96
    padded_width: int = 96
    input_channels_per_view: int = 2
    target_channels_per_view: int = 1
    lidar_latent_channels: int = 32
    latent_height: int = 5
    latent_width: int = 8
    base_channels: int = 32

    @property
    def horizontal_pad(self) -> int:
        pad = self.padded_width - self.raw_width
        if pad < 0 or pad % 2 != 0:
            raise ValueError(
                f"Expected an even non-negative width pad, got padded_width={self.padded_width}, raw_width={self.raw_width}."
            )
        return pad // 2


@dataclass(frozen=True)
class Model2UnetConfig(SharedUnetConfig):
    output_abs_max_m: float = 10.0


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class SharedUnetView(nn.Module):
    def __init__(self, config: SharedUnetConfig):
        super().__init__()
        self.config = config
        c = self.config.base_channels

        self.enc1 = _double_conv(self.config.input_channels_per_view, c)
        self.enc2 = _double_conv(c, c * 2)
        self.enc3 = _double_conv(c * 2, c * 4)
        self.enc4 = _double_conv(c * 4, c * 4)

        self.to_latent = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            nn.Conv2d(c * 4, self.config.lidar_latent_channels, kernel_size=1),
        )

        self.from_latent = nn.Sequential(
            nn.Conv2d(self.config.lidar_latent_channels, c * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
        )

        self.dec3 = _double_conv(c * 4 + c * 4, c * 4)
        self.dec2 = _double_conv(c * 4 + c * 2, c * 2)
        self.dec1 = _double_conv(c * 2 + c, c)
        self.head = nn.Sequential(
            _double_conv(c, c),
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
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
        raise NotImplementedError

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        x1 = self.enc1(padded_input)
        x2 = self.enc2(F.avg_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.enc3(F.avg_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.enc4(F.avg_pool2d(x3, kernel_size=2, stride=2))
        lidar_latent = self.to_latent(x4)
        return {
            "padded_input": padded_input,
            "skip1": x1,
            "skip2": x2,
            "skip3": x3,
            "encoder_features": x4,
            "lidar_latent": lidar_latent,
        }

    def decode(
        self,
        lidar_latent: torch.Tensor,
        *,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        if lidar_latent.shape[1] != self.config.lidar_latent_channels:
            raise ValueError(
                f"Expected latent channels {self.config.lidar_latent_channels}, got {lidar_latent.shape[1]}"
            )

        x = self.from_latent(lidar_latent)
        x = F.interpolate(x, size=skip3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat((x, skip3), dim=1))
        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat((x, skip2), dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((x, skip1), dim=1))
        padded_reconstruction_logits = self.head(x)
        raw_width_reconstruction_logits = self.crop_output(padded_reconstruction_logits)
        _, padded_reconstruction = self.project_metric_reconstruction(padded_reconstruction_logits)
        normalized_reconstruction, raw_width_reconstruction = self.project_metric_reconstruction(
            raw_width_reconstruction_logits
        )
        return {
            "expanded_latent": x,
            "padded_reconstruction_logits": padded_reconstruction_logits,
            "padded_reconstruction": padded_reconstruction,
            "reconstruction_logits": raw_width_reconstruction_logits,
            "normalized_reconstruction": normalized_reconstruction,
            "reconstruction": raw_width_reconstruction,
        }

    def forward(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(noisy_dz)
        decoded = self.decode(
            encoded["lidar_latent"],
            skip1=encoded["skip1"],
            skip2=encoded["skip2"],
            skip3=encoded["skip3"],
        )
        return {**encoded, **decoded}


class SharedUnetModel(nn.Module):
    def __init__(
        self,
        *,
        config: SharedUnetConfig,
        view_cls: type[SharedUnetView],
        share_view_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.share_view_weights = share_view_weights

        self.front_model = view_cls(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = view_cls(config=self.config)

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
        front_output = self.front_model.decode(
            latent_outputs["front"]["lidar_latent"],
            skip1=latent_outputs["front"]["skip1"],
            skip2=latent_outputs["front"]["skip2"],
            skip3=latent_outputs["front"]["skip3"],
        )
        rear_output = self.rear_model.decode(
            latent_outputs["rear"]["lidar_latent"],
            skip1=latent_outputs["rear"]["skip1"],
            skip2=latent_outputs["rear"]["skip2"],
            skip3=latent_outputs["rear"]["skip3"],
        )
        front_output = {**latent_outputs["front"], **front_output}
        rear_output = {**latent_outputs["rear"], **rear_output}
        fused_latent = latent_outputs["fused_latent"]

        return {
            "front": front_output,
            "rear": rear_output,
            "front_target": front_target,
            "rear_target": rear_target,
            "fused_latent": fused_latent,
        }


class Model2UnetView(SharedUnetView):
    def __init__(self, config: Model2UnetConfig | None = None):
        super().__init__(config=config or Model2UnetConfig())
        self.config = config or Model2UnetConfig()

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        metric_reconstruction = float(self.config.output_abs_max_m) * torch.tanh(reconstruction_logits)
        return metric_reconstruction, metric_reconstruction


class Model2Unet(SharedUnetModel):
    def __init__(
        self,
        config: Model2UnetConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model2UnetConfig(),
            view_cls=Model2UnetView,
            share_view_weights=share_view_weights,
        )
