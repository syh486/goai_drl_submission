from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork


DEFAULT_SRU_DEPTH_WEIGHTS = Path(__file__).resolve().parents[1] / "assets/vae_pretrain_new.pth"


@dataclass(frozen=True)
class Model1Config:
    raw_height: int = 96
    raw_width: int = 90
    padded_height: int = 96
    padded_width: int = 96
    sru_encoder_height: int = 40
    sru_encoder_width: int = 64
    input_channels_per_view: int = 2
    target_channels_per_view: int = 1
    pretrained_latent_channels: int = 64
    lidar_latent_channels: int = 32
    distance_scale: float = 10.0

    @property
    def horizontal_pad(self) -> int:
        pad = self.padded_width - self.raw_width
        if pad < 0 or pad % 2 != 0:
            raise ValueError(
                f"Expected an even non-negative width pad, got padded_width={self.padded_width}, raw_width={self.raw_width}."
            )
        return pad // 2


class Model1VaeSampler(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.conv = Conv2dNormActivation(input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.mean_layers(x)
        return x


class _Model1EncoderFpn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        encoder = regnet_x_400mf(weights=None)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        encoder[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channels)


class Model1DepthEncoder(_Model1EncoderFpn):
    def __init__(self, out_channels: int):
        super().__init__(in_channels=1, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = OrderedDict()
        x = self.enc(x)
        features["feat1"] = self.enc_1(x)
        features["feat2"] = self.enc_2(features["feat1"])
        features["feat3"] = self.enc_3(features["feat2"])
        fpn_out = self.fpn(features)
        return fpn_out["feat1"]


class Model1DepthDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = Conv2dNormActivation(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv(z)
        return self.decoder(z)


class Model1OriginalVaenet(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.depth_encoder = Model1DepthEncoder(latent_dim)
        self.vae_sampler = Model1VaeSampler(latent_dim, latent_dim)
        self.depth_decoder = Model1DepthDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_encoder(x)
        x = self.vae_sampler(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.depth_decoder(z)


class Model1View(nn.Module):
    def __init__(
        self,
        config: Model1Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__()
        self.config = config or Model1Config()

        self.input_adapter = nn.Sequential(
            Conv2dNormActivation(
                self.config.input_channels_per_view,
                8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        )

        self.sru_vae = Model1OriginalVaenet(latent_dim=self.config.pretrained_latent_channels)
        self.latent_down = nn.Conv2d(
            self.config.pretrained_latent_channels,
            self.config.lidar_latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.latent_up = nn.Conv2d(
            self.config.lidar_latent_channels,
            self.config.pretrained_latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._load_pretrained_weights(pretrained_weights)
        self._set_frozen_state(freeze_sru_encoder=freeze_sru_encoder, freeze_sru_sampler=freeze_sru_sampler)

    def _load_pretrained_weights(self, pretrained_weights: str | Path | None) -> None:
        if pretrained_weights is None:
            return
        weight_path = Path(pretrained_weights).expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(f"SRU depth VAE weights not found: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.sru_vae.load_state_dict(state_dict, strict=True)

    def _set_frozen_state(self, freeze_sru_encoder: bool, freeze_sru_sampler: bool) -> None:
        if freeze_sru_encoder:
            for param in self.sru_vae.depth_encoder.parameters():
                param.requires_grad = False
        if freeze_sru_sampler:
            for param in self.sru_vae.vae_sampler.parameters():
                param.requires_grad = False

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
        metric_reconstruction = normalized_reconstruction * self.config.distance_scale
        return normalized_reconstruction, metric_reconstruction

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        fused_depth = self.input_adapter(padded_input)
        sru_input = F.interpolate(
            fused_depth,
            size=(self.config.sru_encoder_height, self.config.sru_encoder_width),
            mode="bilinear",
            align_corners=False,
        )
        sru_latent = self.sru_vae(sru_input)
        lidar_latent = self.latent_down(sru_latent)
        return {
            "padded_input": padded_input,
            "fused_depth": fused_depth,
            "sru_input": sru_input,
            "sru_latent": sru_latent,
            "lidar_latent": lidar_latent,
        }

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        expanded_latent = self.latent_up(lidar_latent)
        lowres_reconstruction_logits = self.sru_vae.decode(expanded_latent)
        padded_reconstruction_logits = F.interpolate(
            lowres_reconstruction_logits,
            size=(self.config.padded_height, self.config.padded_width),
            mode="bilinear",
            align_corners=False,
        )
        raw_width_reconstruction_logits = self.crop_output(padded_reconstruction_logits)

        _, lowres_reconstruction = self.project_metric_reconstruction(lowres_reconstruction_logits)
        _, padded_reconstruction = self.project_metric_reconstruction(padded_reconstruction_logits)
        normalized_reconstruction, raw_width_reconstruction = self.project_metric_reconstruction(
            raw_width_reconstruction_logits
        )
        return {
            "expanded_latent": expanded_latent,
            "lowres_reconstruction_logits": lowres_reconstruction_logits,
            "lowres_reconstruction": lowres_reconstruction,
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


class Model1(nn.Module):
    def __init__(
        self,
        config: Model1Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        share_view_weights: bool = True,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__()
        self.config = config or Model1Config()
        self.share_view_weights = share_view_weights

        self.front_model = Model1View(
            config=self.config,
            pretrained_weights=pretrained_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1View(
                config=self.config,
                pretrained_weights=pretrained_weights,
                freeze_sru_encoder=freeze_sru_encoder,
                freeze_sru_sampler=freeze_sru_sampler,
            )

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
        }
