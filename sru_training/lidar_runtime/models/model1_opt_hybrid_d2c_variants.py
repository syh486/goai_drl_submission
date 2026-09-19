from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry_prior_utils import build_geometry_prior_channels

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridView
from .model1_opt_hybrid_additive import _MetricDecoderBranch
from .model1_opt_hybrid_crossview import _CrossViewMixer
from .model1_opt_hybrid_dual import _dual_conv_block
from .model1_opt_hybrid_dual_v3 import Model1OptHybridDualV3D2CConfig
from .vq_lidar_common import LidarVqViewBackbone


def _compute_branch_widths(config: Model1OptHybridDualV3D2CConfig) -> tuple[int, int, int, int, int, int]:
    w1, w2, w3, w4 = config.encoder_widths
    discrete_scale_num = int(config.branch_width_scale_num)
    discrete_scale_den = int(config.branch_width_scale_den)
    continuous_scale_num = int(config.continuous_branch_width_scale_num)
    continuous_scale_den = int(config.continuous_branch_width_scale_den)

    discrete_w2 = max(w1, int(round(w2 * discrete_scale_num / discrete_scale_den)))
    discrete_w3 = max(discrete_w2, int(round(w3 * discrete_scale_num / discrete_scale_den)))
    discrete_w4 = max(discrete_w3, int(round(w4 * discrete_scale_num / discrete_scale_den)))

    continuous_w2 = max(w1, int(round(w2 * continuous_scale_num / continuous_scale_den)))
    continuous_w3 = max(continuous_w2, int(round(w3 * continuous_scale_num / continuous_scale_den)))
    continuous_w4 = max(continuous_w3, int(round(w4 * continuous_scale_num / continuous_scale_den)))
    return discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4


@dataclass(frozen=True)
class Model1OptHybridD2CGeoContConfig(Model1OptHybridDualV3D2CConfig):
    geometry_prior_channels: int = 3


class LidarVqD2CGeoContBackbone(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridD2CGeoContConfig):
        super().__init__(config)
        self.config = config
        w1, _, _, _ = self.config.encoder_widths
        discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4 = _compute_branch_widths(self.config)

        priors = build_geometry_prior_channels(
            target_height=self.config.raw_height,
            target_width=self.config.raw_width,
        )
        self.register_buffer("geometry_priors", priors, persistent=False)

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )
        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, discrete_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(discrete_w2, discrete_w3, kernel_size=3, stride=1),
            _dual_conv_block(discrete_w3, discrete_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(discrete_w3, discrete_w4, kernel_size=3, stride=2),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1 + int(self.config.geometry_prior_channels), continuous_w2, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w2, continuous_w3, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w3, continuous_w4, kernel_size=3, stride=1),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
        )

        joint_width = discrete_w4 + continuous_w4
        inter_scale_num = int(self.config.interaction_width_scale_num)
        inter_scale_den = int(self.config.interaction_width_scale_den)
        inter_width = max(max(discrete_w4, continuous_w4), int(round(joint_width * inter_scale_num / inter_scale_den)))
        self.fuse_to_continuous = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
        )
        self.fuse_act = nn.SiLU(inplace=True)

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
            nn.Conv2d(discrete_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
            nn.Conv2d(continuous_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)

        priors = self.geometry_priors.to(device=padded_input.device, dtype=padded_input.dtype)
        priors = priors.unsqueeze(0).expand(padded_input.shape[0], -1, -1, -1)
        pad = self.config.horizontal_pad
        if pad > 0:
            priors = F.pad(priors, (pad, pad, 0, 0), mode="circular")
        priors = F.interpolate(priors, size=shared_features.shape[-2:], mode="bilinear", align_corners=False)

        discrete_features = self.discrete_branch(shared_features)
        continuous_input = torch.cat((shared_features, priors), dim=1)
        continuous_features = self.continuous_branch(continuous_input)

        joint_features = torch.cat((discrete_features, continuous_features), dim=1)
        continuous_features = self.fuse_act(continuous_features + self.fuse_to_continuous(joint_features))

        discrete_latent = self.to_discrete_latent(discrete_features)
        continuous_latent = self.to_continuous_latent(continuous_features)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)
        encoder_features = torch.cat((discrete_features, continuous_features), dim=1)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "continuous_geometry_priors": priors,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "encoder_features": encoder_features,
            "latent_features": latent_features,
        }


class Model1OptHybridD2CGeoContView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CGeoContConfig | None = None):
        resolved_config = config or Model1OptHybridD2CGeoContConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CGeoContBackbone(self.config)


class Model1OptHybridD2CGeoCont(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoContConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoContConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoContView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoContView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CGateConfig(Model1OptHybridDualV3D2CConfig):
    d2c_gate_init: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.d2c_gate_init < 1.0):
            raise ValueError("d2c_gate_init must be in (0, 1).")


class LidarVqD2CGateBackbone(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridD2CGateConfig):
        super().__init__(config)
        self.config = config

        w1, _, _, _ = self.config.encoder_widths
        discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4 = _compute_branch_widths(self.config)

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )
        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, discrete_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(discrete_w2, discrete_w3, kernel_size=3, stride=1),
            _dual_conv_block(discrete_w3, discrete_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(discrete_w3, discrete_w4, kernel_size=3, stride=2),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1, continuous_w2, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w2, continuous_w3, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w3, continuous_w4, kernel_size=3, stride=1),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
        )

        joint_width = discrete_w4 + continuous_w4
        inter_scale_num = int(self.config.interaction_width_scale_num)
        inter_scale_den = int(self.config.interaction_width_scale_den)
        inter_width = max(max(discrete_w4, continuous_w4), int(round(joint_width * inter_scale_num / inter_scale_den)))
        self.fuse_to_continuous = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
        )
        gate_init = float(self.config.d2c_gate_init)
        gate_logit = torch.log(torch.tensor(gate_init / (1.0 - gate_init), dtype=torch.float32))
        self.d2c_gate_logits = nn.Parameter(torch.full((1, continuous_w4, 1, 1), gate_logit.item()))
        self.fuse_act = nn.SiLU(inplace=True)

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
            nn.Conv2d(discrete_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
            nn.Conv2d(continuous_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)

        joint_features = torch.cat((discrete_features, continuous_features), dim=1)
        d2c_delta = self.fuse_to_continuous(joint_features)
        d2c_gate = torch.sigmoid(self.d2c_gate_logits).to(device=d2c_delta.device, dtype=d2c_delta.dtype)
        continuous_features = self.fuse_act(continuous_features + d2c_gate * d2c_delta)

        discrete_latent = self.to_discrete_latent(discrete_features)
        continuous_latent = self.to_continuous_latent(continuous_features)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)
        encoder_features = torch.cat((discrete_features, continuous_features), dim=1)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "d2c_gate": d2c_gate,
            "d2c_delta": d2c_delta,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "encoder_features": encoder_features,
            "latent_features": latent_features,
        }


class Model1OptHybridD2CGateView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CGateConfig | None = None):
        resolved_config = config or Model1OptHybridD2CGateConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CGateBackbone(self.config)


class Model1OptHybridD2CGate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGateView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CQuantGuideConfig(Model1OptHybridDualV3D2CConfig):
    quantized_guide_detach: bool = True
    quantized_guide_scale: float = 1.0


class LidarVqD2CQuantGuideBackbone(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridD2CQuantGuideConfig):
        super().__init__(config)
        self.config = config
        w1, _, _, _ = self.config.encoder_widths
        discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4 = _compute_branch_widths(self.config)
        self.continuous_feature_width = continuous_w4

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )
        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, discrete_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(discrete_w2, discrete_w3, kernel_size=3, stride=1),
            _dual_conv_block(discrete_w3, discrete_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(discrete_w3, discrete_w4, kernel_size=3, stride=2),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1, continuous_w2, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w2, continuous_w3, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w3, continuous_w4, kernel_size=3, stride=1),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
        )

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
            nn.Conv2d(discrete_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
            nn.Conv2d(continuous_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )
        self.quantized_guide_projector = nn.Sequential(
            nn.Conv2d(self.config.quantized_channels, continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
            nn.SiLU(inplace=True),
        )
        self.quantized_guide_act = nn.SiLU(inplace=True)

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)
        discrete_latent_features = self.to_discrete_latent(discrete_features)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "discrete_latent_features": discrete_latent_features,
        }

    def inject_quantized_guide(
        self,
        continuous_features: torch.Tensor,
        quantized_discrete: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guide = self.quantized_guide_projector(quantized_discrete)
        guide = F.interpolate(guide, size=continuous_features.shape[-2:], mode="bilinear", align_corners=False)
        guided_continuous = self.quantized_guide_act(
            continuous_features + float(self.config.quantized_guide_scale) * guide
        )
        return guided_continuous, guide


class Model1OptHybridD2CQuantGuideView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CQuantGuideConfig | None = None):
        resolved_config = config or Model1OptHybridD2CQuantGuideConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CQuantGuideBackbone(self.config)

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_encoded = self.backbone.encode_backbone(noisy_dz)
        discrete_latent = backbone_encoded["discrete_latent_features"]
        quantized = self.quantizer(discrete_latent)

        guide_source = quantized["quantized_lookup"]
        if self.config.quantized_guide_detach:
            guide_source = guide_source.detach()
        guided_continuous_features, guide = self.backbone.inject_quantized_guide(
            backbone_encoded["continuous_encoder_features"],
            guide_source,
        )
        continuous_latent = self.backbone.to_continuous_latent(guided_continuous_features)

        continuous_penalty_raw = self._continuous_penalty(continuous_latent)
        continuous_penalty = float(self.config.continuous_penalty_weight) * continuous_penalty_raw
        total_regularizer = quantized["vq_loss"] + continuous_penalty

        lidar_latent = torch.cat((quantized["quantized"], continuous_latent), dim=1)
        quantized_lookup = torch.cat((quantized["quantized_lookup"], continuous_latent), dim=1)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)

        return {
            **backbone_encoded,
            "continuous_guide_map": guide,
            "guided_continuous_encoder_features": guided_continuous_features,
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


class Model1OptHybridD2CQuantGuide(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CQuantGuideConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CQuantGuideConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CQuantGuideView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CQuantGuideView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CGeoLateConfig(Model1OptHybridDualV3D2CConfig):
    geometry_prior_channels: int = 3


class LidarVqD2CGeoLateBackbone(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridD2CGeoLateConfig):
        super().__init__(config)
        self.config = config
        w1, _, _, _ = self.config.encoder_widths
        discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4 = _compute_branch_widths(self.config)

        priors = build_geometry_prior_channels(
            target_height=self.config.raw_height,
            target_width=self.config.raw_width,
        )
        self.register_buffer("geometry_priors", priors, persistent=False)

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )
        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, discrete_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(discrete_w2, discrete_w3, kernel_size=3, stride=1),
            _dual_conv_block(discrete_w3, discrete_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(discrete_w3, discrete_w4, kernel_size=3, stride=2),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1, continuous_w2, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w2, continuous_w3, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w3, continuous_w4, kernel_size=3, stride=1),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
        )

        joint_width = discrete_w4 + continuous_w4
        inter_scale_num = int(self.config.interaction_width_scale_num)
        inter_scale_den = int(self.config.interaction_width_scale_den)
        inter_width = max(max(discrete_w4, continuous_w4), int(round(joint_width * inter_scale_num / inter_scale_den)))
        self.fuse_to_continuous = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
        )
        self.continuous_geo_fuser = nn.Sequential(
            nn.Conv2d(continuous_w4 + int(self.config.geometry_prior_channels), continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
            nn.SiLU(inplace=True),
        )
        self.fuse_act = nn.SiLU(inplace=True)

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
            nn.Conv2d(discrete_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
            nn.Conv2d(continuous_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)

        joint_features = torch.cat((discrete_features, continuous_features), dim=1)
        continuous_features = self.fuse_act(continuous_features + self.fuse_to_continuous(joint_features))

        priors = self.geometry_priors.to(device=padded_input.device, dtype=padded_input.dtype)
        priors = priors.unsqueeze(0).expand(padded_input.shape[0], -1, -1, -1)
        pad = self.config.horizontal_pad
        if pad > 0:
            priors = F.pad(priors, (pad, pad, 0, 0), mode="circular")
        priors = F.interpolate(priors, size=continuous_features.shape[-2:], mode="bilinear", align_corners=False)
        continuous_features = self.continuous_geo_fuser(torch.cat((continuous_features, priors), dim=1))

        discrete_latent = self.to_discrete_latent(discrete_features)
        continuous_latent = self.to_continuous_latent(continuous_features)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)
        encoder_features = torch.cat((discrete_features, continuous_features), dim=1)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "continuous_geometry_priors": priors,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "encoder_features": encoder_features,
            "latent_features": latent_features,
        }


class Model1OptHybridD2CGeoLateView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CGeoLateConfig | None = None):
        resolved_config = config or Model1OptHybridD2CGeoLateConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CGeoLateBackbone(self.config)


class Model1OptHybridD2CGeoLate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoLateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoLateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoLateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoLateView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CSpatialGateConfig(Model1OptHybridDualV3D2CConfig):
    pass


class LidarVqD2CSpatialGateBackbone(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridD2CSpatialGateConfig):
        super().__init__(config)
        self.config = config

        w1, _, _, _ = self.config.encoder_widths
        discrete_w2, discrete_w3, discrete_w4, continuous_w2, continuous_w3, continuous_w4 = _compute_branch_widths(self.config)

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )
        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, discrete_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(discrete_w2, discrete_w3, kernel_size=3, stride=1),
            _dual_conv_block(discrete_w3, discrete_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(discrete_w3, discrete_w4, kernel_size=3, stride=2),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1, continuous_w2, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w2, continuous_w3, kernel_size=3, stride=2),
            _dual_conv_block(continuous_w3, continuous_w4, kernel_size=3, stride=1),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
        )

        joint_width = discrete_w4 + continuous_w4
        inter_scale_num = int(self.config.interaction_width_scale_num)
        inter_scale_den = int(self.config.interaction_width_scale_den)
        inter_width = max(max(discrete_w4, continuous_w4), int(round(joint_width * inter_scale_num / inter_scale_den)))
        self.fuse_to_continuous = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, continuous_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(continuous_w4),
        )
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, continuous_w4, kernel_size=1, stride=1, padding=0),
        )
        self.fuse_act = nn.SiLU(inplace=True)

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(discrete_w4, discrete_w4, kernel_size=3, stride=1),
            nn.Conv2d(discrete_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(continuous_w4, continuous_w4, kernel_size=3, stride=1),
            nn.Conv2d(continuous_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)

        joint_features = torch.cat((discrete_features, continuous_features), dim=1)
        d2c_delta = self.fuse_to_continuous(joint_features)
        d2c_gate = torch.sigmoid(self.fuse_gate(joint_features))
        continuous_features = self.fuse_act(continuous_features + d2c_gate * d2c_delta)

        discrete_latent = self.to_discrete_latent(discrete_features)
        continuous_latent = self.to_continuous_latent(continuous_features)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)
        encoder_features = torch.cat((discrete_features, continuous_features), dim=1)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "d2c_gate": d2c_gate,
            "d2c_delta": d2c_delta,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "encoder_features": encoder_features,
            "latent_features": latent_features,
        }


class Model1OptHybridD2CSpatialGateView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CSpatialGateConfig | None = None):
        resolved_config = config or Model1OptHybridD2CSpatialGateConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqD2CSpatialGateBackbone(self.config)


class Model1OptHybridD2CSpatialGate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CSpatialGateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CSpatialGateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CSpatialGateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CSpatialGateView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CAdditiveConfig(Model1OptHybridDualV3D2CConfig):
    residual_abs_max_m: float = 3.0


class Model1OptHybridD2CAdditiveView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CAdditiveConfig | None = None):
        from .model1_opt_hybrid_additive import _MetricDecoderBranch

        resolved_config = config or Model1OptHybridD2CAdditiveConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        from .model1_opt_hybrid_dual_v3 import LidarVqDualBranchBackboneV3

        self.backbone = LidarVqDualBranchBackboneV3(self.config)
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


class Model1OptHybridD2CAdditive(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CAdditiveConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CAdditiveConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CAdditiveView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CAdditiveView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CGeoLateWideContConfig(Model1OptHybridD2CGeoLateConfig):
    branch_width_scale_num: int = 2
    branch_width_scale_den: int = 3
    continuous_branch_width_scale_num: int = 1
    continuous_branch_width_scale_den: int = 1


class Model1OptHybridD2CGeoLateWideContView(Model1OptHybridD2CGeoLateView):
    def __init__(self, config: Model1OptHybridD2CGeoLateWideContConfig | None = None):
        super().__init__(config=config or Model1OptHybridD2CGeoLateWideContConfig())


class Model1OptHybridD2CGeoLateWideCont(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CGeoLateWideContConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CGeoLateWideContConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CGeoLateWideContView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CGeoLateWideContView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CAdditiveResGateConfig(Model1OptHybridD2CAdditiveConfig):
    residual_gate_bias_init: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.residual_gate_bias_init < 1.0):
            raise ValueError("residual_gate_bias_init must be in (0, 1).")


class Model1OptHybridD2CAdditiveResGateView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CAdditiveResGateConfig | None = None):
        resolved_config = config or Model1OptHybridD2CAdditiveResGateConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        from .model1_opt_hybrid_dual_v3 import LidarVqDualBranchBackboneV3

        self.backbone = LidarVqDualBranchBackboneV3(self.config)
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
        bias_init = float(self.config.residual_gate_bias_init)
        bias_logit = torch.log(torch.tensor(bias_init / (1.0 - bias_init), dtype=torch.float32))
        self.residual_gate_head = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        )
        self.residual_gate_bias = nn.Parameter(torch.tensor(bias_logit.item(), dtype=torch.float32))

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        discrete_latent, continuous_latent = self._split_latent(lidar_latent)
        discrete_out = self.discrete_decoder.decode(discrete_latent)
        continuous_out = self.continuous_decoder.decode(continuous_latent)

        gate_input = torch.cat(
            (
                discrete_out["padded_metric"] / float(self.config.distance_scale),
                continuous_out["padded_metric"] / float(self.config.residual_abs_max_m),
            ),
            dim=1,
        )
        residual_gate = torch.sigmoid(self.residual_gate_head(gate_input) + self.residual_gate_bias)
        gated_residual = residual_gate * continuous_out["padded_metric"]
        padded_reconstruction = torch.clamp(
            discrete_out["padded_metric"] + gated_residual,
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
            "gated_continuous_padded_residual": gated_residual,
            "residual_gate": residual_gate,
            "discrete_reconstruction": self.backbone.crop_output(discrete_out["padded_metric"]),
            "continuous_residual": self.backbone.crop_output(continuous_out["padded_metric"]),
            "gated_continuous_residual": self.backbone.crop_output(gated_residual),
        }


class Model1OptHybridD2CAdditiveResGate(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CAdditiveResGateConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CAdditiveResGateConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CAdditiveResGateView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CAdditiveResGateView(config=self.config)


@dataclass(frozen=True)
class Model1OptHybridD2CCrossViewContConfig(Model1OptHybridDualV3D2CConfig):
    crossview_interaction_width_scale_num: int = 1
    crossview_interaction_width_scale_den: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.crossview_interaction_width_scale_num <= 0 or self.crossview_interaction_width_scale_den <= 0:
            raise ValueError("crossview interaction width scale numerator/denominator must be positive.")


class Model1OptHybridD2CCrossViewContView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridD2CCrossViewContConfig | None = None):
        resolved_config = config or Model1OptHybridD2CCrossViewContConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        from .model1_opt_hybrid_dual_v3 import LidarVqDualBranchBackboneV3

        self.backbone = LidarVqDualBranchBackboneV3(self.config)

    def encode(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        return super().encode(noisy_dz)


class Model1OptHybridD2CCrossViewCont(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridD2CCrossViewContConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridD2CCrossViewContConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridD2CCrossViewContView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridD2CCrossViewContView(config=self.config)

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

        front_discrete = front_encoded["lidar_latent"][:, : self.config.quantized_channels]
        rear_discrete = rear_encoded["lidar_latent"][:, : self.config.quantized_channels]
        front_cont = front_encoded["lidar_latent"][:, self.config.quantized_channels :]
        rear_cont = rear_encoded["lidar_latent"][:, self.config.quantized_channels :]

        mixed_front_cont = self.crossview_cont_mixer(front_cont, rear_cont)
        mixed_rear_cont = self.crossview_cont_mixer(rear_cont, front_cont)

        front_encoded["lidar_latent"] = torch.cat((front_discrete, mixed_front_cont), dim=1)
        rear_encoded["lidar_latent"] = torch.cat((rear_discrete, mixed_rear_cont), dim=1)
        front_encoded["pre_quant_latent_continuous"] = mixed_front_cont
        rear_encoded["pre_quant_latent_continuous"] = mixed_rear_cont
        front_encoded["quantized_lookup"] = torch.cat((front_encoded["quantized_lookup_discrete"], mixed_front_cont), dim=1)
        rear_encoded["quantized_lookup"] = torch.cat((rear_encoded["quantized_lookup_discrete"], mixed_rear_cont), dim=1)
        fused_latent = torch.cat((front_encoded["lidar_latent"], rear_encoded["lidar_latent"]), dim=1)
        return {"front": front_encoded, "rear": rear_encoded, "fused_latent": fused_latent}
