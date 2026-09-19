from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridView
from .model1_opt_hybrid_dual import _dual_conv_block
from .model1_opt_hybrid_dual_v2 import Model1OptHybridDualV2Config
from .vq_lidar_common import LidarVqViewBackbone


@dataclass(frozen=True)
class Model1OptHybridDualV3Config(Model1OptHybridDualV2Config):
    continuous_branch_width_scale_num: int = 3
    continuous_branch_width_scale_den: int = 4
    interaction_width_scale_num: int = 1
    interaction_width_scale_den: int = 2
    enable_fuse_to_discrete: bool = True
    enable_fuse_to_continuous: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.continuous_branch_width_scale_num <= 0 or self.continuous_branch_width_scale_den <= 0:
            raise ValueError("continuous branch width scale numerator/denominator must be positive.")
        if self.interaction_width_scale_num <= 0 or self.interaction_width_scale_den <= 0:
            raise ValueError("interaction width scale numerator/denominator must be positive.")


@dataclass(frozen=True)
class Model1OptHybridDualV3C2DConfig(Model1OptHybridDualV3Config):
    enable_fuse_to_discrete: bool = True
    enable_fuse_to_continuous: bool = False


@dataclass(frozen=True)
class Model1OptHybridDualV3D2CConfig(Model1OptHybridDualV3Config):
    enable_fuse_to_discrete: bool = False
    enable_fuse_to_continuous: bool = True


@dataclass(frozen=True)
class Model1OptHybridDualV3WideContConfig(Model1OptHybridDualV3Config):
    branch_width_scale_num: int = 2
    branch_width_scale_den: int = 3
    continuous_branch_width_scale_num: int = 1
    continuous_branch_width_scale_den: int = 1


class LidarVqDualBranchBackboneV3(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridDualV3Config):
        super().__init__(config)
        self.config = config

        w1, w2, w3, w4 = self.config.encoder_widths
        discrete_scale_num = int(self.config.branch_width_scale_num)
        discrete_scale_den = int(self.config.branch_width_scale_den)
        continuous_scale_num = int(self.config.continuous_branch_width_scale_num)
        continuous_scale_den = int(self.config.continuous_branch_width_scale_den)

        discrete_w2 = max(w1, int(round(w2 * discrete_scale_num / discrete_scale_den)))
        discrete_w3 = max(discrete_w2, int(round(w3 * discrete_scale_num / discrete_scale_den)))
        discrete_w4 = max(discrete_w3, int(round(w4 * discrete_scale_num / discrete_scale_den)))

        continuous_w2 = max(w1, int(round(w2 * continuous_scale_num / continuous_scale_den)))
        continuous_w3 = max(continuous_w2, int(round(w3 * continuous_scale_num / continuous_scale_den)))
        continuous_w4 = max(continuous_w3, int(round(w4 * continuous_scale_num / continuous_scale_den)))

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

        inter_scale_num = int(self.config.interaction_width_scale_num)
        inter_scale_den = int(self.config.interaction_width_scale_den)
        joint_width = discrete_w4 + continuous_w4
        inter_width = max(max(discrete_w4, continuous_w4), int(round(joint_width * inter_scale_num / inter_scale_den)))

        self.fuse_to_discrete = nn.Sequential(
            nn.Conv2d(joint_width, inter_width, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_width, discrete_w4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(discrete_w4),
        )
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
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)

        joint_features = torch.cat((discrete_features, continuous_features), dim=1)
        if self.config.enable_fuse_to_discrete:
            discrete_features = self.fuse_act(discrete_features + self.fuse_to_discrete(joint_features))
        if self.config.enable_fuse_to_continuous:
            continuous_features = self.fuse_act(continuous_features + self.fuse_to_continuous(joint_features))

        discrete_latent = self.to_discrete_latent(discrete_features)
        continuous_latent = self.to_continuous_latent(continuous_features)
        latent_features = torch.cat((discrete_latent, continuous_latent), dim=1)
        encoder_features = torch.cat((discrete_features, continuous_features), dim=1)
        return {
            "padded_input": padded_input,
            "shared_features": shared_features,
            "discrete_encoder_features": discrete_features,
            "continuous_encoder_features": continuous_features,
            "encoder_features": encoder_features,
            "latent_features": latent_features,
        }


class Model1OptHybridDualV3View(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridDualV3Config | None = None):
        resolved_config = config or Model1OptHybridDualV3Config()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqDualBranchBackboneV3(self.config)


class Model1OptHybridDualV3(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridDualV3Config | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridDualV3Config()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridDualV3View(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridDualV3View(config=self.config)


class Model1OptHybridDualV3C2DView(Model1OptHybridDualV3View):
    def __init__(self, config: Model1OptHybridDualV3C2DConfig | None = None):
        super().__init__(config=config or Model1OptHybridDualV3C2DConfig())


class Model1OptHybridDualV3C2D(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridDualV3C2DConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridDualV3C2DConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridDualV3C2DView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridDualV3C2DView(config=self.config)


class Model1OptHybridDualV3D2CView(Model1OptHybridDualV3View):
    def __init__(self, config: Model1OptHybridDualV3D2CConfig | None = None):
        super().__init__(config=config or Model1OptHybridDualV3D2CConfig())


class Model1OptHybridDualV3D2C(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridDualV3D2CConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridDualV3D2CConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridDualV3D2CView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridDualV3D2CView(config=self.config)


class Model1OptHybridDualV3WideContView(Model1OptHybridDualV3View):
    def __init__(self, config: Model1OptHybridDualV3WideContConfig | None = None):
        super().__init__(config=config or Model1OptHybridDualV3WideContConfig())


class Model1OptHybridDualV3WideCont(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridDualV3WideContConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridDualV3WideContConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridDualV3WideContView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridDualV3WideContView(config=self.config)
