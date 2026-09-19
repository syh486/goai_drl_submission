from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView
from .model1_opt_hybrid_dual import _dual_conv_block
from .vq_lidar_common import LidarVqViewBackbone


@dataclass(frozen=True)
class Model1OptHybridDualV2Config(Model1OptHybridConfig):
    quantized_channels: int = 8
    continuous_channels: int = 24
    continuous_budget: float = 0.15
    dead_code_threshold: float = 0.1
    branch_width_scale_num: int = 3
    branch_width_scale_den: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.branch_width_scale_num <= 0 or self.branch_width_scale_den <= 0:
            raise ValueError("branch width scale numerator/denominator must be positive.")


class LidarVqDualBranchBackboneV2(LidarVqViewBackbone):
    def __init__(self, config: Model1OptHybridDualV2Config):
        super().__init__(config)
        self.config = config

        w1, w2, w3, w4 = self.config.encoder_widths
        scale_num = int(self.config.branch_width_scale_num)
        scale_den = int(self.config.branch_width_scale_den)

        branch_w2 = max(w1, int(round(w2 * scale_num / scale_den)))
        branch_w3 = max(branch_w2, int(round(w3 * scale_num / scale_den)))
        branch_w4 = max(branch_w3, int(round(w4 * scale_num / scale_den)))

        self.shared_stem = _dual_conv_block(
            self.config.input_channels_per_view,
            w1,
            kernel_size=3,
            stride=2,
        )

        self.discrete_branch = nn.Sequential(
            _dual_conv_block(w1, branch_w2, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _dual_conv_block(branch_w2, branch_w3, kernel_size=3, stride=1),
            _dual_conv_block(branch_w3, branch_w3, kernel_size=3, stride=1, dilation=2),
            _dual_conv_block(branch_w3, branch_w4, kernel_size=3, stride=2),
            _dual_conv_block(branch_w4, branch_w4, kernel_size=3, stride=1),
        )
        self.continuous_branch = nn.Sequential(
            _dual_conv_block(w1, branch_w2, kernel_size=3, stride=2),
            _dual_conv_block(branch_w2, branch_w3, kernel_size=3, stride=2),
            _dual_conv_block(branch_w3, branch_w4, kernel_size=3, stride=1),
            _dual_conv_block(branch_w4, branch_w4, kernel_size=3, stride=1),
        )

        self.to_discrete_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(branch_w4, branch_w4, kernel_size=3, stride=1),
            nn.Conv2d(branch_w4, self.config.quantized_channels, kernel_size=1, stride=1, padding=0),
        )
        self.to_continuous_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.config.latent_height, self.config.latent_width)),
            _dual_conv_block(branch_w4, branch_w4, kernel_size=3, stride=1),
            nn.Conv2d(branch_w4, self.config.continuous_channels, kernel_size=1, stride=1, padding=0),
        )

    def encode_backbone(self, noisy_dz: torch.Tensor) -> dict[str, torch.Tensor]:
        padded_input = self.pad_input(noisy_dz)
        shared_features = self.shared_stem(padded_input)
        discrete_features = self.discrete_branch(shared_features)
        continuous_features = self.continuous_branch(shared_features)
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


class Model1OptHybridDualV2View(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridDualV2Config | None = None):
        resolved_config = config or Model1OptHybridDualV2Config()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone = LidarVqDualBranchBackboneV2(self.config)


class Model1OptHybridDualV2(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridDualV2Config | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridDualV2Config()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridDualV2View(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridDualV2View(config=self.config)
