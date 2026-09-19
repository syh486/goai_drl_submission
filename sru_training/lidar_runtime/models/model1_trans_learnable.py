from __future__ import annotations

from dataclasses import dataclass

import torch

from .deform_lidar_common import LearnableDeformConvBlock, build_encoder_with_deform_layers
from .model1_vae import Model1VaeConfig
from .model2_vae import SharedVaeModel, SharedVaeView


@dataclass(frozen=True)
class Model1TransLearnableConfig(Model1VaeConfig):
    deform_layers: tuple[int, ...] = (0, 1)
    learnable_max_offset: float = 1.5


class Model1TransLearnableView(SharedVaeView):
    def __init__(self, config: Model1TransLearnableConfig | None = None):
        resolved_config = config or Model1TransLearnableConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.encoder = build_encoder_with_deform_layers(
            self.config,
            deform_layers=tuple(self.config.deform_layers),
            deform_block_factory=self._make_deform_block,
        )

    def _make_deform_block(
        self,
        layer_idx: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        input_height: int,
        input_width: int,
    ) -> LearnableDeformConvBlock:
        return LearnableDeformConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            max_offset=float(self.config.learnable_max_offset),
        )

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction


class Model1TransLearnable(SharedVaeModel):
    def __init__(
        self,
        config: Model1TransLearnableConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model1TransLearnableConfig(),
            view_cls=Model1TransLearnableView,
            share_view_weights=share_view_weights,
        )
