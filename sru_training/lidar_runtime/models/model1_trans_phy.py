from __future__ import annotations

from dataclasses import dataclass

import torch

from .deform_lidar_common import PhysicsDeformConvBlock, build_encoder_with_deform_layers
from .model1_vae import Model1VaeConfig
from .model2_vae import SharedVaeModel, SharedVaeView


@dataclass(frozen=True)
class Model1TransPhyConfig(Model1VaeConfig):
    deform_layers: tuple[int, ...] = (0, 1)
    physics_offset_gain: float = 1.0
    physics_min_scale: float = 0.5
    physics_max_scale: float = 2.0


class Model1TransPhyView(SharedVaeView):
    def __init__(self, config: Model1TransPhyConfig | None = None):
        resolved_config = config or Model1TransPhyConfig()
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
    ) -> PhysicsDeformConvBlock:
        return PhysicsDeformConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            input_height=input_height,
            input_width=input_width,
            gain=float(self.config.physics_offset_gain),
            min_scale=float(self.config.physics_min_scale),
            max_scale=float(self.config.physics_max_scale),
        )

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction


class Model1TransPhy(SharedVaeModel):
    def __init__(
        self,
        config: Model1TransPhyConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model1TransPhyConfig(),
            view_cls=Model1TransPhyView,
            share_view_weights=share_view_weights,
        )
