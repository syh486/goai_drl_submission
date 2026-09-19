from __future__ import annotations

from dataclasses import dataclass

from .deform_lidar_common import LearnableDeformConvBlock, build_encoder_with_deform_layers
from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView


@dataclass(frozen=True)
class Model1OptHybridTransLearnableConfig(Model1OptHybridConfig):
    deform_layers: tuple[int, ...] = (0, 1)
    learnable_max_offset: float = 1.5


class Model1OptHybridTransLearnableView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridTransLearnableConfig | None = None):
        resolved_config = config or Model1OptHybridTransLearnableConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        self.backbone.encoder = build_encoder_with_deform_layers(
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


class Model1OptHybridTransLearnable(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridTransLearnableConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridTransLearnableConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridTransLearnableView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridTransLearnableView(config=self.config)
