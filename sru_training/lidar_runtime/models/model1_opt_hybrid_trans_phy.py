from __future__ import annotations

from dataclasses import dataclass

from .deform_lidar_common import PhysicsDeformConvBlock, build_encoder_with_deform_layers
from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView


@dataclass(frozen=True)
class Model1OptHybridTransPhyConfig(Model1OptHybridConfig):
    deform_layers: tuple[int, ...] = (0, 1)
    physics_offset_gain: float = 1.0
    physics_min_scale: float = 0.5
    physics_max_scale: float = 2.0


class Model1OptHybridTransPhyView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridTransPhyConfig | None = None):
        resolved_config = config or Model1OptHybridTransPhyConfig()
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


class Model1OptHybridTransPhy(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridTransPhyConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridTransPhyConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridTransPhyView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridTransPhyView(config=self.config)
