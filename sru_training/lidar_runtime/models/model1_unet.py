from __future__ import annotations

from dataclasses import dataclass

import torch

from .model2_unet import Model2UnetConfig, SharedUnetModel, SharedUnetView


@dataclass(frozen=True)
class Model1UnetConfig(Model2UnetConfig):
    distance_scale: float = 10.0


class Model1UnetView(SharedUnetView):
    def __init__(self, config: Model1UnetConfig | None = None):
        super().__init__(config=config or Model1UnetConfig())
        self.config = config or Model1UnetConfig()

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction


class Model1Unet(SharedUnetModel):
    def __init__(
        self,
        config: Model1UnetConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model1UnetConfig(),
            view_cls=Model1UnetView,
            share_view_weights=share_view_weights,
        )
