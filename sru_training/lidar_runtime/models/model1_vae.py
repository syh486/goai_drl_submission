from __future__ import annotations

from dataclasses import dataclass

import torch

from .model2_vae import Model2VaeConfig, SharedVaeModel, SharedVaeView


@dataclass(frozen=True)
class Model1VaeConfig(Model2VaeConfig):
    distance_scale: float = 10.0


class Model1VaeView(SharedVaeView):
    def __init__(self, config: Model1VaeConfig | None = None):
        super().__init__(config=config or Model1VaeConfig())
        self.config = config or Model1VaeConfig()

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_reconstruction = torch.sigmoid(reconstruction_logits)
        metric_reconstruction = normalized_reconstruction * float(self.config.distance_scale)
        return normalized_reconstruction, metric_reconstruction


class Model1Vae(SharedVaeModel):
    def __init__(
        self,
        config: Model1VaeConfig | None = None,
        share_view_weights: bool = True,
    ):
        super().__init__(
            config=config or Model1VaeConfig(),
            view_cls=Model1VaeView,
            share_view_weights=share_view_weights,
        )
