from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model2 import DEFAULT_SRU_DEPTH_WEIGHTS, Model2, Model2Config, Model2View


@dataclass(frozen=True)
class Model2V3Config(Model2Config):
    output_min_m: float = -10.0
    output_max_m: float = 10.0
    symmetric_output_abs_max_m: float = 10.0


class Model2V3View(Model2View):
    def __init__(
        self,
        config: Model2V3Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__(
            config=config or Model2V3Config(),
            pretrained_weights=pretrained_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        self.config = config or Model2V3Config()

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        metric_reconstruction = float(self.config.symmetric_output_abs_max_m) * torch.tanh(reconstruction_logits)
        return metric_reconstruction, metric_reconstruction


class Model2V3(Model2):
    def __init__(
        self,
        config: Model2V3Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        share_view_weights: bool = True,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__(
            config=config or Model2V3Config(),
            pretrained_weights=pretrained_weights,
            share_view_weights=share_view_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        self.config = config or Model2V3Config()
        self.share_view_weights = share_view_weights

        self.front_model = Model2V3View(
            config=self.config,
            pretrained_weights=pretrained_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model2V3View(
                config=self.config,
                pretrained_weights=pretrained_weights,
                freeze_sru_encoder=freeze_sru_encoder,
                freeze_sru_sampler=freeze_sru_sampler,
            )
