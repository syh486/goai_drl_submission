from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .model2 import DEFAULT_SRU_DEPTH_WEIGHTS, Model2, Model2Config, Model2View


@dataclass(frozen=True)
class Model2V2Config(Model2Config):
    soft_clamp_beta: float = 4.0


class Model2V2View(Model2View):
    def __init__(
        self,
        config: Model2V2Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__(
            config=config or Model2V2Config(),
            pretrained_weights=pretrained_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        self.config = config or Model2V2Config()

    def project_metric_reconstruction(self, reconstruction_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = float(self.config.output_min_m)
        hi = float(self.config.output_max_m)
        beta = float(self.config.soft_clamp_beta)

        # Smooth bounded projection: linear in-range, but keeps non-zero gradients when logits overshoot.
        metric_reconstruction = lo + F.softplus(reconstruction_logits - lo, beta=beta) - F.softplus(
            reconstruction_logits - hi,
            beta=beta,
        )
        return metric_reconstruction, metric_reconstruction


class Model2V2(Model2):
    def __init__(
        self,
        config: Model2V2Config | None = None,
        pretrained_weights: str | Path | None = DEFAULT_SRU_DEPTH_WEIGHTS,
        share_view_weights: bool = True,
        freeze_sru_encoder: bool = True,
        freeze_sru_sampler: bool = True,
    ):
        super().__init__(
            config=config or Model2V2Config(),
            pretrained_weights=pretrained_weights,
            share_view_weights=share_view_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        self.config = config or Model2V2Config()
        self.share_view_weights = share_view_weights

        self.front_model = Model2V2View(
            config=self.config,
            pretrained_weights=pretrained_weights,
            freeze_sru_encoder=freeze_sru_encoder,
            freeze_sru_sampler=freeze_sru_sampler,
        )
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model2V2View(
                config=self.config,
                pretrained_weights=pretrained_weights,
                freeze_sru_encoder=freeze_sru_encoder,
                freeze_sru_sampler=freeze_sru_sampler,
            )
