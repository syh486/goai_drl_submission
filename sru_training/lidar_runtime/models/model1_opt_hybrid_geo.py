from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from geometry_prior_utils import build_geometry_prior_channels

from .model1_opt_hybrid import Model1OptHybrid, Model1OptHybridConfig, Model1OptHybridView


@dataclass(frozen=True)
class Model1OptHybridGeoConfig(Model1OptHybridConfig):
    input_channels_per_view: int = 5
    quantized_channels: int = 8
    continuous_channels: int = 24
    continuous_budget: float = 0.15
    dead_code_threshold: float = 0.1


class Model1OptHybridGeoView(Model1OptHybridView):
    def __init__(self, config: Model1OptHybridGeoConfig | None = None):
        resolved_config = config or Model1OptHybridGeoConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config
        priors = build_geometry_prior_channels(
            target_height=self.config.raw_height,
            target_width=self.config.raw_width,
        )
        self.register_buffer("geometry_priors", priors, persistent=False)

    def augment_input(self, base_input: torch.Tensor) -> torch.Tensor:
        priors = self.geometry_priors.to(device=base_input.device, dtype=base_input.dtype)
        priors = priors.unsqueeze(0).expand(base_input.shape[0], -1, -1, -1)
        return torch.cat((base_input, priors), dim=1)


class Model1OptHybridGeo(Model1OptHybrid):
    def __init__(
        self,
        config: Model1OptHybridGeoConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1OptHybridGeoConfig()
        super().__init__(config=resolved_config, share_view_weights=share_view_weights)
        self.config = resolved_config
        self.front_model = Model1OptHybridGeoView(config=self.config)
        if share_view_weights:
            self.rear_model = self.front_model
        else:
            self.rear_model = Model1OptHybridGeoView(config=self.config)

    @staticmethod
    def split_full_sample(lidar_sample: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_sample.dim() != 4 or lidar_sample.shape[1] != 6:
            raise ValueError(
                f"Expected a full LiDAR sample batch with shape (B, 6, 96, 90), got {tuple(lidar_sample.shape)}"
            )
        return {
            "front_target": lidar_sample[:, 0:1],
            "rear_target": lidar_sample[:, 1:2],
            "front_input_base": torch.stack((lidar_sample[:, 2], lidar_sample[:, 4]), dim=1),
            "rear_input_base": torch.stack((lidar_sample[:, 3], lidar_sample[:, 5]), dim=1),
        }

    def _augment_inputs(
        self,
        front_input_base: torch.Tensor,
        rear_input_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.front_model.augment_input(front_input_base), self.rear_model.augment_input(rear_input_base)

    def extract_latents(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input, rear_input = self._augment_inputs(split["front_input_base"], split["rear_input_base"])
        elif front_input is not None and rear_input is not None:
            if front_input.shape[1] == 2 and rear_input.shape[1] == 2:
                front_input, rear_input = self._augment_inputs(front_input, rear_input)
            elif front_input.shape[1] != 5 or rear_input.shape[1] != 5:
                raise ValueError("Expected front_input/rear_input to have 2 or 5 channels for model1_opt_hybrid_geo.")
        else:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")
        return super().extract_latents(front_input=front_input, rear_input=rear_input)

    def forward(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
        front_target: torch.Tensor | None = None,
        rear_target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input, rear_input = self._augment_inputs(split["front_input_base"], split["rear_input_base"])
            front_target = split["front_target"]
            rear_target = split["rear_target"]
        elif front_input is not None and rear_input is not None:
            if front_input.shape[1] == 2 and rear_input.shape[1] == 2:
                front_input, rear_input = self._augment_inputs(front_input, rear_input)
            elif front_input.shape[1] != 5 or rear_input.shape[1] != 5:
                raise ValueError("Expected front_input/rear_input to have 2 or 5 channels for model1_opt_hybrid_geo.")
        else:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")
        return super().forward(
            front_input=front_input,
            rear_input=rear_input,
            front_target=front_target,
            rear_target=rear_target,
        )
