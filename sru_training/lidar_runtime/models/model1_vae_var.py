from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from probability_map_utils import load_probability_map, resolve_probability_map_paths

from .model2_vae import Model2VaeConfig, SharedVaeModel, SharedVaeView, _conv_block


@dataclass(frozen=True)
class Model1VaeVarConfig(Model2VaeConfig):
    distance_scale: float = 10.0
    input_channels_per_view: int = 3
    target_channels_per_view: int = 1
    probability_map_dir: str | None = None
    require_probability_maps: bool = False
    min_variance_m2: float = 1.0e-3
    max_variance_m2: float = 25.0


class Model1VaeVarView(SharedVaeView):
    def __init__(self, config: Model1VaeVarConfig | None = None):
        resolved_config = config or Model1VaeVarConfig()
        super().__init__(config=resolved_config)
        self.config = resolved_config

        w1 = self.config.encoder_widths[0]
        self.decode_head = nn.Sequential(
            _conv_block(w1, w1, stride=1),
            nn.Conv2d(w1, 2, kernel_size=3, stride=1, padding=1),
        )

    def project_metric_reconstruction(
        self,
        reconstruction_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_logits = reconstruction_logits[:, 0:1]
        logvar_logits = reconstruction_logits[:, 1:2]

        normalized_mu = torch.sigmoid(mu_logits)
        mu_metric = normalized_mu * float(self.config.distance_scale)

        variance = torch.nn.functional.softplus(logvar_logits) + float(self.config.min_variance_m2)
        variance = torch.clamp(variance, max=float(self.config.max_variance_m2))
        log_variance = torch.log(variance)
        return normalized_mu, mu_metric, variance, log_variance

    def decode(self, lidar_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if lidar_latent.dim() != 4:
            raise ValueError(f"Expected latent to be 4D, got shape {tuple(lidar_latent.shape)}")
        if lidar_latent.shape[1] != self.config.lidar_latent_channels:
            raise ValueError(
                f"Expected latent channels {self.config.lidar_latent_channels}, got {lidar_latent.shape[1]}"
            )

        expanded_latent = self.from_latent(lidar_latent)
        x = torch.nn.functional.interpolate(expanded_latent, size=(12, 12), mode="bilinear", align_corners=False)
        x = self.decode_block_12(x)
        x = torch.nn.functional.interpolate(x, size=(24, 24), mode="bilinear", align_corners=False)
        x = self.decode_block_24(x)
        x = torch.nn.functional.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        x = self.decode_block_48(x)
        x = torch.nn.functional.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        x = self.decode_block_96(x)
        padded_output = self.decode_head(x)
        raw_width_output = self.crop_output(padded_output)

        _, padded_mu, padded_var, padded_logvar = self.project_metric_reconstruction(padded_output)
        normalized_mu, raw_width_mu, raw_width_var, raw_width_logvar = self.project_metric_reconstruction(raw_width_output)
        return {
            "expanded_latent": expanded_latent,
            "padded_reconstruction_logits": padded_output,
            "padded_reconstruction": padded_mu,
            "padded_variance": padded_var,
            "padded_log_variance": padded_logvar,
            "reconstruction_logits": raw_width_output[:, 0:1],
            "variance_logits": raw_width_output[:, 1:2],
            "normalized_reconstruction": normalized_mu,
            "reconstruction": raw_width_mu,
            "variance": raw_width_var,
            "log_variance": raw_width_logvar,
        }


class Model1VaeVar(SharedVaeModel):
    def __init__(
        self,
        config: Model1VaeVarConfig | None = None,
        share_view_weights: bool = True,
    ):
        resolved_config = config or Model1VaeVarConfig()
        super().__init__(
            config=resolved_config,
            view_cls=Model1VaeVarView,
            share_view_weights=share_view_weights,
        )
        self.config = resolved_config

        front_path, rear_path = resolve_probability_map_paths(
            probability_map_dir=self.config.probability_map_dir,
            require_probability_maps=self.config.require_probability_maps,
        )
        front_prob = load_probability_map(front_path, target_height=self.config.raw_height, target_width=self.config.raw_width)
        rear_prob = load_probability_map(rear_path, target_height=self.config.raw_height, target_width=self.config.raw_width)
        self.register_buffer("front_probability_map", front_prob.unsqueeze(0), persistent=False)
        self.register_buffer("rear_probability_map", rear_prob.unsqueeze(0), persistent=False)

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
        front_prob = self.front_probability_map.to(device=front_input_base.device, dtype=front_input_base.dtype)
        rear_prob = self.rear_probability_map.to(device=rear_input_base.device, dtype=rear_input_base.dtype)
        front_prob = front_prob.expand(front_input_base.shape[0], -1, -1)
        rear_prob = rear_prob.expand(rear_input_base.shape[0], -1, -1)
        front_input = torch.cat((front_input_base, front_prob.unsqueeze(1)), dim=1)
        rear_input = torch.cat((rear_input_base, rear_prob.unsqueeze(1)), dim=1)
        return front_input, rear_input

    def extract_latents(
        self,
        lidar_sample: torch.Tensor | None = None,
        *,
        front_input: torch.Tensor | None = None,
        rear_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar_sample is not None:
            split = self.split_full_sample(lidar_sample)
            front_input_base = split["front_input_base"]
            rear_input_base = split["rear_input_base"]
            front_input, rear_input = self._augment_inputs(front_input_base, rear_input_base)
        elif front_input is not None and rear_input is not None:
            if front_input.shape[1] == 2 and rear_input.shape[1] == 2:
                front_input, rear_input = self._augment_inputs(front_input, rear_input)
            elif front_input.shape[1] != 3 or rear_input.shape[1] != 3:
                raise ValueError("Expected front_input/rear_input to have 2 or 3 channels for model1_vae_var.")
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
            elif front_input.shape[1] != 3 or rear_input.shape[1] != 3:
                raise ValueError("Expected front_input/rear_input to have 2 or 3 channels for model1_vae_var.")
        else:
            raise ValueError("Either lidar_sample or both front_input/rear_input tensors must be provided.")

        return super().forward(
            front_input=front_input,
            rear_input=rear_input,
            front_target=front_target,
            rear_target=rear_target,
        )
