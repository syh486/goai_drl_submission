from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

from lidar_geometry import load_lidar_geometry


def conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def conv_output_dim(input_dim: int, *, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1) -> int:
    return ((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _resample_vertical_angles(target_height: int) -> np.ndarray:
    geometry = load_lidar_geometry()
    source = np.asarray(geometry.vertical_ray_angles, dtype=np.float32)
    source_coords = np.arange(source.shape[0], dtype=np.float32)
    target_coords = np.linspace(0.0, float(source.shape[0] - 1), num=int(target_height), dtype=np.float32)
    return np.interp(target_coords, source_coords, source).astype(np.float32, copy=False)


def _local_angle_diffs(angles_deg: np.ndarray) -> np.ndarray:
    if angles_deg.shape[0] == 1:
        return np.ones((1,), dtype=np.float32)
    diffs = np.empty_like(angles_deg, dtype=np.float32)
    diffs[1:-1] = 0.5 * (angles_deg[2:] - angles_deg[:-2])
    diffs[0] = angles_deg[1] - angles_deg[0]
    diffs[-1] = angles_deg[-1] - angles_deg[-2]
    return np.clip(diffs, 1.0e-4, None)


def build_physics_offset(
    *,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    kernel_size: int = 3,
    gain: float = 1.0,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
) -> torch.Tensor:
    if kernel_size != 3:
        raise ValueError(f"Only kernel_size=3 is currently supported, got {kernel_size}.")

    input_angles = _resample_vertical_angles(int(input_height))
    input_angle_diffs = _local_angle_diffs(input_angles)
    base_scale = float(np.mean(input_angle_diffs))

    input_rows = np.arange(int(input_height), dtype=np.float32)
    output_center_rows = np.linspace(0.0, float(input_height - 1), num=int(output_height), dtype=np.float32)
    local_diffs = np.interp(output_center_rows, input_rows, input_angle_diffs).astype(np.float32, copy=False)

    raw_scale = base_scale / np.clip(local_diffs, 1.0e-4, None)
    effective_scale = 1.0 + float(gain) * (raw_scale - 1.0)
    effective_scale = np.clip(effective_scale, float(min_scale), float(max_scale))

    offset = np.zeros((1, 2 * kernel_size * kernel_size, int(output_height), int(output_width)), dtype=np.float32)
    row_scale = torch.from_numpy(effective_scale).view(1, int(output_height), 1).expand(1, int(output_height), int(output_width))

    kernel_positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for kernel_index, (dy, dx) in enumerate(kernel_positions):
        y_channel = 2 * kernel_index
        x_channel = y_channel + 1
        if dy != 0:
            offset[0, y_channel] = (float(dy) * (row_scale - 1.0)).numpy()
        if dx != 0:
            offset[0, x_channel] = 0.0
    return torch.from_numpy(offset)


class LearnableDeformConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int, max_offset: float):
        super().__init__()
        self.stride = int(stride)
        self.padding = 1
        self.max_offset = float(max_offset)
        self.offset_conv = nn.Conv2d(
            in_channels,
            18,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        self.bias = None
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = torch.tanh(self.offset_conv(x)) * self.max_offset
        y = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
        )
        y = self.norm(y)
        return self.act(y)


class PhysicsDeformConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        input_height: int,
        input_width: int,
        gain: float,
        min_scale: float,
        max_scale: float,
    ):
        super().__init__()
        self.stride = int(stride)
        self.padding = 1

        output_height = conv_output_dim(int(input_height), stride=self.stride, padding=self.padding)
        output_width = conv_output_dim(int(input_width), stride=self.stride, padding=self.padding)
        fixed_offset = build_physics_offset(
            input_height=int(input_height),
            input_width=int(input_width),
            output_height=int(output_height),
            output_width=int(output_width),
            kernel_size=3,
            gain=float(gain),
            min_scale=float(min_scale),
            max_scale=float(max_scale),
        )
        self.register_buffer("fixed_offset", fixed_offset, persistent=False)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        self.bias = None
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.fixed_offset.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        y = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
        )
        y = self.norm(y)
        return self.act(y)


def build_encoder_with_deform_layers(
    config: Any,
    *,
    deform_layers: tuple[int, ...],
    deform_block_factory: Callable[[int, int, int, int, int, int], nn.Module],
) -> nn.Sequential:
    layer_set = set(int(layer_idx) for layer_idx in deform_layers)
    if any(layer_idx < 0 or layer_idx > 4 for layer_idx in layer_set):
        raise ValueError(f"deform_layers must be chosen from 0..4, got {sorted(layer_set)}")

    w1, w2, w3, w4 = config.encoder_widths
    specs = (
        (config.input_channels_per_view, w1, 2),
        (w1, w2, 2),
        (w2, w3, 2),
        (w3, w4, 1),
        (w4, w4, 1),
    )

    layers: list[nn.Module] = []
    current_height = int(config.padded_height)
    current_width = int(config.padded_width)
    for layer_idx, (in_channels, out_channels, stride) in enumerate(specs):
        if layer_idx in layer_set:
            block = deform_block_factory(
                layer_idx,
                int(in_channels),
                int(out_channels),
                int(stride),
                int(current_height),
                int(current_width),
            )
        else:
            block = conv_block(int(in_channels), int(out_channels), stride=int(stride))
        layers.append(block)
        current_height = conv_output_dim(current_height, stride=int(stride))
        current_width = conv_output_dim(current_width, stride=int(stride))
    return nn.Sequential(*layers)
