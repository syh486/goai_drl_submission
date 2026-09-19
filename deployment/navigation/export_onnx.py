#!/usr/bin/env python3
"""Export and numerically verify the S10 high-level navigation ONNX models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sru_training.rsl_rl.modules.actor_critic_sru import (
    ActorCriticSRU,
    _ActorCriticSRUONNXExporterSingleCam,
)
from sru_training.s10_lidar_encoder import (
    ENCODER_DISTANCE_SCALE_M,
    ENCODER_SHAPE,
    ENCODER_WORLD_Z_SCALE_M,
    S10LegacyLidarEncoder,
)
from sru_training.s10_policy_config import (
    S10ActionSpec,
    mdpo_config,
    observation_spec_from_policy_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "checkpoints/navigation/sru_deploy_model_2750.pt"
DEFAULT_ENCODER = REPO_ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "deployment/models"


class StaticAdaptiveAvgPool2d(torch.nn.Module):
    """Exact fixed-shape adaptive average pooling using ONNX slice/reduce ops."""

    def __init__(self, input_size: tuple[int, int], output_size: tuple[int, int]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_height, input_width = self.input_size
        output_height, output_width = self.output_size
        rows: list[torch.Tensor] = []
        for row in range(output_height):
            row_start = (row * input_height) // output_height
            row_end = ((row + 1) * input_height + output_height - 1) // output_height
            cells: list[torch.Tensor] = []
            for column in range(output_width):
                column_start = (column * input_width) // output_width
                column_end = ((column + 1) * input_width + output_width - 1) // output_width
                cells.append(
                    value[..., row_start:row_end, column_start:column_end].mean(
                        dim=(-2, -1), keepdim=True
                    )
                )
            rows.append(torch.cat(cells, dim=-1))
        return torch.cat(rows, dim=-2)


def _replace_adaptive_pool_for_export(module: torch.nn.Module) -> None:
    """Replace the encoder's known 12x12 -> 5x8 pools without changing weights."""

    for name, child in tuple(module.named_children()):
        if isinstance(child, torch.nn.AdaptiveAvgPool2d):
            output_size = tuple(int(value) for value in child.output_size)
            if output_size != (5, 8):
                raise ValueError(f"unexpected adaptive pool output size: {output_size}")
            setattr(module, name, StaticAdaptiveAvgPool2d((12, 12), output_size))
        else:
            _replace_adaptive_pool_for_export(child)


class EncoderExportWrapper(torch.nn.Module):
    """Metric raster input wrapper preserving the trained encoder contract."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        front_distance: torch.Tensor,
        rear_distance: torch.Tensor,
        front_world_z: torch.Tensor,
        rear_world_z: torch.Tensor,
    ) -> torch.Tensor:
        front_input = torch.stack(
            (
                front_distance / ENCODER_DISTANCE_SCALE_M,
                front_world_z / ENCODER_WORLD_Z_SCALE_M,
            ),
            dim=1,
        )
        rear_input = torch.stack(
            (
                rear_distance / ENCODER_DISTANCE_SCALE_M,
                rear_world_z / ENCODER_WORLD_Z_SCALE_M,
            ),
            dim=1,
        )
        return self.model.extract_latents(
            front_input=front_input, rear_input=rear_input
        )["fused_latent"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(checkpoint: Path, policy_index: int) -> tuple[ActorCriticSRU, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_key = "model_state_dict" if policy_index == 1 else "model_state_dict_2"
    if state_key not in payload:
        raise KeyError(f"{checkpoint} does not contain {state_key}")
    state_dict = payload[state_key]
    spec = observation_spec_from_policy_state(state_dict)
    if spec.next_waypoint_dim:
        raise ValueError("deployment export only supports current-goal observations")
    policy_config = dict(mdpo_config(smoke=False)["policy"])
    policy_config.pop("class_name")
    policy_config["rnn_hidden_size"] = int(
        state_dict["memory_a.rnn.cells.0.transform_gate.weight"].shape[0]
    )
    policy = ActorCriticSRU(
        spec.actor_obs_dim,
        spec.critic_obs_dim,
        2,
        **policy_config,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy, spec


def _actor_wrapper(policy: ActorCriticSRU) -> _ActorCriticSRUONNXExporterSingleCam:
    if policy.num_cameras != 1:
        raise ValueError(f"expected one fused LiDAR camera, got {policy.num_cameras}")
    wrapper = _ActorCriticSRUONNXExporterSingleCam(
        attn_image_net=policy.attn_image_net,
        memory_a=policy.memory_a,
        linear_dropout_actor=policy.linear_dropout_actor,
        actor=policy.actor,
        image_input_dims=policy.image_input_dims,
        num_image_features=policy.num_image_features,
        actor_proprioceptive_input_dim=policy.actor_proprioceptive_input_dim,
        normalizer=None,
    )
    return wrapper.eval()


def _export_encoder(wrapper: EncoderExportWrapper, path: Path) -> None:
    height, width = ENCODER_SHAPE
    dummy = tuple(torch.zeros(1, height, width) for _ in range(4))
    names = ["front_distance", "rear_distance", "front_world_z", "rear_world_z"]
    dynamic_axes = {name: {0: "batch_size"} for name in names}
    dynamic_axes["fused_latent"] = {0: "batch_size"}
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=names,
        output_names=["fused_latent"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )


def _make_inputs(sequence_length: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (sequence_length, *ENCODER_SHAPE)
    # Metric values cover valid returns, max-range cells, and signed world Z.
    front_distance = rng.uniform(0.35, 10.0, size=shape).astype(np.float32)
    rear_distance = rng.uniform(0.35, 10.0, size=shape).astype(np.float32)
    for value in (front_distance, rear_distance):
        value[rng.random(shape) < 0.08] = 10.0
    front_world_z = rng.uniform(-3.0, 3.0, size=shape).astype(np.float32)
    rear_world_z = rng.uniform(-3.0, 3.0, size=shape).astype(np.float32)
    proprio = rng.normal(0.0, 0.35, size=(sequence_length, 15)).astype(np.float32)
    proprio[:, 8] = rng.uniform(-1.0, -0.75, size=sequence_length)
    return {
        "front_distance": front_distance,
        "rear_distance": rear_distance,
        "front_world_z": front_world_z,
        "rear_world_z": rear_world_z,
        "proprio": proprio,
    }


def _error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


@torch.inference_mode()
def _verify(
    encoder_reference: EncoderExportWrapper,
    actor_wrapper: _ActorCriticSRUONNXExporterSingleCam,
    encoder_path: Path,
    actor_path: Path,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(encoder_path))
    onnx.checker.check_model(onnx.load(actor_path))
    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    actor_session = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])

    encoder_feed = {key: inputs[key] for key in (
        "front_distance", "rear_distance", "front_world_z", "rear_world_z"
    )}
    torch_latent = encoder_reference(*(
        torch.from_numpy(encoder_feed[key]) for key in encoder_feed
    )).cpu().numpy()
    ort_latent = encoder_session.run(["fused_latent"], encoder_feed)[0]

    rnn = actor_wrapper.rnn
    torch_h = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
    torch_c = torch.zeros_like(torch_h)
    ort_h = np.zeros(tuple(torch_h.shape), dtype=np.float32)
    ort_c = np.zeros_like(ort_h)
    torch_actions: list[np.ndarray] = []
    ort_actions: list[np.ndarray] = []
    state_errors: list[float] = []
    for index in range(len(inputs["proprio"])):
        obs = np.concatenate((inputs["proprio"][index:index + 1], ort_latent[index:index + 1].reshape(1, -1)), axis=1)
        torch_action, torch_h, torch_c = actor_wrapper(
            torch.from_numpy(obs), torch_h, torch_c
        )
        ort_action, ort_h, ort_c = actor_session.run(
            ["actions", "h_out", "c_out"],
            {"obs": obs, "h_in": ort_h, "c_in": ort_c},
        )
        torch_actions.append(torch_action.cpu().numpy())
        ort_actions.append(ort_action)
        state_errors.extend((
            float(np.max(np.abs(torch_h.cpu().numpy() - ort_h))),
            float(np.max(np.abs(torch_c.cpu().numpy() - ort_c))),
        ))
    torch_actions_array = np.concatenate(torch_actions)
    ort_actions_array = np.concatenate(ort_actions)

    action_spec = S10ActionSpec()
    target = np.zeros((len(ort_actions_array), 3), dtype=np.float32)
    target[:, 0] = np.clip(
        np.tanh(ort_actions_array[:, 0]) * action_spec.policy_scale_vx,
        -action_spec.max_vx,
        action_spec.max_vx,
    )
    target[:, 2] = np.clip(
        np.tanh(ort_actions_array[:, 1]) * action_spec.policy_scale_yaw,
        -action_spec.max_yaw,
        action_spec.max_yaw,
    )
    filtered = np.zeros_like(target)
    for index in range(len(target)):
        previous = filtered[index - 1] if index else np.zeros(3, dtype=np.float32)
        filtered[index] = 0.5 * previous + 0.5 * target[index]

    result = {
        "encoder": _error(torch_latent, ort_latent),
        "actor_actions": _error(torch_actions_array, ort_actions_array),
        "actor_state_max_abs": max(state_errors),
        "sequence_length": int(len(ort_actions_array)),
    }
    if result["encoder"]["max_abs"] > 2.0e-4:
        raise RuntimeError(f"encoder ONNX mismatch: {result['encoder']}")
    if result["actor_actions"]["max_abs"] > 2.0e-4 or result["actor_state_max_abs"] > 2.0e-4:
        raise RuntimeError(f"actor ONNX mismatch: {result}")
    outputs = {
        **inputs,
        "encoder_fused_latent": ort_latent.astype(np.float32),
        "actor_actions": ort_actions_array.astype(np.float32),
        "filtered_cmd_vx_vy_yaw": filtered,
    }
    return result, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--policy-index", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = args.policy.expanduser().resolve()
    encoder_checkpoint = args.encoder.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    encoder = S10LegacyLidarEncoder(encoder_checkpoint, "cpu")
    encoder_reference = EncoderExportWrapper(encoder.model).eval()
    encoder_export_model = copy.deepcopy(encoder.model)
    _replace_adaptive_pool_for_export(encoder_export_model)
    encoder_wrapper = EncoderExportWrapper(encoder_export_model).eval()
    policy, spec = _load_policy(policy_path, args.policy_index)
    actor_wrapper = _actor_wrapper(policy)

    encoder_path = output_dir / "s10_lidar_encoder.onnx"
    actor_path = output_dir / "sru_policy1_model2750.onnx"
    _export_encoder(encoder_wrapper, encoder_path)
    policy.export_onnx(str(output_dir), actor_path.name)

    inputs = _make_inputs(args.sequence_length, args.seed)
    verification, outputs = _verify(
        encoder_reference, actor_wrapper, encoder_path, actor_path, inputs
    )
    vectors_path = output_dir / "high_level_verification_vectors.npz"
    np.savez_compressed(vectors_path, **outputs)

    action_spec = S10ActionSpec()
    manifest = {
        "schema_version": 1,
        "policy": {
            "source": str(policy_path.relative_to(REPO_ROOT)),
            "source_sha256": _sha256(policy_path),
            "policy_index": args.policy_index,
            "onnx": actor_path.name,
            "onnx_sha256": _sha256(actor_path),
            "inputs": {
                "obs": ["batch", spec.actor_obs_dim],
                "h_in": [1, "batch", int(actor_wrapper.rnn.hidden_size)],
                "c_in": [1, "batch", int(actor_wrapper.rnn.hidden_size)],
            },
            "outputs": ["actions", "h_out", "c_out"],
            "observation_order": [
                "base_lin_vel_body_xyz[3]",
                "base_ang_vel_body_xyz[3]",
                "projected_gravity_body_xyz[3]",
                "previous_raw_policy_action[2]",
                "goal_body_xy_z_distance[4]",
                "fused_lidar_latent_flat[64*5*8]",
            ],
            "recurrent_reset": "zero h_in and c_in at navigation start only",
        },
        "encoder": {
            "source": str(encoder_checkpoint.relative_to(REPO_ROOT)),
            "source_sha256": _sha256(encoder_checkpoint),
            "model_name": encoder.model_name,
            "onnx": encoder_path.name,
            "onnx_sha256": _sha256(encoder_path),
            "inputs": {
                "front_distance": ["batch", 96, 90],
                "rear_distance": ["batch", 96, 90],
                "front_world_z": ["batch", 96, 90],
                "rear_world_z": ["batch", 96, 90],
            },
            "input_units": {"distance": "metres", "world_z": "metres"},
            "internal_normalization": {
                "distance_divisor_m": ENCODER_DISTANCE_SCALE_M,
                "world_z_divisor_m": ENCODER_WORLD_Z_SCALE_M,
            },
            "output": {"fused_latent": ["batch", 64, 5, 8]},
        },
        "action_processing": {
            "raw_action": "unbounded [vx,yaw] actor output",
            "tanh_then_scale": [action_spec.policy_scale_vx, action_spec.policy_scale_yaw],
            "clamp_vx_yaw": [action_spec.max_vx, action_spec.max_yaw],
            "low_pass_alpha_previous": 0.5,
            "policy_hz": action_spec.policy_hz,
            "command_order": ["vx", "vy=0", "yaw_rate"],
        },
        "verification": verification,
        "verification_vectors": vectors_path.name,
    }
    manifest_path = output_dir / "high_level_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
