from __future__ import annotations

from pathlib import Path

import torch

from model_specs import DEFAULT_MODEL_NAME, build_model


def resolve_model_name_from_checkpoint_payload(
    payload: dict[str, object],
    model_name_override: str | None = None,
) -> str:
    if model_name_override is not None:
        return model_name_override
    args_payload = payload.get("args", {})
    if isinstance(args_payload, dict) and "model_name" in args_payload:
        return str(args_payload["model_name"])
    return DEFAULT_MODEL_NAME


def resolve_model_config_from_checkpoint_payload(payload: dict[str, object]) -> dict[str, object] | None:
    model_config = payload.get("model_config")
    if isinstance(model_config, dict):
        return model_config
    args_payload = payload.get("args", {})
    if isinstance(args_payload, dict) and isinstance(args_payload.get("model_config"), dict):
        return args_payload["model_config"]
    return None


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    model_name_override: str | None = None,
) -> tuple[torch.nn.Module, str, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = resolve_model_name_from_checkpoint_payload(payload, model_name_override)
    model_config = resolve_model_config_from_checkpoint_payload(payload)
    model = build_model(model_name, model_config=model_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, model_name, payload
