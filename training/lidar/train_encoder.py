"""Fine-tune the S10 LiDAR encoder on growing random-terrain MuJoCo replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The migrated runtime retains the original model package's top-level import
# layout (``model_specs``, ``models``). Make that contract explicit at this
# entry point instead of relying on the caller's PYTHONPATH.
RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "sru_training" / "lidar_runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from sru_training.lidar_runtime.inference_utils import load_checkpoint_model
from .replay import ReplayIndex, build_training_batch_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-samples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--updates-per-new-sample", type=float, default=0.5)
    parser.add_argument("--updates-per-chunk", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--near-pixel-weight", type=float, default=2.0)
    parser.add_argument("--near-threshold-m", type=float, default=9.9)
    parser.add_argument("--noise-std-m", type=float, default=0.03)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def freeze_stable_parts(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
        if hasattr(module, "cluster_size") and hasattr(module, "embed_avg"):
            module.eval()
    for name, parameter in model.named_parameters():
        if ".discrete_decoder." in name or ".continuous_decoder." in name:
            parameter.requires_grad = False


def to_torch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value).to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in batch.items()
    }


def reconstruction_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], near_weight: float, near_threshold: float) -> tuple[torch.Tensor, dict[str, float]]:
    front_target = batch["front_target"]
    rear_target = batch["rear_target"]
    front_error = (output["front"]["reconstruction"] - front_target).abs()
    rear_error = (output["rear"]["reconstruction"] - rear_target).abs()
    front_weights = torch.where(front_target < near_threshold, near_weight, 1.0)
    rear_weights = torch.where(rear_target < near_threshold, near_weight, 1.0)
    loss = (
        (front_error * front_weights).sum() / front_weights.sum().clamp_min(1.0)
        + (rear_error * rear_weights).sum() / rear_weights.sum().clamp_min(1.0)
    )
    return loss, {
        "front_l1_m": float(front_error.detach().mean()),
        "rear_l1_m": float(rear_error.detach().mean()),
    }


def split_indices(replay: ReplayIndex, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if len(replay) < 2:
        return np.arange(len(replay), dtype=np.int64), np.empty((0,), dtype=np.int64)
    val_mod = max(2, int(round(1.0 / max(fraction, 1.0e-3))))
    val = np.asarray([record.episode_id % val_mod == 0 for record in replay.records], dtype=bool)
    if not val.any() or val.all():
        val = np.zeros(len(replay), dtype=bool)
        val[::val_mod] = True
    return np.flatnonzero(~val), np.flatnonzero(val)


@torch.no_grad()
def validate(model: nn.Module, replay: ReplayIndex, indices: np.ndarray, args: argparse.Namespace, device: torch.device, rng: np.random.Generator) -> dict[str, float]:
    model.eval()
    sums = {"front_l1_m": 0.0, "rear_l1_m": 0.0, "consistency_l1": 0.0, "finite_ratio": 0.0}
    count = 0
    for batch_id in range(args.validation_batches):
        chosen = indices[batch_id * args.batch_size : (batch_id + 1) * args.batch_size]
        if len(chosen) == 0:
            break
        batch_a = build_training_batch_torch(replay.load_raw_batch(chosen), device=device, noise_std_m=args.noise_std_m, dropout_probability=args.dropout_probability)
        batch_b = build_training_batch_torch(replay.load_raw_batch(chosen), device=device, noise_std_m=args.noise_std_m, dropout_probability=args.dropout_probability)
        output = model(front_input=batch_a["front_input"], rear_input=batch_a["rear_input"])
        latent_b = model.extract_latents(front_input=batch_b["front_input"], rear_input=batch_b["rear_input"])["fused_latent"]
        front_l1 = F.l1_loss(output["front"]["reconstruction"], batch_a["front_target"])
        rear_l1 = F.l1_loss(output["rear"]["reconstruction"], batch_a["rear_target"])
        batch_count = len(chosen)
        sums["front_l1_m"] += float(front_l1) * batch_count
        sums["rear_l1_m"] += float(rear_l1) * batch_count
        sums["consistency_l1"] += float(F.l1_loss(output["fused_latent"], latent_b)) * batch_count
        sums["finite_ratio"] += float(torch.isfinite(output["fused_latent"]).float().mean()) * batch_count
        count += batch_count
    model.train()
    freeze_stable_parts(model)
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, model_name: str, base_payload: dict, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 0,
        "step": int(step),
        "args": {**vars(args), "model_name": model_name},
        "model_config": base_payload.get("model_config"),
        "metrics": metrics,
        "base_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "input_contract": {"distance_scale_m": 10.0, "world_z_scale_m": 3.0, "native_shape": [96, 900], "encoder_shape": [96, 90]},
    }
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.warmup_samples = min(args.warmup_samples, 4)
        args.max_steps = min(args.max_steps, 3)
        args.batch_size = min(args.batch_size, 2)
        args.validation_batches = 1
        args.validate_every = 1
        args.save_every = 1
        args.poll_seconds = 0.2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    model, model_name, base_payload = load_checkpoint_model(args.checkpoint, device=device)
    model.train()
    freeze_stable_parts(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    replay = ReplayIndex(args.replay_dir)
    rng = np.random.default_rng(args.seed)
    update_budget = 0.0
    previous_samples = 0
    active_chunk: np.ndarray | None = None
    chunk_updates_left = 0
    best_val = float("inf")
    step = 0
    (args.output_dir.expanduser().resolve() / "run_config.json").write_text(json.dumps({**vars(args), "model_name": model_name, "base_checkpoint": str(args.checkpoint.resolve())}, default=str, indent=2), encoding="utf-8")
    while step < args.max_steps:
        total_samples = replay.refresh()
        if total_samples > previous_samples:
            update_budget += (total_samples - previous_samples) * args.updates_per_new_sample
            previous_samples = total_samples
            print(f"[replay] samples={total_samples} update_budget={update_budget:.1f}", flush=True)
        if total_samples < args.warmup_samples or update_budget < 1.0:
            time.sleep(args.poll_seconds)
            continue
        train_indices, val_indices = split_indices(replay, args.validation_fraction)
        if chunk_updates_left <= 0 or active_chunk is None:
            active_chunk = replay.choose_chunk_indices(train_indices, rng)
            chunk_updates_left = max(1, args.updates_per_chunk)
        selected = rng.choice(active_chunk, size=args.batch_size, replace=len(active_chunk) < args.batch_size).astype(np.int64, copy=False)
        chunk_updates_left -= 1
        batch = build_training_batch_torch(replay.load_raw_batch(selected), device=device, noise_std_m=args.noise_std_m, dropout_probability=args.dropout_probability)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            output = model(front_input=batch["front_input"], rear_input=batch["rear_input"], front_target=batch["front_target"], rear_target=batch["rear_target"])
            recon, recon_metrics = reconstruction_loss(output, batch, args.near_pixel_weight, args.near_threshold_m)
            aug = build_training_batch_torch(replay.load_raw_batch(selected), device=device, noise_std_m=args.noise_std_m, dropout_probability=args.dropout_probability)
            latent_aug = model.extract_latents(front_input=aug["front_input"], rear_input=aug["rear_input"])["fused_latent"]
            consistency = F.l1_loss(output["fused_latent"], latent_aug)
            vq = output.get("aux_losses", {}).get("vq", recon.new_zeros(()))
            loss = recon + args.consistency_weight * consistency + args.vq_weight * vq
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        update_budget -= 1.0
        metrics = {"loss": float(loss.detach()), "reconstruction": float(recon.detach()), "consistency_l1": float(consistency.detach()), "vq": float(vq.detach()), "gradient_norm": float(grad_norm), **recon_metrics}
        if step == 1 or step % 10 == 0:
            print(f"[train] step={step} samples={total_samples} loss={metrics['loss']:.4f} recon={metrics['reconstruction']:.4f} consistency={metrics['consistency_l1']:.4f} vq={metrics['vq']:.4f} grad={metrics['gradient_norm']:.3f}", flush=True)
        if step % args.validate_every == 0 or step == args.max_steps:
            val = validate(model, replay, val_indices, args, device, rng)
            metrics.update({f"val_{key}": value for key, value in val.items()})
            print(f"[val] step={step} front_l1_m={val['front_l1_m']:.4f} rear_l1_m={val['rear_l1_m']:.4f} consistency={val['consistency_l1']:.4f} finite={val['finite_ratio']:.4f}", flush=True)
            score = val["front_l1_m"] + val["rear_l1_m"]
            if score < best_val:
                best_val = score
                save_checkpoint(args.output_dir.expanduser().resolve() / "best.pt", model, optimizer, step, model_name, base_payload, args, metrics)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.output_dir.expanduser().resolve() / "latest.pt", model, optimizer, step, model_name, base_payload, args, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
