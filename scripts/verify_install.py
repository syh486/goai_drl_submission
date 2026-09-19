#!/usr/bin/env python3
"""Validate the maintained SRU training and mapping/localization assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import kiss_icp
import mujoco
import numpy as np
import onnxruntime
import torch
import warp
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.navigation.core import load_hardware_config
from sru_training.s10_policy_config import S10ObservationSpec


def require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_training() -> None:
    if not torch.cuda.is_available():
        message = "CUDA is not visible; SRU training and Warp LiDAR cannot run in this process"
        if os.environ.get("S10_REQUIRE_CUDA") == "1":
            raise RuntimeError(message)
        print(f"WARN {message}")
    required = (
        "checkpoints/navigation/sru_deploy_model_2750.pt",
        "checkpoints/navigation/sru_stage4_base_model_1800.pt",
        "checkpoints/navigation/sru_stage5_base_model_1499.pt",
        "checkpoints/lidar_encoder_random_terrain_ft/best.pt",
        "sru_training/assets/vae_heightscan3.pth",
        "training/configs/ppo_sru_stage4_full_no_pits_128.yaml",
        "training/configs/ppo_sru_stage5_lower_density_stairs_128.yaml",
        "training/configs/ppo_sru_stage5_final_128.yaml",
    )
    for relative in required:
        require_file(relative)

    for name in (
        "ppo_sru_stage4_full_no_pits_128.yaml",
        "ppo_sru_stage5_lower_density_stairs_128.yaml",
        "ppo_sru_stage5_final_128.yaml",
    ):
        config = yaml.safe_load((ROOT / "training/configs" / name).read_text())
        arguments = config["arguments"]
        expected = {
            "algorithm": "ppo",
            "ppo_preset": "sru",
            "task_mode": "random_goal_sru",
            "low_level_profile": "official_20260828",
            "sensor_backend": "warp",
            "lidar_horizontal_samples": 900,
        }
        mismatches = {
            key: (arguments.get(key), value)
            for key, value in expected.items()
            if arguments.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"{name}: {mismatches}")

    spec = S10ObservationSpec()
    spec.validate()
    if spec.next_waypoint_dim != 0:
        raise RuntimeError("maintained actor must not use next-waypoint context")


def validate_mapping() -> None:
    required = (
        "deployment/models/sru_policy1_model2750.onnx",
        "deployment/models/s10_lidar_encoder.onnx",
        "deployment/models/high_level_manifest.json",
        "src/S10_sdk_deploy/policy/policy.onnx",
        "deployment/config/hardware_localization.yaml",
        "deployment/config/hardware_navigation.yaml",
        "artifacts/localization/route_map_single_session/localization_map_manifest.json",
        "artifacts/localization/route_map_single_session/canonical_route_poses.npy",
        "artifacts/localization/route_map_single_session/canonical_route_trajectory.npz",
        "artifacts/localization/route_map_single_session/evidence/heldout_metric_report.json",
        (
            "artifacts/localization/route_map_multisession_candidate/"
            "localization_map_manifest.json"
        ),
        (
            "artifacts/localization/route_map_multisession_candidate/"
            "evidence/heldout_metric_report.json"
        ),
    )
    for relative in required:
        require_file(relative)

    map_root = ROOT / "artifacts/localization/route_map_single_session"
    manifest = json.loads(
        (map_root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    if int(manifest["submap_count"]) != 163:
        raise RuntimeError(f"unexpected frozen-map manifest: {manifest['submap_count']}")

    report = json.loads(
        (map_root / "evidence/heldout_metric_report.json").read_text(encoding="utf-8")
    )
    expected = {
        "evaluated_anchor_count": 41,
        "qualified_anchor_count": 37,
        "full_route_metric_accuracy_qualified": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"unexpected held-out metric {key}: {report.get(key)!r}")
    if not np.isclose(report["maximum_translation_error_m"], 0.22278357695352952):
        raise RuntimeError("frozen held-out maximum error changed")

    candidate_root = (
        ROOT / "artifacts/localization/route_map_multisession_candidate"
    )
    candidate_manifest = json.loads(
        (candidate_root / "localization_map_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_count = sum(
        int(entry["reference_session"]) == 0
        for entry in candidate_manifest["submaps"]
    )
    variant_count = sum(
        int(entry["reference_session"]) > 0
        for entry in candidate_manifest["submaps"]
    )
    if (
        int(candidate_manifest["submap_count"]) != 201
        or canonical_count != 163
        or variant_count != 38
    ):
        raise RuntimeError(
            "multi-session map mismatch: "
            f"manifest={candidate_manifest['submap_count']} "
            f"canonical={canonical_count} variants={variant_count}"
        )
    candidate_report = json.loads(
        (candidate_root / "evidence/heldout_metric_report.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_expected = {
        "evaluated_anchor_count": 41,
        "qualified_anchor_count": 41,
        "heldout_map_repeatability_qualified": True,
        "full_route_metric_accuracy_qualified": False,
    }
    for key, value in candidate_expected.items():
        if candidate_report.get(key) != value:
            raise RuntimeError(
                f"unexpected multi-session metric {key}: "
                f"{candidate_report.get(key)!r}"
            )
    if not np.isclose(
        candidate_report["maximum_translation_error_m"],
        0.09291344726627075,
    ):
        raise RuntimeError("multi-session held-out maximum error changed")

    localization = load_hardware_config(
        ROOT / "deployment/config/hardware_localization.yaml"
    )
    navigation = load_hardware_config(
        ROOT / "deployment/config/hardware_navigation.yaml"
    )
    if localization["runtime"]["enable_motion"]:
        raise RuntimeError("localization config must default to motion disabled")
    if navigation["runtime"]["enable_motion"]:
        raise RuntimeError("navigation config must default to motion disabled")
    if navigation["topics"]["steer"] != "/s10/navigation/steer":
        raise RuntimeError("autonomous steer must remain isolated from /STEER")
    for name, config in (("localization", localization), ("navigation", navigation)):
        if config["map_localization"].get(
            "multisession_consensus_enabled", False
        ):
            raise RuntimeError(
                f"{name} config must not enable multi-session consensus by default"
            )

    for model in (
        ROOT / "deployment/models/sru_policy1_model2750.onnx",
        ROOT / "deployment/models/s10_lidar_encoder.onnx",
    ):
        onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    print(
        f"PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}; "
        f"MuJoCo {mujoco.__version__}; Warp {warp.__version__}; "
        f"ONNX Runtime {onnxruntime.__version__}; KISS-ICP {kiss_icp.__version__}"
    )
    validate_training()
    validate_mapping()
    subprocess.run(
        ["sha256sum", "-c", str(ROOT / "CHECKSUMS.sha256")],
        cwd=ROOT,
        check=True,
    )
    print(
        "VERIFY_OK training=ready map_evidence=37/41 "
        "multisession_evidence=41/41 runtime_map=external"
    )


if __name__ == "__main__":
    main()
