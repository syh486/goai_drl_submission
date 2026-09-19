"""Validated storage for physically consistent waypoint entry states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1

STATE_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "onnx_last_action",
    "backend_last_cmd",
    "backend_last_action",
    "previous_high_level_action",
    "reward_joint_acc",
    "filtered_cmd",
    "filter_alpha",
    "policy_scale",
    "policy_bias",
)


@dataclass(frozen=True)
class EntryStateBankContract:
    nq: int
    nv: int
    nu: int
    dof: int
    simulation_assets_sha256: str
    low_level_checkpoint_sha256: str | None
    waypoint_sha256: str


class EntryStateBank:
    """Immutable in-memory index over a compressed ``npz`` state bank."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract: EntryStateBankContract | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"entry-state bank does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as payload:
            if "metadata_json" not in payload or "segments" not in payload:
                raise RuntimeError(f"invalid entry-state bank: {self.path}")
            self.metadata = json.loads(str(payload["metadata_json"].item()))
            self.segments = np.asarray(payload["segments"], dtype=np.int64).copy()
            self.arrays = {
                name: np.asarray(payload[name]).copy()
                for name in STATE_FIELDS
                if name in payload
            }
        self._validate(contract)
        self._indices_by_segment = {
            int(segment): np.flatnonzero(self.segments == segment)
            for segment in np.unique(self.segments)
        }

    def _validate(self, contract: EntryStateBankContract | None) -> None:
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported entry-state schema in {self.path}: "
                f"{self.metadata.get('schema_version')}"
            )
        if self.segments.ndim != 1 or not len(self.segments):
            raise RuntimeError(f"entry-state bank has no samples: {self.path}")
        missing = sorted(set(STATE_FIELDS) - set(self.arrays))
        if missing:
            raise RuntimeError(f"entry-state bank is missing fields {missing}: {self.path}")
        for name, values in self.arrays.items():
            if values.shape[0] != len(self.segments):
                raise RuntimeError(
                    f"entry-state field {name} has {values.shape[0]} rows, "
                    f"expected {len(self.segments)}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"entry-state field {name} contains non-finite values")
        if contract is None:
            return
        expected_shapes = {
            "qpos": (contract.nq,),
            "qvel": (contract.nv,),
            "ctrl": (contract.nu,),
            "onnx_last_action": (16,),
            "backend_last_cmd": (3,),
            "backend_last_action": (2,),
            "previous_high_level_action": (2,),
            "reward_joint_acc": (contract.dof,),
            "filtered_cmd": (3,),
            "filter_alpha": (3,),
            "policy_scale": (2,),
            "policy_bias": (2,),
        }
        for name, shape in expected_shapes.items():
            if self.arrays[name].shape[1:] != shape:
                raise RuntimeError(
                    f"entry-state field {name} has shape {self.arrays[name].shape[1:]}, "
                    f"expected {shape}"
                )
        for name in (
            "simulation_assets_sha256",
            "low_level_checkpoint_sha256",
            "waypoint_sha256",
        ):
            if self.metadata.get(name) != getattr(contract, name):
                raise RuntimeError(
                    f"entry-state bank {name} does not match the current backend"
                )

    @property
    def sample_count(self) -> int:
        return len(self.segments)

    @property
    def available_segments(self) -> tuple[int, ...]:
        return tuple(sorted(self._indices_by_segment))

    def counts_by_segment(self) -> dict[int, int]:
        return {
            segment: len(indices)
            for segment, indices in self._indices_by_segment.items()
        }

    def sample(self, segment: int, rng: np.random.Generator) -> dict[str, Any] | None:
        indices = self._indices_by_segment.get(int(segment))
        if indices is None or not len(indices):
            return None
        row = int(indices[int(rng.integers(0, len(indices)))])
        return {name: values[row].copy() for name, values in self.arrays.items()}


def save_entry_state_bank(
    path: str | Path,
    *,
    segments: np.ndarray,
    samples: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    """Write one atomic, compressed state-bank artifact."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(samples)
    payload["segments"] = np.asarray(segments, dtype=np.int64)
    payload["metadata_json"] = np.asarray(
        json.dumps({"schema_version": SCHEMA_VERSION, **metadata}, sort_keys=True)
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)
    return destination
