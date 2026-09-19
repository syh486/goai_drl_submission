#!/usr/bin/env python3
"""Read-only audit of S10 MuJoCo addresses against the learned-policy order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s10_policy_protocol import audit_mujoco_s10_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "locowheeledlegged/assets/s10/official/mjcf/S10.xml",
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    report = audit_mujoco_s10_protocol(model, mujoco)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("S10_MUJOCO_PROTOCOL_AUDIT=PASS")


if __name__ == "__main__":
    main()
