"""Remove superseded periodic checkpoints while preserving run metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


MODEL_PATTERN = re.compile(r"model_(\d+)\.pt$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--keep-latest", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.keep_latest < 1:
        raise ValueError("--keep-latest must be positive")

    root = args.root.expanduser().resolve()
    removals = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        numbered = []
        for path in run_dir.glob("model_*.pt"):
            match = MODEL_PATTERN.fullmatch(path.name)
            if match:
                numbered.append((int(match.group(1)), path))
        numbered.sort()
        removals.extend(path for _, path in numbered[:-args.keep_latest])
    total_bytes = sum(path.stat().st_size for path in removals)
    print(
        "TRAINING_LOG_PRUNE",
        {
            "root": str(root),
            "files": len(removals),
            "gib": round(total_bytes / (1024 ** 3), 3),
            "mode": "apply" if args.apply else "dry-run",
        },
        flush=True,
    )
    if args.apply:
        for path in removals:
            path.unlink()


if __name__ == "__main__":
    main()
