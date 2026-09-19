"""Control detached mapping laps from the factory G12 A/B buttons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pid_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def _qualified(path: Path) -> bool:
    report = path / "mapping_qualification.json"
    try:
        return bool(json.loads(report.read_text(encoding="utf-8"))["qualified"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


class GamepadMappingController:
    def __init__(
        self,
        node,
        *,
        topic: str,
        collection_root: Path,
        session_names: list[str],
        runtime_dir: Path,
        debounce_s: float,
    ) -> None:
        from std_msgs.msg import String

        self.node = node
        self.collection_root = collection_root.expanduser().resolve()
        self.session_names = session_names
        self.runtime_dir = runtime_dir.expanduser().resolve()
        self.debounce_s = float(debounce_s)
        self.last_key_s: dict[str, float] = {}
        self.commands: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()
        self.busy = False
        self.recording = (
            _pid_alive(self.runtime_dir / "pid")
            and _pid_alive(self.runtime_dir / "imu_pid")
        )
        self.session_index = 0
        while (
            self.session_index < len(self.session_names)
            and _qualified(self.collection_root / self.session_names[self.session_index])
        ):
            self.session_index += 1
        node.create_subscription(String, topic, self._key, 10)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self._status(
            "READY",
            "A=start current lap; B=stop, attach IMU and qualify; no motion publisher",
        )

    def _status(self, state: str, detail: str) -> None:
        payload = {
            "state": state,
            "detail": detail,
            "recording": self.recording,
            "busy": self.busy,
            "next_session": (
                self.session_names[self.session_index]
                if self.session_index < len(self.session_names) else None
            ),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        line = "S10_GAMEPAD_MAPPING " + json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        self.node.get_logger().info(line)

    def _key(self, message) -> None:
        key = str(message.data).strip()
        if key not in {"G12_KEY_A", "G12_KEY_B"}:
            return
        now_s = time.monotonic()
        if now_s - self.last_key_s.get(key, float("-inf")) < self.debounce_s:
            return
        self.last_key_s[key] = now_s
        with self.lock:
            if self.busy:
                self._status("IGNORED", f"{key}: transition already in progress")
                return
            if key == "G12_KEY_A":
                if self.recording:
                    self._status("IGNORED", "A: a mapping lap is already recording")
                    return
                if self.session_index >= len(self.session_names):
                    self._status("COMPLETE", "all configured A/B laps are already qualified")
                    return
                self.busy = True
                self.commands.put("start")
            else:
                if not self.recording:
                    self._status("IGNORED", "B: no active mapping lap")
                    return
                self.busy = True
                self.commands.put("stop")

    def _available_session_name(self) -> str:
        base = self.session_names[self.session_index]
        if not (self.collection_root / base).exists():
            return base
        return f"{base}_retry_{time.strftime('%Y%m%d_%H%M%S')}"

    def _run(self, script: str, *arguments: str) -> int:
        command = [
            str(REPO_ROOT / "deployment" / "scripts" / "mapping" / script),
            *arguments,
        ]
        self._status("RUNNING", " ".join(command))
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode

    def _worker(self) -> None:
        while True:
            command = self.commands.get()
            if command is None:
                return
            if command == "start":
                name = self._available_session_name()
                returncode = self._run("start_support_mapping_lap.sh", name)
                with self.lock:
                    self.recording = returncode == 0
                    self.busy = False
                self._status(
                    "RECORDING" if returncode == 0 else "START_FAILED",
                    f"session={name} returncode={returncode}",
                )
            elif command == "stop":
                returncode = self._run("stop_support_mapping_lap.sh")
                with self.lock:
                    self.recording = False
                    self.busy = False
                    if returncode == 0:
                        self.session_index += 1
                self._status(
                    "QUALIFIED" if returncode == 0 else "QUALIFICATION_FAILED",
                    f"returncode={returncode}",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/GAMEPAD_KEY")
    parser.add_argument(
        "--collection-root", type=Path,
        default=Path.home() / "s10_route_collection",
    )
    parser.add_argument(
        "--sessions", nargs="+", default=["canonical_lap", "support_lap"],
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=Path("/tmp/s10_support_mapping"),
    )
    parser.add_argument("--debounce-s", type=float, default=1.0)
    args, ros_args = parser.parse_known_args()

    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=ros_args)
    node = rclpy.create_node("s10_gamepad_mapping_controller")
    controller = GamepadMappingController(
        node,
        topic=args.topic,
        collection_root=args.collection_root,
        session_names=list(args.sessions),
        runtime_dir=args.runtime_dir,
        debounce_s=args.debounce_s,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        controller.commands.put(None)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
