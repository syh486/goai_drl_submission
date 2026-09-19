"""ROS2 waypoint collector driven by onboard KISS/ESKF odometry only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import sys
import threading
import time

import numpy as np
import yaml

from deployment.waypoints.lidar_snapshots import LidarSnapshotFrame, WaypointLidarSnapshotStore
from deployment.navigation.ros2_node import REPO_ROOT, _read_cloud, _stamp_seconds
from deployment.waypoints.collection import (
    CollectionLimits,
    CollectionRejected,
    PoseSample,
    WaypointCollectionSession,
)


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value


class WaypointCollectorNode:
    def __init__(self, node, config: dict, *, output: Path, resume: bool, interactive: bool):
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Bool, Int32, String
        from std_srvs.srv import Trigger

        self.node = node
        self.String = String
        self.Int32 = Int32
        collection = config.get("collection", {})
        self.session = WaypointCollectionSession(
            output,
            limits=CollectionLimits.from_config(collection),
            description=str(collection.get("description", "route collected onboard")),
            resume=resume,
            anchor_path=Path(collection["anchor_file"]) if collection.get("anchor_file") else None,
        )
        topics = config["topics"]
        self.lidar_store = None
        self.lidar_capture_period_s = 0.0
        self.last_lidar_pair_capture_s = float("-inf")
        self.lidar_pending_messages = {"front": None, "rear": None}
        self.max_lidar_pair_skew_s = float(
            collection.get("max_lidar_pair_skew_s", 0.03)
        )
        if bool(collection.get("save_lidar_observations", True)):
            capture_hz = float(collection.get("lidar_snapshot_capture_hz", 4.0))
            if capture_hz <= 0.0:
                raise ValueError("lidar_snapshot_capture_hz must be positive")
            self.lidar_capture_period_s = 1.0 / capture_hz
            self.lidar_store = WaypointLidarSnapshotStore(
                output,
                pairs_per_waypoint=int(collection.get("lidar_pairs_per_waypoint", 3)),
                max_pair_skew_s=float(collection.get("max_lidar_pair_skew_s", 0.03)),
                max_age_s=float(collection.get("max_lidar_observation_age_s", 1.5)),
            )
        self.health = False
        self.quality_state = "NO_DIAGNOSTICS"
        self.diagnostics: dict[str, object] = {}
        self.last_health_receipt_s = 0.0
        self.gamepad_enabled = bool(collection.get("gamepad_controls", True))
        self.gamepad_start_key = str(collection.get("gamepad_start_key", "G12_KEY_A"))
        self.gamepad_mark_key = str(collection.get("gamepad_mark_key", "G12_KEY_B"))
        self.gamepad_debounce_s = float(collection.get("gamepad_debounce_s", 0.6))
        self.gamepad_retry_timeout_s = float(
            collection.get("gamepad_retry_timeout_s", 8.0)
        )
        self.gamepad_retry_interval_s = float(
            collection.get("gamepad_retry_interval_s", 0.25)
        )
        self.gamepad_last_key_s: dict[str, float] = {}
        self.gamepad_pending: tuple[str, str, float] | None = None
        self.gamepad_next_attempt_s = 0.0
        self.gamepad_audit_path = self.session.output_path.with_suffix(
            ".gamepad_events.jsonl"
        )
        self.exit_requested = False
        self.command_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.status_pub = node.create_publisher(String, topics["collector_status"], 10)
        self.count_pub = node.create_publisher(Int32, topics["collector_count"], 10)
        node.create_subscription(Odometry, topics["odometry"], self._odometry, 50)
        if self.lidar_store is not None:
            node.create_subscription(
                PointCloud2,
                topics["front_cloud"],
                lambda message: self._cloud("front", message),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                PointCloud2,
                topics["rear_cloud"],
                lambda message: self._cloud("rear", message),
                qos_profile_sensor_data,
            )
        node.create_subscription(Bool, topics["localization_healthy"], self._health, 10)
        node.create_subscription(String, topics["diagnostics"], self._diagnostic, 10)
        node.create_subscription(String, topics["collector_command"], self._command_message, 10)
        if self.gamepad_enabled:
            node.create_subscription(String, topics["gamepad_key"], self._gamepad_key, 10)
        node.create_service(Trigger, topics["mark_service"], self._mark_service)
        node.create_service(Trigger, topics["undo_service"], self._undo_service)
        node.create_service(Trigger, topics["save_service"], self._save_service)
        node.create_timer(0.1, self._drain_commands)
        node.create_timer(1.0, self._publish_status)
        if interactive:
            threading.Thread(target=self._terminal_loop, daemon=True).start()
        node.get_logger().warning(
            f"Waypoint collector cannot publish robot motion; output={output}"
        )
        if self.gamepad_enabled:
            node.get_logger().warning(
                "Gamepad collection enabled: "
                f"{self.gamepad_start_key}=mark start/finish and save, "
                f"{self.gamepad_mark_key}=mark next. "
                "Do not run the repository low-level DDS controller at the same time."
            )
        self._print_help()

    def _cloud(self, side: str, message) -> None:
        receipt_s = time.monotonic()
        stamp_s = _stamp_seconds(message.header.stamp)
        self.lidar_pending_messages[side] = (message, stamp_s, receipt_s)
        other_side = "rear" if side == "front" else "front"
        other = self.lidar_pending_messages[other_side]
        if other is None:
            return
        current = self.lidar_pending_messages[side]
        if abs(stamp_s - other[1]) > self.max_lidar_pair_skew_s:
            older_side = side if stamp_s < other[1] else other_side
            self.lidar_pending_messages[older_side] = None
            return

        pair = {
            side: current,
            other_side: other,
        }
        self.lidar_pending_messages = {"front": None, "rear": None}
        if receipt_s - self.last_lidar_pair_capture_s < self.lidar_capture_period_s:
            return
        try:
            for pair_side in ("front", "rear"):
                pair_message, pair_stamp_s, pair_receipt_s = pair[pair_side]
                points, timestamps, rings = _read_cloud(pair_message)
                self.lidar_store.append(
                    pair_side,
                    LidarSnapshotFrame(
                        points_xyz=points,
                        point_timestamps=timestamps,
                        rings=rings,
                        stamp_s=pair_stamp_s,
                        receipt_s=pair_receipt_s,
                        frame_id=str(pair_message.header.frame_id),
                    ),
                )
            self.last_lidar_pair_capture_s = receipt_s
        except Exception as error:
            self.node.get_logger().error(f"collector LiDAR pair failed: {error}")

    def _health(self, message) -> None:
        self.health = bool(message.data)
        self.last_health_receipt_s = time.monotonic()

    def _diagnostic(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if payload.get("name") != "s10_local_navigation":
            return
        self.quality_state = str(payload.get("state", "UNKNOWN"))
        values = payload.get("values", {})
        if isinstance(values, dict):
            self.diagnostics = {
                str(key): _parse_scalar(str(value)) for key, value in values.items()
            }

    def _odometry(self, message) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        healthy = self.health and time.monotonic() - self.last_health_receipt_s <= 0.5
        self.session.append(PoseSample(
            stamp_s=_stamp_seconds(message.header.stamp),
            receipt_s=time.monotonic(),
            position=np.asarray((pose.position.x, pose.position.y, pose.position.z)),
            quaternion_wxyz=np.asarray((
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z,
            )),
            linear_velocity=np.asarray((twist.linear.x, twist.linear.y, twist.linear.z)),
            angular_velocity=np.asarray((twist.angular.x, twist.angular.y, twist.angular.z)),
            healthy=healthy,
            quality_state=self.quality_state,
            diagnostics=dict(self.diagnostics),
        ))

    def _command_message(self, message) -> None:
        self.command_queue.put(str(message.data))

    def _gamepad_key(self, message) -> None:
        key = str(message.data).strip()
        if key not in {self.gamepad_start_key, self.gamepad_mark_key}:
            return
        now_s = time.monotonic()
        previous_s = self.gamepad_last_key_s.get(key, float("-inf"))
        if now_s - previous_s < self.gamepad_debounce_s:
            return
        self.gamepad_last_key_s[key] = now_s
        self._audit_gamepad(key, "received")
        if self.gamepad_pending is not None:
            self.node.get_logger().warning(
                f"ignored {key}: another gamepad mark is still waiting for a stable pose"
            )
            return

        if key == self.gamepad_start_key:
            if self.session.nodes:
                success, message_text = self._execute("save")
                phase = "accepted" if success else "rejected"
                detail = (
                    f"{message_text}; collection exit requested"
                    if success else message_text
                )
                self._audit_gamepad(key, phase, detail)
                if success:
                    self.exit_requested = True
                    self.node.get_logger().info(f"gamepad {key}: {detail}")
                else:
                    self.node.get_logger().warning(f"gamepad {key}: {detail}")
                return
            command = "mark start start"
        else:
            if not self.session.nodes:
                self._audit_gamepad(key, "rejected", "start waypoint is missing")
                self.node.get_logger().warning(
                    f"gamepad {key}: press {self.gamepad_start_key} "
                    "at the start point first"
                )
                return
            command = "mark"

        self.gamepad_pending = (key, command, now_s + self.gamepad_retry_timeout_s)
        self.gamepad_next_attempt_s = now_s
        self.node.get_logger().info(
            f"gamepad {key}: mark requested; waiting for a stable localization window"
        )
        self._retry_gamepad()

    def _retry_gamepad(self) -> None:
        if self.gamepad_pending is None:
            return
        now_s = time.monotonic()
        if now_s < self.gamepad_next_attempt_s:
            return
        key, command, deadline_s = self.gamepad_pending
        success, message_text = self._execute(command, log_rejection=False)
        if success:
            self.gamepad_pending = None
            self._audit_gamepad(key, "accepted", message_text)
            self.node.get_logger().info(f"gamepad {key}: {message_text}")
            return
        if now_s >= deadline_s:
            self.gamepad_pending = None
            self._audit_gamepad(key, "timed_out", message_text)
            self.node.get_logger().warning(
                f"gamepad {key}: mark timed out after {self.gamepad_retry_timeout_s:.1f}s: "
                f"{message_text}"
            )
            return
        self.gamepad_next_attempt_s = now_s + self.gamepad_retry_interval_s

    def _audit_gamepad(self, key: str, phase: str, detail: str = "") -> None:
        latest = self.session.samples[-1] if self.session.samples else None
        payload: dict[str, object] = {
            "wall_time_s": time.time(),
            "monotonic_time_s": time.monotonic(),
            "key": key,
            "phase": phase,
            "detail": detail,
            "waypoint_count": len(self.session.nodes),
        }
        if latest is not None:
            payload["latest_pose"] = {
                "sensor_stamp_s": latest.stamp_s,
                "age_s": max(0.0, time.monotonic() - latest.receipt_s),
                "position": latest.position.tolist(),
                "quaternion_wxyz": latest.quaternion_wxyz.tolist(),
                "linear_velocity": latest.linear_velocity.tolist(),
                "angular_velocity": latest.angular_velocity.tolist(),
                "healthy": latest.healthy,
                "quality_state": latest.quality_state,
                "diagnostics": latest.diagnostics,
            }
        try:
            self.gamepad_audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.gamepad_audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self.node.get_logger().error(f"cannot write gamepad audit: {error}")

    def _mark_service(self, _request, response):
        response.success, response.message = self._execute("mark")
        return response

    def _undo_service(self, _request, response):
        response.success, response.message = self._execute("undo")
        return response

    def _save_service(self, _request, response):
        response.success, response.message = self._execute("save")
        return response

    def _execute(self, command_line: str, *, log_rejection: bool = True) -> tuple[bool, str]:
        try:
            parts = shlex.split(command_line)
            if not parts:
                return False, "empty command"
            command = parts[0].lower()
            if command in {"mark", "m"}:
                lidar_pairs = (
                    self.lidar_store.prepare(time.monotonic())
                    if self.lidar_store is not None else None
                )
                arguments = [part for part in parts[1:] if part != "--force"]
                name = arguments[0] if arguments else None
                tags = arguments[1].split(",") if len(arguments) >= 2 else []
                event = self.session.mark(
                    name=name,
                    tags=tags,
                    force_close_spacing="--force" in parts[1:],
                    min_samples=(
                        self.session.limits.start_min_samples
                        if not self.session.nodes and name == "start" else None
                    ),
                )
                if lidar_pairs is not None:
                    metadata = self.lidar_store.save(event["index"], lidar_pairs)
                    self.session.attach_lidar_observation(event["index"], metadata)
                message = f"marked #{event['index']} {event['name']}"
            elif command in {"undo", "u"}:
                event = self.session.undo()
                if self.lidar_store is not None:
                    self.lidar_store.delete(event["removed_index"])
                message = f"removed #{event['removed_index']} {event['removed_name']}"
            elif command in {"save", "s"}:
                self.session.save()
                message = f"saved {len(self.session.nodes)} waypoints"
            elif command in {"status", "p"}:
                message = json.dumps(self.session.status(), ensure_ascii=False)
            elif command in {"help", "h", "?"}:
                self._print_help()
                return True, "help printed"
            else:
                return False, f"unknown command: {command}"
        except (CollectionRejected, FileExistsError, ValueError) as error:
            if log_rejection:
                self.node.get_logger().warning(f"waypoint command rejected: {error}")
            return False, str(error)
        self.node.get_logger().info(message)
        self._publish_status()
        return True, message

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                self._retry_gamepad()
                return
            self._execute(command)

    def _terminal_loop(self) -> None:
        while True:
            try:
                command = input("waypoint> ")
            except EOFError:
                return
            self.command_queue.put(command)

    @staticmethod
    def _print_help() -> None:
        print(
            "Commands: mark [name] [tag1,tag2] [--force] | undo | save | status | help",
            flush=True,
        )

    def _publish_status(self) -> None:
        status = self.session.status()
        message = self.String()
        message.data = json.dumps(status, ensure_ascii=False)
        self.status_pub.publish(message)
        count = self.Int32()
        count.data = int(status["waypoint_count"])
        self.count_pub.publish(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "deployment/config/waypoint_collection.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--anchor-file",
        type=Path,
        help="start LiDAR anchor written automatically by the localization process",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument(
        "--gamepad-controls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use G12 A/B events to mark the start/subsequent waypoints",
    )
    parser.add_argument(
        "--lidar-snapshots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save synchronized raw front/rear clouds beside each waypoint",
    )
    args, ros_args = parser.parse_known_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"collector config must be a mapping: {config_path}")
    if args.lidar_snapshots is not None:
        config.setdefault("collection", {})["save_lidar_observations"] = (
            args.lidar_snapshots
        )
    if args.gamepad_controls is not None:
        config.setdefault("collection", {})["gamepad_controls"] = args.gamepad_controls
    output = args.output or Path(config["collection"]["output_file"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if args.anchor_file is not None:
        anchor_file = args.anchor_file.expanduser()
        if not anchor_file.is_absolute():
            anchor_file = REPO_ROOT / anchor_file
        config.setdefault("collection", {})["anchor_file"] = str(anchor_file.resolve())

    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=ros_args)
    node = rclpy.create_node("s10_waypoint_collector")
    runtime = WaypointCollectorNode(
        node,
        config,
        output=output,
        resume=args.resume,
        interactive=not args.no_interactive and sys.stdin.isatty(),
    )
    try:
        while rclpy.ok() and not runtime.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        if runtime.session.nodes:
            runtime.session.save()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
