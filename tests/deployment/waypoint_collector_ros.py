"""ROS2 transport smoke for the waypoint collector services."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import yaml

from deployment.waypoints.ros2_collector import WaypointCollectorNode


def main() -> None:
    import rclpy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String
    from std_srvs.srv import Trigger

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "deployment/config/waypoint_collection.yaml").read_text(encoding="utf-8")
    )
    config["collection"]["save_lidar_observations"] = False
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "route.yaml"
        rclpy.init()
        collector_node = rclpy.create_node("waypoint_collector_smoke")
        source_node = rclpy.create_node("waypoint_collector_smoke_source")
        runtime = WaypointCollectorNode(
            collector_node, config, output=output, resume=False, interactive=False
        )
        health_pub = source_node.create_publisher(
            Bool, config["topics"]["localization_healthy"], 10
        )
        diagnostic_pub = source_node.create_publisher(
            String, config["topics"]["diagnostics"], 10
        )
        odometry_pub = source_node.create_publisher(
            Odometry, config["topics"]["odometry"], 50
        )
        gamepad_pub = source_node.create_publisher(
            String, config["topics"]["gamepad_key"], 10
        )
        client = source_node.create_client(Trigger, config["topics"]["mark_service"])
        deadline = time.monotonic() + 3.0
        while not client.wait_for_service(timeout_sec=0.05):
            rclpy.spin_once(collector_node, timeout_sec=0.01)
            if time.monotonic() > deadline:
                raise RuntimeError("mark service was not created")

        def publish_stationary_window(x: float) -> None:
            for index in range(10):
                healthy = Bool()
                healthy.data = True
                health_pub.publish(healthy)

                diagnostic = String()
                diagnostic.data = (
                    '{"name":"s10_local_navigation","state":"GOOD","values":'
                    '{"covariance_trace":"0.2","support_height_map_m":"0.0",'
                    '"icp_accepted":"true"}}'
                )
                diagnostic_pub.publish(diagnostic)

                odometry = Odometry()
                odometry.header.stamp = source_node.get_clock().now().to_msg()
                odometry.pose.pose.position.x = x + index * 0.0005
                odometry.pose.pose.position.z = 0.425
                odometry.pose.pose.orientation.w = 1.0
                odometry_pub.publish(odometry)
                for _ in range(4):
                    rclpy.spin_once(collector_node, timeout_sec=0.01)
                    rclpy.spin_once(source_node, timeout_sec=0.01)
                time.sleep(0.05)

        publish_stationary_window(0.0)

        # B cannot create a route before A has recorded the start point.
        key = String()
        key.data = "G12_KEY_B"
        gamepad_pub.publish(key)
        for _ in range(5):
            rclpy.spin_once(collector_node, timeout_sec=0.02)
            rclpy.spin_once(source_node, timeout_sec=0.02)
        assert not output.exists()

        key.data = "G12_KEY_A"
        gamepad_pub.publish(key)
        deadline = time.monotonic() + 2.0
        while not output.exists() and time.monotonic() < deadline:
            rclpy.spin_once(collector_node, timeout_sec=0.02)
            rclpy.spin_once(source_node, timeout_sec=0.02)
        route = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert len(route["nodes"]) == 1
        assert route["nodes"][0]["name"] == "start"
        assert route["nodes"][0]["tags"] == ["start"]
        audit_lines = output.with_suffix(".gamepad_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        audit = [yaml.safe_load(line) for line in audit_lines]
        assert any(item["key"] == "G12_KEY_A" and item["phase"] == "accepted" for item in audit)
        assert all("latest_pose" in item for item in audit if item["phase"] == "received")

        time.sleep(config["collection"]["window_s"] + 0.05)
        publish_stationary_window(0.5)
        key.data = "G12_KEY_B"
        gamepad_pub.publish(key)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(collector_node, timeout_sec=0.02)
            rclpy.spin_once(source_node, timeout_sec=0.02)
            route = yaml.safe_load(output.read_text(encoding="utf-8"))
            if len(route["nodes"]) == 2:
                break
        assert len(route["nodes"]) == 2

        # The service remains available as the recovery/control-room path.
        time.sleep(config["collection"]["window_s"] + 0.05)
        publish_stationary_window(1.0)
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(collector_node, timeout_sec=0.02)
            rclpy.spin_once(source_node, timeout_sec=0.02)
        response = future.result()
        assert response is not None and response.success, response.message if response else None
        route = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert len(route["nodes"]) == 3
        assert route["nodes"][0]["height_source"] == "base_height_minus_calibrated_clearance"
        assert route["nodes"][1]["height_source"] == "local_lidar_support_plane"

        # A after the route has started saves the route and requests a clean exit.
        time.sleep(config["collection"]["gamepad_debounce_s"] + 0.05)
        key.data = "G12_KEY_A"
        gamepad_pub.publish(key)
        deadline = time.monotonic() + 2.0
        while not runtime.exit_requested and time.monotonic() < deadline:
            rclpy.spin_once(collector_node, timeout_sec=0.02)
            rclpy.spin_once(source_node, timeout_sec=0.02)
        assert runtime.exit_requested
        route = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert len(route["nodes"]) == 3
        audit_lines = output.with_suffix(".gamepad_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        audit = [yaml.safe_load(line) for line in audit_lines]
        assert any(
            item["key"] == "G12_KEY_A"
            and item["phase"] == "accepted"
            and "exit requested" in item["detail"]
            for item in audit
        )
        print("WAYPOINT_COLLECTOR_ROS_OK", response.message, flush=True)

        collector_node.destroy_node()
        source_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
