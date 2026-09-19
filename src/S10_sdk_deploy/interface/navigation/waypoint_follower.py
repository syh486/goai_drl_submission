#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int32


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('s10_waypoint_follower')

        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('max_forward_speed', 0.35)
        self.declare_parameter('min_forward_speed', 0.10)
        self.declare_parameter('max_yaw_rate', 0.70)
        self.declare_parameter('yaw_kp', 1.4)
        self.declare_parameter('turn_in_place_angle', 0.75)
        self.declare_parameter('slowdown_distance', 1.2)
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('target_timeout', 0.5)
        self.declare_parameter('progress_timeout', 6.0)
        self.declare_parameter('progress_epsilon', 0.15)
        self.declare_parameter('recovery_speed', 1.00)
        self.declare_parameter('recovery_duration', 4.0)
        # Keep recovery active until the official waypoint hit radius; a larger
        # value cancels the terrain boost before the robot can cross a lip.
        self.declare_parameter('recovery_stop_distance', 0.20)
        self.declare_parameter('forward_sign', 1.0)
        self.declare_parameter('yaw_sign', 1.0)

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.max_forward = float(self.get_parameter('max_forward_speed').value)
        self.min_forward = float(self.get_parameter('min_forward_speed').value)
        self.max_yaw = float(self.get_parameter('max_yaw_rate').value)
        self.yaw_kp = float(self.get_parameter('yaw_kp').value)
        self.turn_in_place_angle = float(self.get_parameter('turn_in_place_angle').value)
        self.slowdown_distance = float(self.get_parameter('slowdown_distance').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.target_timeout = float(self.get_parameter('target_timeout').value)
        self.progress_timeout = float(self.get_parameter('progress_timeout').value)
        self.progress_epsilon = float(self.get_parameter('progress_epsilon').value)
        self.recovery_speed = float(self.get_parameter('recovery_speed').value)
        self.recovery_duration = float(self.get_parameter('recovery_duration').value)
        self.recovery_stop_distance = float(
            self.get_parameter('recovery_stop_distance').value)
        self.forward_sign = float(self.get_parameter('forward_sign').value)
        self.yaw_sign = float(self.get_parameter('yaw_sign').value)

        if self.control_rate <= 0.0:
            raise ValueError('control_rate must be positive')
        if self.max_forward <= 0.0 or self.max_yaw <= 0.0:
            raise ValueError('maximum speeds must be positive')
        if self.slowdown_distance <= 0.0:
            raise ValueError('slowdown_distance must be positive')
        if self.progress_timeout <= 0.0 or self.progress_epsilon <= 0.0:
            raise ValueError('progress thresholds must be positive')
        if (
            self.recovery_speed < self.max_forward
            or self.recovery_duration <= 0.0
            or self.recovery_stop_distance <= 0.0
        ):
            raise ValueError('recovery settings are inconsistent with normal control')

        self.position = None
        self.yaw = 0.0
        self.target = None
        self.waypoint_index = -1
        self.complete = False
        self.last_odom_time = 0.0
        self.last_target_time = 0.0
        self.last_report_time = 0.0
        self.reported_complete = False
        self.progress_reference_distance = None
        self.progress_reference_time = 0.0
        self.recovery_until = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.heading_error_pub = self.create_publisher(Float64, '/s10/follower/heading_error', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/s10/ground_truth/odom', self._odom_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/s10/track/next_waypoint', self._target_callback, 10)
        self.index_sub = self.create_subscription(
            Int32, '/s10/track/waypoint_index', self._index_callback, 10)
        self.complete_sub = self.create_subscription(
            Bool, '/s10/track/complete', self._complete_callback, 10)
        self.timer = self.create_timer(1.0 / self.control_rate, self._control_step)

        self.get_logger().info(
            f'Waypoint follower ready: max_forward={self.max_forward:.2f} m/s '
            f'max_yaw={self.max_yaw:.2f} rad/s'
        )

    def _odom_callback(self, msg: Odometry):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation
        self.yaw = quaternion_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w)
        self.last_odom_time = time.monotonic()

    def _target_callback(self, msg: PointStamped):
        self.target = (msg.point.x, msg.point.y)
        self.last_target_time = time.monotonic()

    def _index_callback(self, msg: Int32):
        if msg.data != self.waypoint_index:
            self.waypoint_index = msg.data
            self.progress_reference_distance = None
            self.recovery_until = 0.0
            self.get_logger().info(f'Tracking waypoint {self.waypoint_index}')

    def _complete_callback(self, msg: Bool):
        self.complete = msg.data

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _control_step(self):
        now = time.monotonic()
        if self.complete:
            self._publish_stop()
            if not self.reported_complete:
                self.get_logger().info('Track complete; follower stopped')
                self.reported_complete = True
            return

        data_stale = (
            self.position is None
            or self.target is None
            or now - self.last_odom_time > self.odom_timeout
            or now - self.last_target_time > self.target_timeout
        )
        if data_stale:
            self._publish_stop()
            return

        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(desired_yaw - self.yaw)

        yaw_rate = max(-self.max_yaw, min(self.max_yaw, self.yaw_kp * yaw_error))
        if abs(yaw_error) >= self.turn_in_place_angle:
            forward = 0.0
        else:
            distance_scale = min(1.0, distance / self.slowdown_distance)
            forward = max(self.min_forward, self.max_forward * distance_scale)
            forward *= max(0.0, math.cos(yaw_error))

        if self.progress_reference_distance is None:
            self.progress_reference_distance = distance
            self.progress_reference_time = now
        elif distance <= self.progress_reference_distance - self.progress_epsilon:
            self.progress_reference_distance = distance
            self.progress_reference_time = now
        elif (
            self.waypoint_index > 0
            and abs(yaw_error) < 0.25
            and now - self.progress_reference_time >= self.progress_timeout
        ):
            self.recovery_until = now + self.recovery_duration
            self.progress_reference_distance = distance
            self.progress_reference_time = now
            self.get_logger().warning(
                f'No waypoint progress for {self.progress_timeout:.1f}s; '
                f'boosting forward speed to {self.recovery_speed:.2f} m/s'
            )

        if distance <= self.recovery_stop_distance:
            self.recovery_until = 0.0
        elif now < self.recovery_until and abs(yaw_error) < self.turn_in_place_angle:
            forward = max(forward, self.recovery_speed)

        cmd = Twist()
        cmd.linear.x = self.forward_sign * forward
        cmd.angular.z = self.yaw_sign * yaw_rate
        self.cmd_pub.publish(cmd)

        error_msg = Float64()
        error_msg.data = yaw_error
        self.heading_error_pub.publish(error_msg)

        if now - self.last_report_time >= 2.0:
            self.get_logger().info(
                f'waypoint={self.waypoint_index} distance={distance:.2f} '
                f'yaw_error={yaw_error:.2f} '
                f'cmd=({cmd.linear.x:.2f}, {cmd.angular.z:.2f})'
            )
            self.last_report_time = now


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
