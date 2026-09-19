"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

import os
import time
import socket
import struct
import threading
import argparse
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from std_msgs.msg import Bool, Float32MultiArray, Float64, Int32
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData, JointDataCmd

from s10_lidar import S10LidarLiveWorker, S10LidarReplayWorker

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sru_training.s10_height_scan import MuJoCoHeightScan



MODEL_NAME = "S10"
# Get the directory of the current Python file
CURRENT_DIR = Path(__file__).resolve().parent
MJCF_DIR = (CURRENT_DIR / ".." / ".." / ".." / "S10_description" / "s10_mjcf" / "mjcf").resolve()

SCENE_XML_PATHS = {
    "track": MJCF_DIR / "S10_track.xml",
}
DEFAULT_SCENE_NAME = os.environ.get("S10_MUJOCO_SCENE", "track")
XML_PATH = str(SCENE_XML_PATHS.get(DEFAULT_SCENE_NAME, SCENE_XML_PATHS["track"]).resolve())
USE_VIEWER = os.environ.get("S10_USE_VIEWER", "1").lower() not in {"0", "false", "no"}
TRACK_VIEWER = False
DT = 0.001
RENDER_INTERVAL = 10
TRACK_BODY_NAME = "base_link"
CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -25
CAMERA_DISTANCE = 18.0
COLLISION_GEOM_GROUP = 1
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2])
TRACK_REACH_RADIUS = float(os.environ.get("S10_TRACK_REACH_RADIUS", "0.2"))
TRACK_DISTANCE_MODE = os.environ.get("S10_TRACK_DISTANCE_MODE", "xy").lower()
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
TRACK_HEIGHT_POST_PREFIX = "track_height_post_"
LIDAR_REPLAY_DIR = os.environ.get("S10_LIDAR_REPLAY_DIR", "").strip()
LIVE_LIDAR = os.environ.get("S10_LIVE_LIDAR", "0").lower() in {"1", "true", "yes"}
LIDAR_SAMPLE_HZ = float(os.environ.get("S10_LIDAR_SAMPLE_HZ", "5.0"))
LIDAR_CHUNK_SIZE = int(os.environ.get("S10_LIDAR_CHUNK_SIZE", "64"))
LIDAR_MAX_SAMPLES = int(os.environ.get("S10_LIDAR_MAX_SAMPLES", "0"))
RANDOM_COVERAGE = os.environ.get("S10_LIDAR_RANDOM_COVERAGE", "0").lower() in {"1", "true", "yes"}
RANDOM_COVERAGE_RATIO = float(os.environ.get("S10_LIDAR_RANDOM_COVERAGE_RATIO", "0.7"))
RANDOM_COVERAGE_SEED = int(os.environ.get("S10_LIDAR_RANDOM_COVERAGE_SEED", "20260812"))

# Calibaration parameters (for sim-to-real consistency)
JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-35, -145, 156, 0.,
                             35, -145, 156, 0,
                             -35, 145, -156, 0,
                             35, 145, -156, 0])
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "S10": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run S10 MuJoCo ROS2 simulation.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_XML_PATHS),
        default=DEFAULT_SCENE_NAME if DEFAULT_SCENE_NAME in SCENE_XML_PATHS else "track",
        help="Built-in MJCF scene to load. Defaults to S10_MUJOCO_SCENE or 'track'.",
    )
    parser.add_argument(
        "--xml-path",
        default=os.environ.get("S10_MUJOCO_XML"),
        help="Custom MJCF path. Overrides --scene and S10_MUJOCO_SCENE.",
    )
    parser.add_argument("--model-key", default=MODEL_NAME, help="Robot key used for initial joint pose.")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def resolve_xml_path(scene_name: str, xml_path: str | None) -> str:
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(SCENE_XML_PATHS[scene_name].resolve())


class MuJoCoSimulationNode(Node):
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_path: str = XML_PATH):

        super().__init__('mujoco_simulation')

        # 加载 MJCF
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self.timestamp = 0.0

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]  # 0..15
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for S10"

        # 初始化站立姿态
        self._set_initial_pose(model_key)
        self._init_track_progress()
        self.coverage_rng = np.random.default_rng(RANDOM_COVERAGE_SEED)
        self.coverage_episode_id = 0
        self.coverage_segment_index = 0
        self.coverage_episode_start_time = 0.0
        self.lidar_sampler = None
        self.lidar_live_worker = None
        self.height_scanner = None
        if LIDAR_REPLAY_DIR:
            self.lidar_sampler = S10LidarReplayWorker(
                xml_path,
                LIDAR_REPLAY_DIR,
                chunk_size=LIDAR_CHUNK_SIZE,
                max_samples=LIDAR_MAX_SAMPLES,
            )
            self.lidar_sample_period_steps = max(1, int(round(1.0 / max(LIDAR_SAMPLE_HZ * DT, 1.0e-6))))
            self.get_logger().info(
                f"[LIDAR] replay={LIDAR_REPLAY_DIR}, hz={LIDAR_SAMPLE_HZ:.2f}, "
                f"period_steps={self.lidar_sample_period_steps}"
            )
        elif LIVE_LIDAR:
            self.lidar_live_worker = S10LidarLiveWorker(xml_path)
            self.height_scanner = MuJoCoHeightScan(self.model)
            self.lidar_sample_period_steps = max(1, int(round(1.0 / max(LIDAR_SAMPLE_HZ * DT, 1.0e-6))))
            self.get_logger().info(
                f"[LIDAR] live in-memory mode, hz={LIDAR_SAMPLE_HZ:.2f}, "
                f"period_steps={self.lidar_sample_period_steps}"
            )

        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.get_logger().info(f"[INFO] MuJoCo MJCF loaded: {xml_path}")
        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # ROS Publishers
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)
        self.odom_pub = self.create_publisher(Odometry, '/s10/ground_truth/odom', 50)
        self.track_index_pub = self.create_publisher(Int32, '/s10/track/waypoint_index', 10)
        self.track_count_pub = self.create_publisher(Int32, '/s10/track/waypoint_count', 10)
        self.track_next_pub = self.create_publisher(PointStamped, '/s10/track/next_waypoint', 10)
        self.track_distance_pub = self.create_publisher(Float64, '/s10/track/distance_to_waypoint', 10)
        self.track_elapsed_pub = self.create_publisher(Float64, '/s10/track/elapsed_time', 10)
        self.track_complete_pub = self.create_publisher(Bool, '/s10/track/complete', 10)
        self.track_path_pub = self.create_publisher(NavPath, '/s10/track/path', 10)
        self.lidar_front_pub = self.create_publisher(Float32MultiArray, '/s10/lidar/front_raw', 2)
        self.lidar_rear_pub = self.create_publisher(Float32MultiArray, '/s10/lidar/rear_raw', 2)
        self.lidar_qpos_pub = self.create_publisher(Float32MultiArray, '/s10/lidar/root_qpos', 2)
        self.height_scan_pub = self.create_publisher(Float32MultiArray, '/s10/height_scan/raw', 2)

        # ROS Subscriber
        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            '/JOINTS_CMD',
            self._cmd_callback,
            50
        )

        # 可视化
        self.viewer = None
        if USE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]  # ,3-6 basequat，0-2 basepos
        qpos0[:3] = TRACK_START_BASE_POS
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _track_geom_index(self, name: str, prefix: str):
        if not name or not name.startswith(prefix):
            return None
        suffix = name[len(prefix):]
        index_text = suffix.split("_", 1)[0]
        if not index_text.isdigit():
            return None
        return int(index_text)

    def _find_track_geoms(self):
        waypoint_geoms = {}
        point_related_geoms = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            waypoint_index = self._track_geom_index(name, TRACK_WAYPOINT_PREFIX)
            if waypoint_index is not None:
                waypoint_geoms[waypoint_index] = geom_id
                point_related_geoms.setdefault(waypoint_index, []).append(geom_id)
                continue

            post_index = self._track_geom_index(name, TRACK_HEIGHT_POST_PREFIX)
            if post_index is not None:
                point_related_geoms.setdefault(post_index, []).append(geom_id)

        return waypoint_geoms, point_related_geoms

    def _init_track_progress(self):
        self.track_enabled = False
        self.track_complete = False
        self.track_next_index = 0
        self.track_start_time = None
        self.track_finish_time = None
        self.track_last_path_publish_time = -1.0
        self.track_waypoint_positions = np.empty((0, 3), dtype=np.float64)
        self.track_point_geom_ids = {}

        waypoint_geoms, point_related_geoms = self._find_track_geoms()
        if not waypoint_geoms:
            return

        expected_indices = list(range(max(waypoint_geoms) + 1))
        missing = [index for index in expected_indices if index not in waypoint_geoms]
        if missing:
            self.get_logger().warn(f"Track progress disabled; missing waypoint geoms: {missing}")
            return

        self.track_waypoint_geom_ids = [waypoint_geoms[index] for index in expected_indices]
        self.track_point_geom_ids = {
            index: point_related_geoms.get(index, [waypoint_geoms[index]])
            for index in expected_indices
        }
        self.track_waypoint_positions = np.array(
            [self.data.geom_xpos[geom_id].copy() for geom_id in self.track_waypoint_geom_ids],
            dtype=np.float64,
        )
        self.track_original_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy()
            for geom_ids in self.track_point_geom_ids.values()
            for geom_id in geom_ids
        }
        self.track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
        if self.track_body_id < 0:
            self.get_logger().warn(f"Track progress disabled; cannot find body '{TRACK_BODY_NAME}'")
            return

        self.track_enabled = True
        self.get_logger().info(
            f"[INFO] Track progress enabled: {len(self.track_waypoint_positions)} waypoints, "
            f"radius={TRACK_REACH_RADIUS:.3f}m, distance_mode={TRACK_DISTANCE_MODE}"
        )

    def _sample_coverage_pose(self):
        """Sample a virtual LiDAR pose so coverage does not depend on route completion."""
        if not self.track_enabled or len(self.track_waypoint_positions) < 2:
            return None
        segment = int(self.coverage_rng.integers(0, len(self.track_waypoint_positions) - 1))
        start = self.track_waypoint_positions[segment]
        end = self.track_waypoint_positions[segment + 1]
        fraction = float(self.coverage_rng.uniform(0.15, 0.85))
        position = start + fraction * (end - start)
        tangent = end[:2] - start[:2]
        if float(np.linalg.norm(tangent)) < 1.0e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float64)
        tangent_yaw = float(np.arctan2(tangent[1], tangent[0]))
        yaw = tangent_yaw + float(self.coverage_rng.uniform(-np.pi, np.pi))

        self.coverage_segment_index = segment
        self.coverage_episode_id += 1
        self.coverage_episode_start_time = self.timestamp
        root_pos = np.asarray((position[0], position[1], position[2] + 0.2), dtype=np.float64)
        root_quat = np.asarray((np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)), dtype=np.float64)
        return root_pos, root_quat

    def _hide_track_point(self, waypoint_index: int):
        for geom_id in self.track_point_geom_ids.get(waypoint_index, []):
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _track_distance(self, robot_pos: np.ndarray, waypoint_pos: np.ndarray) -> float:
        if TRACK_DISTANCE_MODE == "xyz":
            return float(np.linalg.norm(robot_pos - waypoint_pos))
        return float(np.linalg.norm(robot_pos[:2] - waypoint_pos[:2]))

    def _update_track_progress(self):
        if not self.track_enabled or self.track_complete:
            return
        if self.track_next_index >= len(self.track_waypoint_positions):
            return

        robot_pos = self.data.xpos[self.track_body_id]
        waypoint_pos = self.track_waypoint_positions[self.track_next_index]
        distance = self._track_distance(robot_pos, waypoint_pos)
        if distance > TRACK_REACH_RADIUS:
            return

        reached_index = self.track_next_index
        self._hide_track_point(reached_index)

        if reached_index == 0 and self.track_start_time is None:
            self.track_start_time = self.timestamp
            self.get_logger().info(
                f"[TRACK] Timer started at waypoint 0, sim_time={self.track_start_time:.3f}s"
            )
        else:
            self.get_logger().info(
                f"[TRACK] Reached waypoint {reached_index}, sim_time={self.timestamp:.3f}s, "
                f"distance={distance:.3f}m"
            )

        self.track_next_index += 1
        if self.track_next_index >= len(self.track_waypoint_positions):
            self.track_complete = True
            self.track_finish_time = self.timestamp
            elapsed = 0.0 if self.track_start_time is None else self.track_finish_time - self.track_start_time
            self.get_logger().info(
                f"[TRACK] Final waypoint reached. Timer stopped at sim_time={self.track_finish_time:.3f}s, "
                f"elapsed={elapsed:.3f}s"
            )

    def _publish_track_status(self):
        if not self.track_enabled:
            return

        stamp = Time()
        sec = int(self.timestamp)
        stamp.sec = sec
        stamp.nanosec = int((self.timestamp - sec) * 1e9)

        index_msg = Int32()
        index_msg.data = self.track_next_index
        self.track_index_pub.publish(index_msg)

        count_msg = Int32()
        count_msg.data = len(self.track_waypoint_positions)
        self.track_count_pub.publish(count_msg)

        complete_msg = Bool()
        complete_msg.data = self.track_complete
        self.track_complete_pub.publish(complete_msg)

        elapsed_msg = Float64()
        if self.track_start_time is None:
            elapsed_msg.data = 0.0
        elif self.track_complete:
            elapsed_msg.data = self.track_finish_time - self.track_start_time
        else:
            elapsed_msg.data = self.timestamp - self.track_start_time
        self.track_elapsed_pub.publish(elapsed_msg)

        distance_msg = Float64()
        if self.track_complete:
            distance_msg.data = 0.0
        else:
            robot_pos = self.data.xpos[self.track_body_id]
            waypoint_pos = self.track_waypoint_positions[self.track_next_index]
            distance_msg.data = self._track_distance(robot_pos, waypoint_pos)

            next_msg = PointStamped()
            next_msg.header.stamp = stamp
            next_msg.header.frame_id = "world"
            next_msg.point.x = float(waypoint_pos[0])
            next_msg.point.y = float(waypoint_pos[1])
            next_msg.point.z = float(waypoint_pos[2])
            self.track_next_pub.publish(next_msg)
        self.track_distance_pub.publish(distance_msg)

        if self.timestamp - self.track_last_path_publish_time >= 1.0:
            path_msg = NavPath()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = "world"
            for waypoint_pos in self.track_waypoint_positions:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(waypoint_pos[0])
                pose.pose.position.y = float(waypoint_pos[1])
                pose.pose.position.z = float(waypoint_pos[2])
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.track_path_pub.publish(path_msg)
            self.track_last_path_publish_time = self.timestamp

    def _configure_viewer(self):
        with self.viewer.lock():
            track_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                TRACK_BODY_NAME,
            )
            if TRACK_VIEWER and track_body_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = track_body_id
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.lookat[:] = self.data.qpos[:3]

                if TRACK_VIEWER:
                    self.get_logger().warn(
                        f"Cannot find body '{TRACK_BODY_NAME}'; viewer camera tracking disabled"
                    )

            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.azimuth = CAMERA_AZIMUTH
            self.viewer.cam.elevation = CAMERA_ELEVATION
            self.viewer.cam.distance = CAMERA_DISTANCE

            if COLLISION_GEOM_GROUP < len(self.viewer.opt.geomgroup):
                self.viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0

    def _cmd_callback(self, msg: JointsDataCmd):
        """Convert received (published) positions/velocities to internal (raw)"""
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque  # tau_ff no processing

        # Convert: raw = published * dir + offset_rad
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        # 主模拟循环
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                # 控制律
                self._apply_joint_torque()
                # 模拟一步
                mujoco.mj_step(self.model, self.data)

                self.timestamp = step * DT
                self._update_track_progress()

                if self.lidar_sampler is not None and step % self.lidar_sample_period_steps == 0:
                    use_random_pose = RANDOM_COVERAGE and self.coverage_rng.random() < RANDOM_COVERAGE_RATIO
                    coverage_pose = self._sample_coverage_pose() if use_random_pose else None
                    if coverage_pose is None:
                        self.lidar_sampler.submit(
                            self.data.qpos,
                            self.data.qpos[:3],
                            self.data.qpos[3:7],
                            episode_id=self.coverage_episode_id,
                            segment_index=self.coverage_segment_index,
                            sim_time=self.timestamp,
                        )
                    else:
                        self.lidar_sampler.submit(
                            self.data.qpos,
                            coverage_pose[0],
                            coverage_pose[1],
                            episode_id=self.coverage_episode_id,
                            segment_index=self.coverage_segment_index,
                            sim_time=self.timestamp,
                        )
                    if self.lidar_sampler.reached_limit:
                        self.get_logger().info(
                            f"[LIDAR] reached max samples={self.lidar_sampler.total_samples}; stopping replay capture."
                        )
                        self.lidar_sampler.close()
                        self.lidar_sampler = None

                if self.lidar_live_worker is not None and step % self.lidar_sample_period_steps == 0:
                    self.lidar_live_worker.submit(self.data.qpos)
                    live_frame = self.lidar_live_worker.get_latest()
                    if live_frame is not None:
                        front_raw, rear_raw, root_qpos = live_frame
                        self._publish_array(self.lidar_front_pub, front_raw)
                        self._publish_array(self.lidar_rear_pub, rear_raw)
                        self._publish_array(self.lidar_qpos_pub, root_qpos)
                        if self.height_scanner is not None:
                            self._publish_array(self.height_scan_pub, self.height_scanner.raw_scan_pose(self.data, root_qpos))

                # 采样 & 发送观测 (every 5 steps for 200 Hz)
                if step % 5 == 0:
                    self._publish_robot_state(step)

                # Publish navigation-facing track state at 20 Hz.
                if step % 50 == 0:
                    self._publish_track_status()

                # 可视化
                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.sync()

            # Handle ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.0)

    def destroy_node(self):
        if self.lidar_sampler is not None:
            self.lidar_sampler.close()
        if self.lidar_live_worker is not None:
            self.lidar_live_worker.close()
        return super().destroy_node()

    @staticmethod
    def _publish_array(publisher, array: np.ndarray) -> None:
        message = Float32MultiArray()
        message.data = np.asarray(array, dtype=np.float32).reshape(-1).tolist()
        publisher.publish(message)

    def _apply_joint_torque(self):
        # 当前关节状态
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        requested_torque = (
                self.kp_cmd * (self.pos_cmd - q) +
                self.kd_cmd * (self.vel_cmd - dq) +
                self.tau_ff
        )

        # Keep abnormal PD values from overflowing DDS messages when the
        # policy loses balance. MuJoCo defines per-actuator control limits in
        # the MJCF; apply the same limits before writing data.ctrl.
        requested_torque = np.nan_to_num(
            requested_torque,
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )
        ctrl_range = self.model.actuator_ctrlrange[:self.dof_num]
        self.input_tq = np.clip(
            requested_torque,
            ctrl_range[:, 0].reshape(-1, 1),
            ctrl_range[:, 1].reshape(-1, 1),
        )

        # 写入 control 缓冲区
        self.data.ctrl[:] = self.input_tq.flatten()

    # --------------------------------------------------------
    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = q

        # roll (X-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # pitch (Y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # 防止数值漂移导致 |t2|>1
        pitch = np.arcsin(t2)

        # yaw (Z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    # --------------------------------------------------------

    def _publish_robot_state(self, step: int):
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec

        # ----- IMU -----
        q_world = self.data.sensordata[:4]  # quaternion (w, x, y, z) in MuJoCo convention
        rpy_rad = self.quaternion_to_euler(q_world)  # returns [roll, pitch, yaw] in radians

        # Convert to degrees
        rpy_deg = [angle * (180.0 / 3.141592653589793) for angle in rpy_rad]

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]  # body frame

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        # ----- 关节 -----
        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        # Convert raw to published: published = (raw - offset_rad) * dir
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR  # Torque also needs direction flip
        dds_float_max = np.finfo(np.float32).max
        pub_pos = np.clip(
            np.nan_to_num(pub_pos, nan=0.0, posinf=dds_float_max, neginf=-dds_float_max),
            -dds_float_max,
            dds_float_max,
        )
        pub_vel = np.clip(
            np.nan_to_num(pub_vel, nan=0.0, posinf=dds_float_max, neginf=-dds_float_max),
            -dds_float_max,
            dds_float_max,
        )
        pub_tau = np.clip(
            np.nan_to_num(pub_tau, nan=0.0, posinf=dds_float_max, neginf=-dds_float_max),
            -dds_float_max,
            dds_float_max,
        )

        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]  # Dummy name (four spaces)
            joint.data_id = 0  # Dummy
            joint.status_word = 1  # Normal
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0  # Dummy normal temp
            joint.driver_temp = 45.0  # Dummy normal temp
        self.joints_pub.publish(joints_msg)

        # ----- Ground-truth odometry -----
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = TRACK_BODY_NAME
        odom_msg.pose.pose.position.x = float(self.data.qpos[0])
        odom_msg.pose.pose.position.y = float(self.data.qpos[1])
        odom_msg.pose.pose.position.z = float(self.data.qpos[2])
        odom_msg.pose.pose.orientation.w = float(self.data.qpos[3])
        odom_msg.pose.pose.orientation.x = float(self.data.qpos[4])
        odom_msg.pose.pose.orientation.y = float(self.data.qpos[5])
        odom_msg.pose.pose.orientation.z = float(self.data.qpos[6])
        odom_msg.twist.twist.linear.x = float(self.data.qvel[0])
        odom_msg.twist.twist.linear.y = float(self.data.qvel[1])
        odom_msg.twist.twist.linear.z = float(self.data.qvel[2])
        odom_msg.twist.twist.angular.x = float(self.data.qvel[3])
        odom_msg.twist.twist.angular.y = float(self.data.qvel[4])
        odom_msg.twist.twist.angular.z = float(self.data.qvel[5])
        self.odom_pub.publish(odom_msg)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cli_args, ros_args = parse_cli_args()
    rclpy.init(args=ros_args)
    sim_node = MuJoCoSimulationNode(
        model_key=cli_args.model_key,
        xml_path=resolve_xml_path(cli_args.scene, cli_args.xml_path),
    )
    try:
        sim_node.start()
    except KeyboardInterrupt:
        pass
    finally:
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
