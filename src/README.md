# ROS2 And S10 SDK Integration

- `S10_sdk_deploy/`：官方 S10 状态机、低层 ONNX runner、DDS 命令接口和 dry-run 工具。
- `drdds/`：机器人 IMU、关节、遥控器和自主 `Steer` 消息。
- `dual_airy_merger/`：前后 Airy 点云合并、网络和时间戳配置。
- `rslidar_msg/`：RoboSense ROS2 消息定义。

RoboSense `rsLiDAR_sdk` 不复制进仓库，由
`deployment/scripts/hardware/setup_hardware_ros2.sh` 按固定版本下载。S10 SDK 内置的
Eigen、gamepad 和 ONNX Runtime 依赖保留，以维持官方 ARM/x86 构建方式。

