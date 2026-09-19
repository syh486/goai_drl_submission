# Deployment Entry Points

- `hardware/`：ROS2 环境、话题检查、Airy 驱动、AGX 同步。
- `mapping/`：记录、GLIM、回环、多圈构图和 held-out 评测。
- `localization/`：只读定位、运行前检查和 detached 进程管理。
- `waypoints/`：路线采集、A/B 按键流程和路线数据校验。
- `navigation/`：高层 ONNX、官方低层、dry-run 和完整导航栈。

所有脚本均从仓库根目录解析配置和模型，允许通过环境变量覆盖 Python、ROS2 setup、地图
目录和机器人地址。实机默认保持运动输出关闭。

