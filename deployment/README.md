# 建图、定位与导航部署

`deployment/` 保存从传感器记录到相对目标生成的完整软件链。目录按职责拆分，Python
模块通过 `python -m deployment.<package>.<module>` 调用。

## 目录

| 目录 | 职责 |
| --- | --- |
| `common/` | 点云方向、四元数、轨迹读取和公共接口 |
| `mapping/` | 关键帧记录、GLIM 数据导出、回环、多圈优化和子图构建 |
| `localization/` | LiDAR/IMU/轮速局部传播、有序子图匹配和定位评测 |
| `waypoints/` | A/B 按键采点、LiDAR 快照、路线校验和优化后重绑定 |
| `navigation/` | ONNX 模型、相对目标、ROS2 订阅发布和命令隔离 |
| `scripts/` | 按功能分类的 shell 入口 |
| `config/` | 实机话题、外参、门控和安全默认值 |
| `models/` | 高层 SRU 与 LiDAR encoder ONNX |
| `native/` | 多圈图优化和 VGICP 的原生加速代码 |
| `maps/` | 运行时地图挂载点；仓库不内置现场完整子图 |

## 三圈建图协议

- **Canonical lap**：定义地图坐标、路线方向和有序锚点。
- **Support lap**：提供独立视角和跨圈 SE(3) 约束。
- **Held-out lap**：不参与建图，只用于因果回放和误差验收。

```text
双 LiDAR + IMU 记录
  -> ROS bag / GLIM LIO
  -> canonical 局部子图
  -> canonical-support 双向配准约束
  -> 稀疏 SE(3) 图优化
  -> 有序路线子图地图
  -> held-out 连续定位回放
```

构图和验收入口：

```bash
deployment/scripts/mapping/build_multilap_route_map.sh \
  CANONICAL_SESSION CANONICAL_GLIM \
  SUPPORT_SESSION SUPPORT_GLIM OUTPUT_ROOT

deployment/scripts/mapping/evaluate_multilap_map.sh \
  OUTPUT_ROOT/localization_map \
  HELDOUT_SESSION HELDOUT_GLIM HELDOUT_EVALUATION
```

运行时地图由 `localization_map_manifest.json`、参考轨迹和 `submaps/` 构成。完成构图后可
将地图复制或链接到 `deployment/maps/current/`，也可设置 `S10_ROUTE_MAP_DIR` 指向外部目录。

## Waypoint 采集

采集时保存路线坐标、起点 anchor、标记事件、前后 LiDAR 快照以及连续建图关键帧：

```bash
deployment/scripts/waypoints/start_route_collection.sh SESSION_NAME
deployment/scripts/waypoints/stop_route_collection.sh
```

地图优化完成后使用 `deployment.waypoints.rebind` 把标记时刻绑定到优化轨迹。导航过程中
路线管理器只向高层提供当前 waypoint，不提供后续目标上下文。

## 在线定位

局部里程计持续更新 `T_odom_base`，地图匹配间歇更新 `T_map_odom`：

```text
T_map_base = T_map_odom @ T_odom_base
```

只读启动：

```bash
export S10_ROUTE_MAP_DIR=/absolute/path/to/localization_map
deployment/scripts/localization/run_route_map_localization.sh
```

`continuous_map_localization.py` 使用路线索引限制重复结构候选，通过 fitness、RMSE、创新量、
反向一致性和连续多帧确认控制地图修正。地图观测暂时不可用时进入 coasting，只传播局部
里程计，不伪造地图匹配结果。

## 导航与安全边界

`navigation/ros2_node.py` 组合 LiDAR、IMU、关节、本地里程计、路线地图和 ONNX 推理。
默认配置满足：

- `enable_motion: false`；
- 不向工厂 `/STEER` 直接发布自主命令；
- 定位模式不创建运动 publisher；
- 地图健康、传感器时效或模型校验失败时保持命令隔离。

完整 ONNX 栈入口：

```bash
deployment/scripts/navigation/run_onnx_navigation_stack.sh ROUTE_YAML
```

## 证据边界

`artifacts/localization/` 保存比赛期间地图清单、优化轨迹和 held-out 报告，但不包含完整
子图点云。多会话候选在桌面回放达到 41/41 个有效锚点低于 10 cm；AGX 在线试验仍受
处理吞吐、同步和丢帧限制。离线重复定位结果不能替代实机在线资格。
