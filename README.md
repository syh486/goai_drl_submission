# S10 轮足机器人全地形自主导航

深度强化学习队面向 GOAI 2026 全地形巡检任务构建的分层自主导航系统。作者：宋逸涵。

项目同时覆盖四个相互独立、通过明确接口连接的技术模块：

1. **SRU 高层导航**：双 LiDAR、本体状态和相对目标输入，输出前进速度与偏航角速度。
2. **Hybrid VQ-AE LiDAR 编码器**：将前后 `96 x 900` 扫描压缩为空间 latent。
3. **路线建图与定位**：三圈 LiDAR-IMU 数据构建有序局部子图，并在线生成机体系相对目标。
4. **HIMLoco S10 低层**：基于本体历史和内部模型，将速度指令转换为 16 维腿轮动作。

## 成果摘要

- 高层 PPO/SRU 策略在 50 个随机地形回合中成功 36 次。
- 双 LiDAR encoder 已完成随机地形数据重训，并导出 ONNX。
- 冻结单会话路线地图的留出圈评测为 37/41 个锚点误差低于 10 cm。
- 多会话候选地图的留出圈桌面回放为 41/41 低于 10 cm，最大误差 9.29 cm。
- 高层模型、LiDAR encoder、官方低层和 ROS2/DDS 链路均形成独立可检查接口。
- AGX 实机在线定位受到点云吞吐、时间同步和丢帧限制；仓库不把离线结果表述为已经完成的全链路实机自主导航。

## 仓库结构

```text
training/                    高层 PPO/SRU 训练入口、地形、encoder 与评测
sru_training/                MuJoCo backend、SRU 网络和 rsl_rl 运行库
s10_himloco_go2w_port/       唯一保留的 S10 HIMLoco 底层训练项目
deployment/
  common/                    坐标、点云和轨迹公共工具
  mapping/                   记录、GLIM 导出、多圈优化和地图构建
  localization/              局部里程计、地图匹配、恢复和离线评测
  waypoints/                 目标点采集、快照、重绑定和校验
  navigation/                ONNX 推理、目标管理与 ROS2 节点
  scripts/                   按上述功能分类的命令行入口
src/                         S10 SDK、DDS 接口和双 Airy ROS2 包
checkpoints/                 高层导航与 LiDAR encoder 的 PyTorch 权重
deployment/models/           部署用高层和 encoder ONNX
artifacts/localization/      地图清单、优化轨迹和留出圈评测证据
tests/                       训练、地形和部署合同测试
```

完整路线子图和原始实机数据体积较大、且与比赛现场绑定，因此不放入 GitHub 主体。
运行时地图统一安装到 `deployment/maps/current/`，比赛数据产生的清单和量化证据保留在
`artifacts/localization/`。

## 高层训练

环境准备见 `environment.yml` 和 `requirements.txt`。正式配置保留 Stage 4、Stage 5、
最终 Stage 5 和 integration smoke：

```bash
./training/run_training.sh training/configs/ppo_sru_stage5_final_128.yaml
```

随机地形评测：

```bash
conda run --no-capture-output -n race \
  python -m training.evaluation.play_random_policy \
  --checkpoint checkpoints/navigation/sru_deploy_model_2750.pt \
  --terrain-profile stage5_lower_density_stairs
```

模块说明见 [training/README.md](training/README.md) 和
[sru_training/README.md](sru_training/README.md)。

## 底层训练

底层项目只保留 `s10_himloco_go2w_port/`。它以 HIMLoco-for-Go2W 为算法参考，使用
Isaac Lab 训练 S10 轮足策略，并显式固定关节顺序、动作尺度、历史状态和部署 ABI：

```bash
cd s10_himloco_go2w_port
python scripts/s10/train_go2w_him.py --num-envs 4096 --headless
```

详细协议见
[GO2W_REFERENCE_PORT_PROTOCOL.md](s10_himloco_go2w_port/docs/s10/GO2W_REFERENCE_PORT_PROTOCOL.md)。

## 建图与定位

建图主线采用 canonical、support、held-out 三圈协议。前两圈用于构图与跨圈约束，第三圈
只用于独立验收：

```bash
deployment/scripts/mapping/build_multilap_route_map.sh \
  CANONICAL_SESSION CANONICAL_GLIM \
  SUPPORT_SESSION SUPPORT_GLIM OUTPUT_ROOT

deployment/scripts/mapping/evaluate_multilap_map.sh \
  OUTPUT_ROOT/localization_map \
  HELDOUT_SESSION HELDOUT_GLIM HELDOUT_EVALUATION
```

完整方法和地图格式见 [deployment/README.md](deployment/README.md)。

## Waypoint 与导航部署

采集节点使用遥控器事件标记 waypoint，同时保存双 LiDAR 快照和连续建图记录。地图生成后，
waypoint 通过优化轨迹重绑定到 `map` 坐标系。在线定位根据
`T_map_base = T_map_odom @ T_odom_base` 生成当前机体系相对目标。

只读定位入口：

```bash
export S10_ROUTE_MAP_DIR=/absolute/path/to/localization_map
deployment/scripts/localization/run_route_map_localization.sh
```

ONNX 导航栈默认保持命令隔离：

```bash
deployment/scripts/navigation/run_onnx_navigation_stack.sh ROUTE_YAML
```

涉及实体机器人时必须先检查话题、SDK 模式和急停，并保持 `enable_motion: false` 完成
dry-run。仓库中的默认配置不会直接接管机器人。

## 模型与证据

| 内容 | 路径 |
| --- | --- |
| 最终高层 checkpoint | `checkpoints/navigation/sru_deploy_model_2750.pt` |
| LiDAR encoder checkpoint | `checkpoints/lidar_encoder_random_terrain_ft/best.pt` |
| 高层 ONNX | `deployment/models/sru_policy1_model2750.onnx` |
| Encoder ONNX | `deployment/models/s10_lidar_encoder.onnx` |
| 底层实验 checkpoint | `s10_himloco_go2w_port/checkpoints/` |
| 定位评测证据 | `artifacts/localization/` |

模型与关键证据可通过以下命令校验：

```bash
sha256sum -c CHECKSUMS.sha256
./verify_install.sh
```

## 第三方项目与生成式 AI

项目使用或参考 SRU、rsl_rl、HIMLoco、Isaac Lab、MuJoCo、KISS-ICP、GLIM、
RoboSense rsLiDAR SDK 和 Deep Robotics S10 SDK。来源、许可证和改动边界见
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。比赛期间新增的迁移、训练、
评测、定位和部署集成代码由 **GPT5.6Sol** 生成，并由团队通过仿真、离线回放和有限实机
窗口验证。
