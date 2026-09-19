# 第三方代码、作者与生成式 AI 说明

## 原始 SRU 工作

本项目使用并迁移了 Spatially-Enhanced Recurrent Units（SRU）导航模型和对应的 on-policy 训练框架。应引用：

Yang, Fan; Frivik, Per; Hoeller, David; Wang, Chen; Cadena, Cesar; Hutter, Marco. *Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning*. The International Journal of Robotics Research, 2025. DOI: `10.1177/02783649251401926`.

原仓库：`https://github.com/leggedrobotics/sru-navigation-learning`

SRU 扩展和修改由 Fan Yang 完成，建立在 Nikita Rudin、ETH Zurich 与 NVIDIA 的 `rsl_rl` 工作之上。其 BSD-3-Clause 许可证全文保存在 `docs/LICENSE_SRU_RSL_RL`。本项目未将比赛迁移与训练结果归属于原 SRU 作者。

## 比赛官方资源

S10 机器人模型、比赛场景、ROS 2 SDK 和官方低层 `policy.onnx` 来自 Deep Robotics 比赛资源仓库。根目录 `LICENSE` 及各第三方目录中的许可证继续适用于这些文件。

S10 SDK 构建链随附 Eigen、ONNX Runtime 和手柄接收代码。Eigen 与 ONNX Runtime 的
许可证和 third-party notices 保留在各自源码目录；手柄接收代码按官方 SDK 的原始边界
保留。仓库只删除了与项目构建无关的 APK、截图和示例工程，没有把这些依赖归为团队自有
实现。

## 仿真与强化学习基础设施

高层环境使用 MuJoCo，底层训练使用 NVIDIA Isaac Lab。二者分别遵循其上游许可证。
项目内保留的 `rsl_rl` 衍生代码继续遵循原项目许可证；团队修改集中在 S10 观测、地形、
奖励、训练协议和模型接口适配。

## 项目自有 LiDAR 线路

`sru_training/lidar_runtime/`、
`checkpoints/lidar_encoder_random_terrain_ft/best.pt` 和对应 ONNX 来自团队在比赛
期间适配并训练的前后双 LiDAR encoder，不是原 SRU 仓库附带的预训练模型。

## HIMLoco 与 LocoWheeledLegged

`s10_himloco_go2w_port/` 的内部模型与训练协议参考
`TrackinBIT/HIMLoco-for-Go2W`，机器人训练工程以 LocoWheeledLegged/Isaac Lab 生态为
基础。S10 端增加机器人资产、关节排列、执行器、动作缩放、历史状态与部署 ABI 适配。
原方法、上游工程和本项目 S10 迁移的作者边界分别保留，不将上游工作归属于本项目。

## KISS-ICP

局部 LiDAR odometry 诊断和部署候选使用 KISS-ICP。原项目由 Ignacio Vizzo、
Tiziano Guadagnino、Benedikt Mersch、Cyrill Stachniss 及贡献者开发，采用
MIT License。原仓库为 `https://github.com/PRBonn/kiss-icp`，许可证全文保存在
`docs/LICENSE_KISS_ICP`。本项目只增加双 Airy 点云转换、IMU/轮速先验和 SRU
目标闭环，不将 KISS-ICP 本身归属于本项目或生成式 AI。

## RoboSense rsLiDAR SDK

双 Airy ROS2 点云驱动使用 RoboSense 官方 `rsLiDAR_sdk v1.5.20` 及其
`rs_driver` 子模块，原仓库为 `https://github.com/RoboSense-LiDAR/rsLiDAR_sdk`。
项目安装脚本固定上游 commit，并仅将编译点类型从默认 `XYZI` 切换为
`XYZIRT`，以保留 ring 和逐点时间戳。上游许可证继续适用于该驱动代码。

## GLIM 建图与定位

正式建图前端使用 GLIM。其源码没有复制进本仓库，部署环境按上游许可证单独安装：

- GLIM：`https://github.com/koide3/glim`，MIT；提供 LIO、子地图和因子图基础。
- gtsam_points：`https://github.com/koide3/gtsam_points`，MIT；由 GLIM 使用的点云因子
  与优化组件。

完整 GLIM 路线数据和子图点云不随 GitHub 仓库分发；仓库仅保留构图代码、地图清单、
优化轨迹与留出圈评测证据。

## 生成式 AI 披露

本项目比赛期间的 MuJoCo 迁移，以及当前保留的建图、定位、评测与部署集成代码由
**GPT5.6Sol** 生成，并经团队在本地运行验证。这里的“生成代码”只描述本比赛项目中的
新增与迁移实现，不描述上游 SRU、Deep Robotics 官方 SDK、GLIM、KISS-ICP 或其他
第三方项目的作者身份。

团队通过合同测试、独立路线回放、模型哈希和实机只读检查验证新增代码。技术报告分别
陈述仿真、桌面回放和 AGX 在线证据，不把多会话候选的 10 cm 桌面结果等同于实机闭环。
