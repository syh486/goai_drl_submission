# MuJoCo SRU Runtime

`sru_training/` 是高层训练的模型与仿真库，包含：

- S10 MuJoCo 向量环境和官方低层 ONNX 调用；
- SRU actor/critic、PPO/MDPO 算法与循环 rollout；
- 双 LiDAR raster、Warp raycast 和 Hybrid VQ-AE encoder；
- critic 高度特征、动作滤波、奖励和 reset 协议。

公共训练入口统一放在 `training/train.py`，本目录不再保存历史 smoke、初赛 waypoint 或
一次性审计脚本。

