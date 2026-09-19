# SRU High-Level Training

本目录保存正式高层训练协议：随机地形生成、双 LiDAR encoder 数据流程、PPO/SRU 训练和
仿真评测。底层动力学与 SRU 网络实现位于 `sru_training/`。

```bash
./training/run_training.sh training/configs/ppo_sru_stage5_final_128.yaml
```

保留配置：

- `integration_sru_smoke.yaml`：小规模接口检查；
- `ppo_sru_stage4_full_no_pits_128.yaml`：完整无坑随机地形；
- `ppo_sru_stage5_lower_density_stairs_128.yaml`：降低楼梯区障碍密度；
- `ppo_sru_stage5_final_128.yaml`：最终续训协议。

LiDAR encoder 的采样、回放和训练入口位于 `training/lidar/`；评测与 GUI play 位于
`training/evaluation/`。训练输出写入本地日志目录，不作为源码提交。

