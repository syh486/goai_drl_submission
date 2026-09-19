# S10 HIMLoco 轮足底层训练

本目录是项目唯一保留的自主底层训练仓库。算法协议参考
`TrackinBIT/HIMLoco-for-Go2W`，机器人资产、质量惯量、关节顺序、动作尺度和执行器参数
迁移到 Deep Robotics S10。

主要入口：

```bash
python scripts/s10/train_go2w_him.py --num-envs 4096 --headless
python scripts/s10/play_go2w_him.py --checkpoint CHECKPOINT
python scripts/s10/play_go2w_him_mujoco.py --checkpoint CHECKPOINT
```

目录说明：

- `locowheeledlegged/config/s10/`：S10 Isaac Lab 任务；
- `locowheeledlegged/him/`：source/target 内部模型与训练 runner；
- `s10_policy_protocol.py`：策略顺序、机器人顺序和动作协议；
- `deploy_real/`：实体部署参考接口；
- `checkpoints/`：保留的底层训练结果；
- `docs/s10/`：严格迁移与参考协议。

完整迁移边界见
[GO2W_REFERENCE_PORT_PROTOCOL.md](docs/s10/GO2W_REFERENCE_PORT_PROTOCOL.md)。训练日志和大体积
评测 trace 不进入源码仓库。

