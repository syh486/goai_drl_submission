# S10 HIMLoco Wheeled-Legged Locomotion

This is the only maintained low-level training project in the repository. It
ports the HIMLoco-for-Go2W training protocol to the Deep Robotics S10 while
keeping the S10 joint order, actuator limits, action scales, history contract,
and deployment ABI explicit.

```bash
python scripts/s10/train_go2w_him.py --num-envs 4096 --headless
python scripts/s10/play_go2w_him.py --checkpoint CHECKPOINT
python scripts/s10/play_go2w_him_mujoco.py --checkpoint CHECKPOINT
```

See [the reference-port protocol](docs/s10/GO2W_REFERENCE_PORT_PROTOCOL.md) for
the algorithmic source, authorized robot-specific changes, and cross-simulator
validation contract. Retained model snapshots are under `checkpoints/`; local
training logs and large trace files are intentionally excluded.

