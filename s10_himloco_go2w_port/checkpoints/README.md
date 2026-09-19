# Retained S10 Low-Level Checkpoints

Only the two deployment-relevant snapshots are retained:

| Checkpoint | Role |
| --- | --- |
| `go2w_strict_remote_20260919/model_9000.pt` | Strict HIMLoco S10 training result before deployment adaptation |
| `go2w_deploy_ft_remote_20260919/model_1250.pt` | Deployment-oriented fine-tuning result initialized from the strict policy |

Intermediate checkpoints, TensorBoard logs and evaluation traces are excluded from the source repository.
