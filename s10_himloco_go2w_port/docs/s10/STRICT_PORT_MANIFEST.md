# S10 strict-port manifest

Baseline: `zhaozijie2022/LocoWheeledLegged` commit `015b8230b1a21331ab6d2b054a2ef8f1e435db84`.

The Go2W task remains in `locowheeledlegged/config/go2w` as the audit baseline. The S10 task is an additive port under `locowheeledlegged/config/s10`.

## Authorized task differences

| Area | Upstream | S10 strict port | Reason |
|---|---|---|---|
| Robot asset | Go2W URDF | Official S10 URDF/MJCF and generated USD | Robot replacement |
| Base/body names | `base`, `*_foot` | `base_link`, `*_wheel` | S10 link names |
| Joint order | Go2W order | Official S10 policy order, explicitly preserved | Deployment ABI |
| Default pose | Go2W pose | Official S10 controller pose | Robot replacement |
| Actuation | Go2W limits/gains | legs 50 Nm, wheels 14 Nm; 80/2 and 0/0.8 gains | Official S10 controller/MJCF |
| Action scale | uniform leg 0.25 | hip-x 0.125, other legs 0.25, wheels 5.0 | Official S10 runner |
| Nominal height | 0.40 m | 0.423 m | Default-pose S10 clearance |
| Initial terrain level | 5 | 2 | Explicitly requested |
| CoM randomization | enabled | disabled | S10 has a fixed measured asymmetric CoM |
| Gait reward | none | exactly one of A/B/C | Requested comparison |
| Model | upstream PPO | upstream PPO retained; HIMLoco added in parallel | Requested model comparison |
| Contact sensor paths | Go2W body names | S10 base/hip body names | Robot-specific link-name replacement; termination rules are unchanged |

## Intentionally unchanged

- All six terrain families, dimensions, ranges, proportions, seed and curriculum.
- XYZ command generator and ranges: `vx=[-1,1]`, `vy=[-0.5,0.5]`, `wz=[-pi/4,pi/4]`.
- `bang_bang_envs=0.05`, command resampling and initial zero-command behavior.
- Tracking rewards, stability penalties, action-rate penalty and joint penalties.
- Episode length, control frequency, physics timestep and terrain progression rule.
- Termination semantics and map-boundary timeout behavior.

## A/B/C gait terms

- A: weak per-wheel lateral clearance corridor.
- B: official-ONNX calibrated per-wheel target bands with command-dependent deadband.
- C: front/rear support width and lateral-center constraints without joint mirroring.

Each term has weight `-0.5`; no other reward weight differs between A/B/C.

## Isaac Lab 4.5 compatibility-only changes

- `ray_alignment="yaw"` maps to `attach_yaw_only=True`.
- Removed `quat_apply_inverse` calls map to `quat_rotate_inverse`.
- The removed upstream `randomize_rigid_body_com` helper is supplied locally for the untouched Go2W task. S10 does not invoke it.

## Validation performed

- Seven static audit assertions cover command identity, terrain identity, reward identity, joint order/action scales, default-height neutrality, A/B/C isolation and asset checksums.
- A, B and C each completed a real Isaac Sim 4.5 smoke run with two environments and one full HIM update.
- Runtime dimensions were identical for all variants: 16 actions, `6 x 57 = 342` policy history and 247 critic observations.
- The default spawn height is 0.424 m and the target height is 0.423 m. On flat ground the unweighted height penalty is only `1e-6` before settling, so the default pose is not driven sideways to satisfy height reward.

## Asset provenance

- URDF SHA256: `67755f9ea87bff2a45801fb3a7c407a06604c117ed9ec951f114b20b3ba01cba`
- MJCF SHA256: `c869bd7032f1ea7a6319cd0165f38e783caf3bccd25593d337ff9fe2779cc8ef`
- USD SHA256: `74c1bc0710a2fc07e998f665db6e75bc212b51a32226027a36092784c804d41e`
