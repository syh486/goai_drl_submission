# S10 Go2W/HIMLoco strict-port protocol

Audit date: 2026-09-18

The algorithmic reference is `TrackinBIT/HIMLoco-for-Go2W` at commit
`011693738c61603c3f22f2bce755098dd36fa7eb`. The port changes robot-specific
physics to the official S10 asset and control protocol; task and HIM semantics
follow the observed execution of that commit, including relevant side effects.

## Joint and action order

Policy order is 12 leg joints in FL, FR, HL, HR order, followed by four wheels.
The S10 robot/MuJoCo order interleaves each wheel after its three leg joints.
The two explicit permutations live in `s10_policy_protocol.py` and are audited
against MuJoCo qpos, qvel, actuator and joint-axis addresses.

IsaacLab's internal leg actuator order is different again: four hip-x joints,
four hip-y joints, then four knees. Reset and delay targets are therefore
resolved by joint name or sliced in the actuator's actual order; policy-order
tuples must never be written directly into an actuator buffer.

## Observation and reset semantics

- One frame is 57 values: body angular velocity, projected gravity, scaled
  command, policy-order joint-position error, policy-order joint velocity, and
  raw action.
- Wheel position and velocity signals are zero before observation noise, which
  reproduces the reference reward-side-effect behavior.
- Actor and critic share the exact same noisy current proprioceptive frame.
- The actor receives six frames newest-first. Reset preserves the five older
  frames instead of clearing history.
- The reference copies the terminal action back into `last_actions` after
  reset. The port therefore preserves the terminal action as the reset-frame
  action, the next action-rate predecessor, and the initial value in every
  delayed-command slot.
- HIM estimator targets for done transitions use the critic observation
  captured before reset, rather than the reset-state critic observation.

## Commands, randomization, and rewards

- Heading control uses `0.5 * wrapped_heading_error` and the reference's
  hard-coded `[-2, 2] rad/s` clamp.
- Friction is reassigned on every episode reset.
- Kp, Kd and motor-strength randomization each use one scalar per environment,
  shared across all joints. Motor strength scales both implicit-PD gains.
- Reward values, positive-total clipping, terrain curriculum, command
  curriculum, observation/action clipping and the 0--3 physics-tick action
  delay follow the locked reference.
- Fixed S10 COM asymmetry is retained. COM randomization is intentionally
  disabled by project decision.

## Intentional S10 adaptations

- Official S10 URDF/USD/MJCF, mass/inertia, default pose and fixed COM.
- Leg action scales: 0.125 for hip-x and 0.25 for hip-y/knee.
- Wheel velocity action scale: 5.0.
- Leg implicit PD: 80/2 with 50 Nm limit; wheel damping: 0.8 with 14 Nm limit.
- Nominal base clearance: 0.423 m.

## Remaining simulator differences

IsaacGym and IsaacLab use different PhysX generations and terrain cooking;
MuJoCo uses a different contact solver again. The IsaacLab stair terrain is a
mesh equivalent of the reference height-field pyramid, so contact details are
not bit-identical. These are physical-engine differences, not tensor-order or
task-protocol differences.

MuJoCo deployment must maintain six newest-first frames, gather joints from
robot order into policy order, zero the four wheel position/velocity observation
signals, apply the policy action scales above, map actions back into robot order,
and update the policy at 50 Hz. Run `scripts/s10/audit_mujoco_protocol.py` before
any deployment test.
