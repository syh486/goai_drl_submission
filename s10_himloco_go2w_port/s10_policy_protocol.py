"""Simulator-independent S10 joint-order and MuJoCo address contract."""

from __future__ import annotations

from typing import Any


LEG_JOINT_NAMES = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
)
WHEEL_JOINT_NAMES = (
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
ROBOT_JOINT_NAMES = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
)
POLICY_TO_ROBOT_INDICES = tuple(ROBOT_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES)
ROBOT_TO_POLICY_INDICES = tuple(POLICY_JOINT_NAMES.index(name) for name in ROBOT_JOINT_NAMES)
WHEEL_BODY_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
BASE_COM_BODY = (-0.000512, 0.057317, 0.001182)
DEFAULT_JOINT_POSITIONS = {
    "fl_hipx_joint": 0.05,
    "fl_hipy_joint": -0.35,
    "fl_knee_joint": 0.65,
    "fl_wheel_joint": 0.0,
    "fr_hipx_joint": -0.05,
    "fr_hipy_joint": -0.35,
    "fr_knee_joint": 0.65,
    "fr_wheel_joint": 0.0,
    "hl_hipx_joint": 0.05,
    "hl_hipy_joint": 0.35,
    "hl_knee_joint": -0.65,
    "hl_wheel_joint": 0.0,
    "hr_hipx_joint": -0.05,
    "hr_hipy_joint": 0.35,
    "hr_knee_joint": -0.65,
    "hr_wheel_joint": 0.0,
}


def robot_values_to_policy(values: Any) -> Any:
    """Gather a robot/MuJoCo-order tensor into the S10 policy order."""

    return values[..., POLICY_TO_ROBOT_INDICES]


def policy_values_to_robot(values: Any) -> Any:
    """Gather a policy-order tensor into the robot/MuJoCo order."""

    return values[..., ROBOT_TO_POLICY_INDICES]


def audit_mujoco_s10_protocol(model: Any, mujoco_module: Any) -> dict[str, Any]:
    """Validate qpos/qvel/actuator addresses in an already loaded MuJoCo model."""

    mj = mujoco_module
    joints: list[str] = []
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    for joint_id in range(1, model.njnt):
        joints.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id))
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
    if tuple(joints) != ROBOT_JOINT_NAMES:
        raise RuntimeError(f"MuJoCo joint order mismatch: {tuple(joints)}")
    if tuple(qpos_addresses) != tuple(range(7, 23)):
        raise RuntimeError(f"MuJoCo qpos addresses mismatch: {qpos_addresses}")
    if tuple(qvel_addresses) != tuple(range(6, 22)):
        raise RuntimeError(f"MuJoCo qvel addresses mismatch: {qvel_addresses}")

    actuators = tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    if actuators != ROBOT_JOINT_NAMES:
        raise RuntimeError(f"MuJoCo actuator order mismatch: {actuators}")

    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    base_com = tuple(float(value) for value in model.body_ipos[base_id])
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(base_com, BASE_COM_BODY)):
        raise RuntimeError(f"MuJoCo base COM mismatch: actual={base_com}, expected={BASE_COM_BODY}")

    sensor_addresses = tuple(int(value) for value in model.sensor_adr)
    sensor_dimensions = tuple(int(value) for value in model.sensor_dim)
    sensor_types = tuple(int(value) for value in model.sensor_type)
    expected_sensor_types = (
        int(mj.mjtSensor.mjSENS_FRAMEQUAT),
        int(mj.mjtSensor.mjSENS_ACCELEROMETER),
        int(mj.mjtSensor.mjSENS_GYRO),
    )
    if sensor_addresses != (0, 4, 7) or sensor_dimensions != (4, 3, 3):
        raise RuntimeError(
            f"MuJoCo IMU sensor layout mismatch: addresses={sensor_addresses}, dimensions={sensor_dimensions}"
        )
    if sensor_types != expected_sensor_types or int(model.nsensordata) != 10:
        raise RuntimeError(
            f"MuJoCo IMU sensor types mismatch: actual={sensor_types}, expected={expected_sensor_types}"
        )

    expected_axes = {
        name: (-1.0, 0.0, 0.0) if "hipx" in name else (0.0, -1.0, 0.0)
        for name in ROBOT_JOINT_NAMES
    }
    for joint_id, name in enumerate(joints, start=1):
        axis = tuple(float(value) for value in model.jnt_axis[joint_id])
        if axis != expected_axes[name]:
            raise RuntimeError(f"MuJoCo axis mismatch for {name}: {axis}")

    return {
        "robot_joint_order": tuple(joints),
        "qpos_addresses": tuple(qpos_addresses),
        "qvel_addresses": tuple(qvel_addresses),
        "actuator_order": actuators,
        "base_com_body_xyz": base_com,
        "sensor_addresses": sensor_addresses,
        "sensor_dimensions": sensor_dimensions,
        "sensor_types": sensor_types,
        "policy_to_robot": POLICY_TO_ROBOT_INDICES,
        "robot_to_policy": ROBOT_TO_POLICY_INDICES,
    }
