import threading
import time

import numpy as np
import pytest

from i2rt.robots.motor_chain_robot import JointCommands, JointStates, MotorChainRobot
from i2rt.robots.utils import ArmType, GripperClosingPositionLimiter, GripperType, JointMapper


def test_large_closing_position_error_is_clamped_to_fixed_gap() -> None:
    limiter = _limiter()

    target, was_limited = limiter.limit_target(
        target_qpos=0.0,
        current_qpos=1.0,
        closed_qpos=0.0,
        open_qpos=1.0,
    )

    assert was_limited
    assert target == pytest.approx(0.6)


def test_small_closing_position_error_is_unchanged() -> None:
    limiter = _limiter()

    target, was_limited = limiter.limit_target(
        target_qpos=0.85,
        current_qpos=1.0,
        closed_qpos=0.0,
        open_qpos=1.0,
    )

    assert not was_limited
    assert target == 0.85


def test_opening_target_is_unchanged() -> None:
    limiter = _limiter()

    target, was_limited = limiter.limit_target(
        target_qpos=1.0,
        current_qpos=0.5,
        closed_qpos=0.0,
        open_qpos=1.0,
    )

    assert target == 1.0
    assert not was_limited


def test_closing_target_is_clamped_with_real_reversed_raw_gripper_limits() -> None:
    limiter = _limiter()

    target, was_limited = limiter.limit_target(
        target_qpos=6.6,
        current_qpos=1.2,
        closed_qpos=6.6,
        open_qpos=1.2,
    )

    assert was_limited
    assert target == pytest.approx(1.6)


@pytest.mark.parametrize("max_error", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_position_error_limit_is_rejected(max_error: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        GripperClosingPositionLimiter(max_error)


def test_motor_chain_update_clamps_target_without_changing_native_controller() -> None:
    robot = _bare_gripper_robot(target_qpos=0.0, current_qpos=1.0)

    captured = _capture_update(robot)
    robot.update()

    command = captured["commands"]
    assert command.pos[1] == pytest.approx(0.6)
    assert command.kp[1] == 20.0
    assert command.kd[1] == 0.5
    assert captured["torques"][1] == 0.0
    assert robot._commands.pos[1] == 0.0


def test_endpoint_noise_keeps_one_continuous_position_controller() -> None:
    robot = _bare_gripper_robot(target_qpos=0.0, current_qpos=0.0)
    outputs = []

    def capture_update(torques: np.ndarray, commands: JointCommands) -> JointStates:
        outputs.append((commands.pos[1], commands.kp[1], commands.kd[1], torques[1]))
        return robot._joint_state

    robot._update_joint_state = capture_update
    for position in (0.001, -0.001, 0.0005, -0.0005, 0.0):
        robot._joint_state.pos[1] = position
        robot.update()

    assert outputs == [(0.0, 20.0, 0.5, 0.0)] * len(outputs)


def test_opening_frame_keeps_native_position_controller_untouched() -> None:
    robot = _bare_gripper_robot(target_qpos=1.0, current_qpos=0.5)

    captured = _capture_update(robot)
    robot.update()

    command = captured["commands"]
    assert command.pos[1] == 1.0
    assert command.kp[1] == 20.0
    assert command.kd[1] == 0.5
    assert captured["torques"][1] == 0.0


def test_runtime_limiter_failure_suppresses_gripper_without_stopping_arm_update() -> None:
    robot = _bare_gripper_robot(target_qpos=0.0, current_qpos=1.0)
    robot._gripper_closing_position_limiter.limit_target = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("bad state")
    )

    captured = _capture_update(robot)
    robot.update()

    command = captured["commands"]
    assert command.pos[1] == 1.0
    assert command.vel[1] == 0.0
    assert command.kp[1] == 0.0
    assert command.kd[1] == 0.0
    assert captured["torques"][1] == 0.0


def _limiter() -> GripperClosingPositionLimiter:
    return GripperClosingPositionLimiter(max_closing_position_error_rad=0.4)


def _capture_update(robot: MotorChainRobot) -> dict:
    captured = {}

    def capture_update(torques: np.ndarray, commands: JointCommands) -> JointStates:
        captured["torques"] = torques.copy()
        captured["commands"] = commands
        return robot._joint_state

    robot._update_joint_state = capture_update
    return captured


def _bare_gripper_robot(*, target_qpos: float, current_qpos: float) -> MotorChainRobot:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot.motor_chain = type("FakeChain", (), {"running": True, "same_bus_device_driver": None, "__len__": lambda _: 2})()
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._commands = JointCommands.init_all_zero(2)
    robot._commands.pos[1] = target_qpos
    robot._commands.kp[1] = 20.0
    robot._commands.kd[1] = 0.5
    robot._joint_state = JointStates(
        names=["0", "1"],
        pos=np.array([0.0, current_qpos]),
        vel=np.zeros(2),
        eff=np.zeros(2),
        temp_mos=np.zeros(2),
        temp_rotor=np.zeros(2),
        timestamp=time.time(),
    )
    robot._gripper_index = 1
    robot._gripper_limits = np.array([0.0, 1.0])
    robot._gripper_force_limiter = None
    robot._limit_gripper_force = -1
    robot._gripper_closing_position_limiter = _limiter()
    robot._max_gripper_closing_position_error_rad = 0.4
    robot._gripper_limit_last_log_time = -float("inf")
    robot._gripper_limit_last_error_time = -float("inf")
    robot._last_gripper_command_qpos = 1.0
    robot._arm_type = ArmType.YAM
    robot._gripper_type = GripperType.LINEAR_4310
    robot._coulomb_friction = np.zeros(2)
    robot.gravity_comp_factor = np.ones(2)
    robot._clip_motor_torque = np.inf
    robot.remapper = JointMapper({1: (0.0, 1.0)}, 2)
    robot._compute_gravity_compensation = lambda _state: np.zeros(2)
    robot._last_motor_torques = None
    return robot
