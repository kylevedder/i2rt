import threading
import time

import numpy as np

from i2rt.robots.motor_chain_robot import JointCommands, JointStates, MotorChainRobot
from i2rt.robots.utils import JointMapper


class _FakeMotorChain:
    running = True
    same_bus_device_driver = None

    def __len__(self) -> int:
        return 2


def _joint_state(pos: tuple[float, float] = (1.0, 2.0)) -> JointStates:
    return JointStates(
        names=["0", "1"],
        pos=np.array(pos),
        vel=np.array([0.1, 0.2]),
        eff=np.array([0.3, 0.4]),
        temp_mos=np.array([30.0, 31.0]),
        temp_rotor=np.array([32.0, 33.0]),
        timestamp=time.time(),
    )


def _bare_robot() -> MotorChainRobot:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot.motor_chain = _FakeMotorChain()
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._commands = JointCommands.init_all_zero(2)
    robot._joint_state = _joint_state()
    robot._last_motor_torques = np.array([0.5, 0.6])
    robot._last_gripper_command_qpos = 1.0
    robot._gripper_index = None
    robot._gripper_limits = None
    robot._gripper_force_limiter = None
    robot._limit_gripper_force = -1
    robot._gripper_closing_position_limiter = None
    robot._max_gripper_closing_position_error_rad = None
    robot._joint_limits = None
    robot._kp = np.array([10.0, 11.0])
    robot._kd = np.array([1.0, 1.1])
    robot._grav_comp_kd = np.zeros(2)
    robot._coulomb_friction = np.zeros(2)
    robot.gravity_comp_factor = np.ones(2)
    robot._clip_motor_torque = np.inf
    robot.remapper = JointMapper({}, 2)
    robot.temp_record_flag = True
    robot._arm_type = "arm"
    robot._gripper_type = "gripper"
    return robot


def test_update_does_not_hold_state_lock_during_motor_io() -> None:
    robot = _bare_robot()
    motor_io_started = threading.Event()
    release_motor_io = threading.Event()
    reader_done = threading.Event()
    robot._compute_gravity_compensation = lambda _state: np.zeros(2)

    def blocked_motor_io(_torques: np.ndarray, _commands: JointCommands) -> JointStates:
        motor_io_started.set()
        assert release_motor_io.wait(timeout=1.0)
        return _joint_state((3.0, 4.0))

    robot._update_joint_state = blocked_motor_io
    update_thread = threading.Thread(target=robot.update)
    update_thread.start()
    assert motor_io_started.wait(timeout=1.0)

    reader_thread = threading.Thread(target=lambda: (robot.get_joint_pos(), reader_done.set()))
    reader_thread.start()
    assert reader_done.wait(timeout=0.2)

    release_motor_io.set()
    update_thread.join(timeout=1.0)
    reader_thread.join(timeout=1.0)
    assert np.array_equal(robot.get_joint_pos(), [3.0, 4.0])


def test_joint_state_command_and_gain_pair_publish_atomically_by_copy() -> None:
    robot = _bare_robot()
    kp = np.array([20.0, 21.0])
    kd = np.array([2.0, 2.1])
    robot.update_kp_kd(kp, kd)
    kp[:] = -1.0
    kd[:] = -1.0

    pos = np.array([0.4, 0.5])
    vel = np.array([0.6, 0.7])
    robot.command_joint_state({"pos": pos, "vel": vel})
    pos[:] = -2.0
    vel[:] = -2.0

    with robot._command_lock:
        command = robot._commands
        assert np.array_equal(command.pos, [0.4, 0.5])
        assert np.array_equal(command.vel, [0.6, 0.7])
        assert np.array_equal(command.kp, [20.0, 21.0])
        assert np.array_equal(command.kd, [2.0, 2.1])


def test_joint_state_command_never_publishes_a_partially_built_command() -> None:
    robot = _bare_robot()
    robot._commands.pos[:] = 9.0
    robot._commands.vel[:] = 8.0
    remap_started = threading.Event()
    release_remap = threading.Event()

    class BlockingMapper:
        def to_robot_joint_pos_space(self, pos: np.ndarray) -> np.ndarray:
            remap_started.set()
            assert release_remap.wait(timeout=1.0)
            return pos.copy()

        def to_robot_joint_vel_space(self, vel: np.ndarray) -> np.ndarray:
            return vel.copy()

    robot.remapper = BlockingMapper()
    command_thread = threading.Thread(
        target=robot.command_joint_state,
        args=({"pos": np.array([1.0, 2.0]), "vel": np.array([3.0, 4.0])},),
    )
    command_thread.start()
    assert remap_started.wait(timeout=1.0)

    with robot._command_lock:
        assert np.array_equal(robot._commands.pos, [9.0, 9.0])
        assert np.array_equal(robot._commands.vel, [8.0, 8.0])

    release_remap.set()
    command_thread.join(timeout=1.0)
    with robot._command_lock:
        assert np.array_equal(robot._commands.pos, [1.0, 2.0])
        assert np.array_equal(robot._commands.vel, [3.0, 4.0])


def test_gain_pair_reader_waits_for_complete_update() -> None:
    robot = _bare_robot()
    copy_started = threading.Event()
    release_copy = threading.Event()
    reader_done = threading.Event()
    observed = {}

    class BlockingGain:
        shape = (2,)

        def copy(self) -> np.ndarray:
            copy_started.set()
            assert release_copy.wait(timeout=1.0)
            return np.array([20.0, 21.0])

    update_thread = threading.Thread(target=robot.update_kp_kd, args=(BlockingGain(), np.array([2.0, 2.1])))
    update_thread.start()
    assert copy_started.wait(timeout=1.0)

    def read_info() -> None:
        observed.update(robot.get_robot_info())
        reader_done.set()

    reader_thread = threading.Thread(target=read_info)
    reader_thread.start()
    assert not reader_done.wait(timeout=0.1)
    release_copy.set()
    update_thread.join(timeout=1.0)
    reader_thread.join(timeout=1.0)

    assert np.array_equal(observed["kp"], [20.0, 21.0])
    assert np.array_equal(observed["kd"], [2.0, 2.1])


def test_robot_getters_return_owned_arrays() -> None:
    robot = _bare_robot()
    robot._joint_limits = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    robot._gripper_limits = np.array([0.0, 1.0])

    pos = robot.get_joint_pos()
    observations = robot.get_observations()
    torques = robot.get_motor_torques()
    info = robot.get_robot_info()
    pos[:] = 99.0
    observations["joint_vel"][:] = 99.0
    observations["temp_mos"][:] = 99.0
    torques[:] = 99.0
    info["kp"][:] = 99.0
    info["joint_limits"][:] = 99.0

    assert np.array_equal(robot._joint_state.pos, [1.0, 2.0])
    assert np.array_equal(robot._joint_state.vel, [0.1, 0.2])
    assert np.array_equal(robot._joint_state.temp_mos, [30.0, 31.0])
    assert np.array_equal(robot._last_motor_torques, [0.5, 0.6])
    assert np.array_equal(robot._kp, [10.0, 11.0])
    assert np.array_equal(robot._joint_limits, [[-1.0, 1.0], [-2.0, 2.0]])


def test_gripper_observation_saturates_endpoint_overshoot_without_mutating_internal_state() -> None:
    robot = _bare_robot()
    robot._gripper_index = 1

    for measured, expected in ((-0.00171, 0.0), (1.00171, 1.0)):
        robot._joint_state = _joint_state((0.5, measured))

        observations = robot.get_observations()

        assert observations["gripper_pos"][0] == expected
        assert robot._joint_state.pos[1] == measured
