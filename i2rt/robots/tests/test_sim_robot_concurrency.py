import threading

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.sim_robot import SimRobot
from i2rt.robots.utils import ArmType, GripperType


def test_command_joint_state_publishes_position_and_velocity_together() -> None:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.NO_GRIPPER, sim=True)
    assert isinstance(robot, SimRobot)
    position_applied = threading.Event()
    release_command = threading.Event()
    reader_done = threading.Event()
    observed = {}
    original_apply = robot._apply_joint_pos_locked

    def blocked_apply(pos: np.ndarray) -> None:
        original_apply(pos)
        position_applied.set()
        assert release_command.wait(timeout=1.0)

    robot._apply_joint_pos_locked = blocked_apply
    pos = np.full(robot.num_dofs(), 0.2)
    vel = np.full(robot.num_dofs(), 0.3)
    writer = threading.Thread(target=robot.command_joint_state, args=({"pos": pos, "vel": vel},))

    def read_state() -> None:
        observed.update(robot.get_joint_state())
        reader_done.set()

    reader = threading.Thread(target=read_state)
    try:
        writer.start()
        assert position_applied.wait(timeout=1.0)
        reader.start()
        assert not reader_done.wait(timeout=0.1)
        release_command.set()
        writer.join(timeout=1.0)
        reader.join(timeout=1.0)

        assert np.array_equal(observed["pos"], pos)
        assert np.array_equal(observed["vel"], vel)
    finally:
        release_command.set()
        robot.close()
