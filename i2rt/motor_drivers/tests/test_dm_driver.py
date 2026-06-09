import threading
from types import SimpleNamespace

import pytest

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, PassiveEncoderInfo
from i2rt.motor_drivers.utils import ReceiveMode
from i2rt.utils.utils import RateRecorder


def test_control_loop_releases_command_lock_during_can_io() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.running = True
    chain.command_lock = threading.RLock()
    chain.commands = [MotorCmd()]
    chain._report_interval = 30.0
    chain._rate_recorder = RateRecorder()
    chain.state_lock = threading.Lock()
    chain.same_bus_device_driver = None
    chain._update_absolute_positions = lambda _feedback: None

    probe_acquired_lock = threading.Event()
    probe_thread = None
    lock_was_available_during_io = False

    def fake_set_commands(commands: list[MotorCmd]) -> list[SimpleNamespace]:
        nonlocal lock_was_available_during_io, probe_thread
        assert commands is chain.commands

        def probe_command_update() -> None:
            with chain.command_lock:
                probe_acquired_lock.set()

        probe_thread = threading.Thread(target=probe_command_update)
        probe_thread.start()
        lock_was_available_during_io = probe_acquired_lock.wait(timeout=0.1)
        chain.running = False
        return [SimpleNamespace(error_code="0x1")]

    chain._set_commands = fake_set_commands
    chain._set_torques_and_update_state()

    assert probe_thread is not None
    probe_thread.join(timeout=1.0)
    assert not probe_thread.is_alive()
    assert lock_was_available_during_io


def test_same_bus_device_failure_does_not_hold_snapshot_lock_or_clear_state() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.same_bus_device_lock = threading.Lock()
    previous_state = [PassiveEncoderInfo(id=1, position=0.25, velocity=0.0, io_inputs=[False, True])]
    chain.same_bus_device_states = previous_state
    chain._same_bus_device_next_poll_time = 0.0
    chain._same_bus_device_last_warning_time = 0.0

    poll_started = threading.Event()
    release_poll = threading.Event()

    class FailingDriver:
        def read_states(self, max_retry: int) -> None:
            assert max_retry == 1
            poll_started.set()
            assert release_poll.wait(timeout=1.0)
            raise AssertionError("encoder disconnected")

    chain.same_bus_device_driver = FailingDriver()
    poll_thread = threading.Thread(target=chain._poll_same_bus_device)
    poll_thread.start()
    assert poll_started.wait(timeout=1.0)

    with chain.same_bus_device_lock:
        assert chain.same_bus_device_states is previous_state

    release_poll.set()
    poll_thread.join(timeout=1.0)
    assert not poll_thread.is_alive()
    assert chain.same_bus_device_states is previous_state
    assert chain._same_bus_device_next_poll_time > 0.0


def test_single_can_attempt_does_not_perform_an_extra_receive_drain() -> None:
    can_interface = CanInterface.__new__(CanInterface)
    can_interface.bus = SimpleNamespace(send=lambda _message: None, channel_info="fake")
    can_interface.name = "fake"
    can_interface.receive_mode = ReceiveMode.p16
    can_interface._receive_message = lambda _motor_id, timeout: None
    can_interface.try_receive_message = lambda _motor_id: (_ for _ in ()).throw(
        AssertionError("single attempt must not drain again")
    )

    with pytest.raises(AssertionError, match="fail to communicate"):
        can_interface._send_message_get_response(1, 1, [0xFF], max_retry=1, drain_on_final_failure=False)


def test_partial_velocity_updates_do_not_overwrite_each_other() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.command_lock = threading.RLock()
    chain.commands = [MotorCmd(vel=1.0), MotorCmd(vel=2.0)]
    start = threading.Barrier(3)

    def update(index: int, velocity: float) -> None:
        start.wait()
        chain.update_command_velocities({index: velocity})

    first = threading.Thread(target=update, args=(0, 10.0))
    second = threading.Thread(target=update, args=(1, 20.0))
    first.start()
    second.start()
    start.wait()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert [command.vel for command in chain.commands] == [10.0, 20.0]
