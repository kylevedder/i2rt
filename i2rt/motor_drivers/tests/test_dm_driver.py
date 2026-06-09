import threading
from types import SimpleNamespace

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd
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
