import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

if "RPi" not in sys.modules:
    gpio = types.ModuleType("RPi.GPIO")
    gpio.HIGH = 1
    rpi = types.ModuleType("RPi")
    rpi.GPIO = gpio
    sys.modules["RPi"] = rpi
    sys.modules["RPi.GPIO"] = gpio

from i2rt.flow_base import flow_base_client, linear_rail_controller
from i2rt.flow_base.flow_base_client import FlowBaseClient
from i2rt.flow_base.flow_base_controller import TimeoutRemoteCommand, Vehicle, VehicleMotorController
from i2rt.flow_base.linear_rail_controller import LinearRailController, SingleMotorControlInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd


def _velocity_chain(count: int) -> DMChainCanInterface:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.command_lock = threading.RLock()
    chain.commands = [MotorCmd() for _ in range(count)]
    chain.motor_list = [(idx, "fake") for idx in range(count)]
    return chain


def test_base_and_rail_velocity_updates_preserve_each_other() -> None:
    chain = _velocity_chain(3)
    base = VehicleMotorController.__new__(VehicleMotorController)
    base.num_casters = 1
    base.motor_interface = chain
    base.homing_check_callback = lambda: (_ for _ in ()).throw(AssertionError("callback must not run under DM lock"))
    rail = SingleMotorControlInterface(chain, target_motor_idx=2)
    start = threading.Barrier(3)

    def update_base() -> None:
        start.wait()
        base.set_velocities({"steer_vel": [1.0], "drive_vel": [2.0]})

    def update_rail() -> None:
        start.wait()
        rail.set_velocity(3.0)

    base_thread = threading.Thread(target=update_base)
    rail_thread = threading.Thread(target=update_rail)
    base_thread.start()
    rail_thread.start()
    start.wait()
    base_thread.join(timeout=1.0)
    rail_thread.join(timeout=1.0)

    assert [command.vel for command in chain.commands] == [1.0, 2.0, 3.0]


def test_limit_callback_updates_state_promptly_and_leaves_zero_as_final_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive_command_started = threading.Event()
    release_positive_command = threading.Event()

    class BlockingMotor:
        def __init__(self) -> None:
            self.commands = []

        def set_velocity(self, velocity: float) -> None:
            self.commands.append(velocity)
            if velocity > 0.0:
                positive_command_started.set()
                assert release_positive_command.wait(timeout=1.0)

        def get_state(self) -> SimpleNamespace:
            return SimpleNamespace(pos=0.0, vel=0.0)

    motor = BlockingMotor()
    rail = LinearRailController(motor, auto_home=False)
    with rail._lock:
        rail.brake_on = False
        rail.last_command_time = time.time()

    monkeypatch.setattr(
        linear_rail_controller.GPIO,
        "input",
        lambda _pin: linear_rail_controller.GPIO.HIGH,
        raising=False,
    )
    command_thread = threading.Thread(target=rail.set_velocity, args=(1.0,))
    callback_thread = threading.Thread(target=rail._upper_limit_callback, args=(0,))
    command_thread.start()
    assert positive_command_started.wait(timeout=1.0)
    callback_thread.start()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with rail._lock:
            if rail.upper_limit_triggered:
                break
        time.sleep(0.001)
    assert rail.upper_limit_triggered

    release_positive_command.set()
    command_thread.join(timeout=1.0)
    callback_thread.join(timeout=1.0)
    assert motor.commands[-1] == 0.0


def test_remote_command_returns_coherent_owned_snapshot() -> None:
    remote = TimeoutRemoteCommand(timeout=10.0)
    remote.remote_set_target_velocity({"target_velocity": np.array([1.0, 2.0, 3.0, 4.0]), "frame": "global"})

    command, frame, valid = remote.get_valid_command()
    remote.remote_set_target_velocity({"target_velocity": np.array([5.0, 6.0, 7.0]), "frame": "local"})
    next_command, next_frame, next_valid = remote.get_valid_command()

    assert valid and next_valid
    assert frame == "global"
    assert np.array_equal(command, [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(next_command, [5.0, 6.0, 7.0, 4.0])
    assert next_frame == "local"


def test_odometry_response_is_owned_and_reset_advances_generation() -> None:
    vehicle = Vehicle.__new__(Vehicle)
    vehicle._lock = threading.Lock()
    vehicle.num_dofs = 3
    vehicle.x = np.array([1.0, 2.0, 3.0])
    vehicle.dx = np.array([0.1, 0.2, 0.3])
    vehicle._odometry_reset_generation = 7

    odometry = vehicle.get_odometry()
    vehicle.x[:] = 9.0
    odometry["translation"][:] = -1.0
    assert np.array_equal(vehicle.x, [9.0, 9.0, 9.0])
    assert odometry["rotation"] == 3.0

    vehicle.reset_odometry()
    assert vehicle._odometry_reset_generation == 8
    assert np.array_equal(vehicle.x, np.zeros(3))
    assert np.array_equal(vehicle.dx, np.zeros(3))


def test_flow_base_client_does_not_hold_command_lock_during_rpc_and_closes_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rpc_started = threading.Event()
    close_called = threading.Event()

    class BlockingFuture:
        def result(self, timeout: float) -> None:
            rpc_started.set()
            assert close_called.wait(timeout=timeout + 1.0)
            raise TimeoutError

    class FakePortalClient:
        def __init__(self, _address: str) -> None:
            pass

        def set_target_velocity(self, _command: dict) -> BlockingFuture:
            return BlockingFuture()

        def close(self, timeout: float) -> None:
            close_called.set()

    monkeypatch.setattr(flow_base_client.portal, "Client", FakePortalClient)
    client = FlowBaseClient()
    assert rpc_started.wait(timeout=1.0)
    assert client._lock.acquire(timeout=0.2)
    client._lock.release()

    client.set_target_velocity(np.array([1.0, 2.0, 3.0]))
    client.close()
    assert not client._thread.is_alive()
