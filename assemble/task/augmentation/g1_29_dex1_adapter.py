"""DDS replay adapter for 14 G1 arm joints and two Dex1 grippers."""

from __future__ import annotations

import time

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


ARM_MOTOR_INDICES = tuple(range(15, 29))


class G1Dex1ReplayPublisher:
    def __init__(self, timeout: float) -> None:
        ChannelFactoryInitialize(1)
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init()
        deadline = time.monotonic() + timeout
        state = None
        while time.monotonic() < deadline:
            state = self.lowstate_subscriber.Read()
            if state is not None:
                break
            time.sleep(0.01)
        if state is None:
            raise TimeoutError("Timed out waiting for G1 rt/lowstate")

        self.command = unitree_hg_msg_dds__LowCmd_()
        self.command.mode_pr = 0
        self.command.mode_machine = state.mode_machine
        for index, motor in enumerate(self.command.motor_cmd):
            motor.mode = 1
            motor.dq = 0.0
            motor.tau = 0.0
            if index < len(state.motor_state):
                motor.q = float(state.motor_state[index].q)
            motor.kp = 40.0 if index in ARM_MOTOR_INDICES else 100.0
            motor.kd = 1.5 if index in ARM_MOTOR_INDICES else 3.0
        self.crc = CRC()

        self.left_publisher = ChannelPublisher("rt/dex1/left/cmd", MotorCmds_)
        self.right_publisher = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        self.left_publisher.Init()
        self.right_publisher.Init()
        self.left_state_subscriber = ChannelSubscriber(
            "rt/dex1/left/state", MotorStates_
        )
        self.right_state_subscriber = ChannelSubscriber(
            "rt/dex1/right/state", MotorStates_
        )
        self.left_state_subscriber.Init()
        self.right_state_subscriber.Init()
        self.left_command = self._make_gripper_command()
        self.right_command = self._make_gripper_command()

    @staticmethod
    def _make_gripper_command() -> MotorCmds_:
        message = MotorCmds_()
        motor = unitree_go_msg_dds__MotorCmd_()
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 5.0
        motor.kd = 0.05
        message.cmds = [motor]
        return message

    def publish(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError("Replay action must be a finite 16-vector")
        for motor_index, target in zip(ARM_MOTOR_INDICES, values[:14]):
            self.command.motor_cmd[motor_index].q = float(target)
        self.command.crc = self.crc.Crc(self.command)
        self.lowcmd_publisher.Write(self.command)
        self.left_command.cmds[0].q = float(values[14])
        self.right_command.cmds[0].q = float(values[15])
        self.left_publisher.Write(self.left_command)
        self.right_publisher.Write(self.right_command)

    def read_state(self, fallback: np.ndarray) -> np.ndarray:
        result = np.asarray(fallback, dtype=np.float32).reshape(16).copy()
        lowstate = self.lowstate_subscriber.Read()
        if lowstate is not None and len(lowstate.motor_state) > ARM_MOTOR_INDICES[-1]:
            result[:14] = [
                float(lowstate.motor_state[index].q) for index in ARM_MOTOR_INDICES
            ]
        left = self.left_state_subscriber.Read()
        right = self.right_state_subscriber.Read()
        if left is not None and left.states:
            result[14] = float(left.states[0].q)
        if right is not None and right.states:
            result[15] = float(right.states[0].q)
        return result
