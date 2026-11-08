#!/usr/bin/env python3
"""Temporary BrainCo-Revo2 protocol bridge for the existing Inspire G1 sim.

This lets Quest hand tracking exercise xr_teleoperate's ``--ee=brainco``
retargeter before a native Revo2-on-G1 Isaac articulation is complete.  It
maps BrainCo's two six-joint command topics to the 12-joint Inspire DDS topic
used by the supplied G1-Inspire Isaac task.  The simulator will still display
the Inspire hand; this is a control-pipeline validation bridge, not a visual
or physical Revo2 replacement.
"""

from __future__ import annotations

import argparse
import signal
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


# BrainCo [thumb, thumb_aux, index, middle, ring, pinky] -> Inspire
# [pinky, ring, middle, index, thumb bend, thumb yaw]. Both retargeters use
# 0=open and 1=closed, while the Inspire sim DDS uses the inverse convention.
def brainco_to_inspire(values: list[float]) -> list[float]:
    v = (values + [0.0] * 6)[:6]
    return [1.0 - v[5], 1.0 - v[4], 1.0 - v[3], 1.0 - v[2], 1.0 - v[0], 1.0 - v[1]]


def inspire_to_brainco(values: list[float]) -> list[float]:
    v = (values + [1.0] * 12)[:12]
    return [1.0 - v[10], 1.0 - v[11], 1.0 - v[9], 1.0 - v[8], 1.0 - v[7], 1.0 - v[6]], [
        1.0 - v[4], 1.0 - v[5], 1.0 - v[3], 1.0 - v[2], 1.0 - v[1], 1.0 - v[0]
    ]


class Bridge:
    def __init__(self) -> None:
        self.left = [0.0] * 6
        self.right = [0.0] * 6
        self.running = True
        self.inspire_cmd = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.left_state = ChannelPublisher("rt/brainco/left/state", MotorStates_)
        self.right_state = ChannelPublisher("rt/brainco/right/state", MotorStates_)
        for publisher in (self.inspire_cmd, self.left_state, self.right_state):
            publisher.Init()
        self.left_sub = ChannelSubscriber("rt/brainco/left/cmd", MotorCmds_)
        self.right_sub = ChannelSubscriber("rt/brainco/right/cmd", MotorCmds_)
        self.inspire_state_sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.left_sub.Init(lambda msg: self._command("left", msg), 1)
        self.right_sub.Init(lambda msg: self._command("right", msg), 1)
        self.inspire_state_sub.Init(self._state, 1)

    def _command(self, side: str, msg: MotorCmds_) -> None:
        values = [float(cmd.q) for cmd in msg.cmds[:6]]
        if side == "left":
            self.left = values
        else:
            self.right = values
        combined = brainco_to_inspire(self.right) + brainco_to_inspire(self.left)
        output = MotorCmds_()
        output.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        for index, value in enumerate(combined):
            output.cmds[index].q = value
            output.cmds[index].dq = 1.0
        self.inspire_cmd.Write(output)

    def _state(self, msg: MotorStates_) -> None:
        values = [float(state.q) for state in msg.states[:12]]
        left, right = inspire_to_brainco(values)
        for publisher, values_for_side in ((self.left_state, left), (self.right_state, right)):
            output = MotorStates_()
            output.states = [unitree_go_msg_dds__MotorState_() for _ in range(6)]
            for index, value in enumerate(values_for_side):
                output.states[index].q = value
            publisher.Write(output)

    def run(self) -> None:
        print("[bridge] BrainCo <-> Inspire DDS bridge ready", flush=True)
        while self.running:
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--network-interface", default=None)
    args = parser.parse_args()
    ChannelFactoryInitialize(args.domain_id, networkInterface=args.network_interface)
    bridge = Bridge()
    signal.signal(signal.SIGINT, lambda *_: setattr(bridge, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(bridge, "running", False))
    bridge.run()


if __name__ == "__main__":
    main()
