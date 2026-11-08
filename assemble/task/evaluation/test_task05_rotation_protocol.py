"""Task05 settled-rotation protocol tests without Isaac/Omni startup."""
from __future__ import annotations

import ast
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2] / "third_party/unitree_sim_isaaclab/tasks/g1_tasks/valve_handwheel_g1_29dof_dex1/valve_handwheel_controller.py"
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "ValveHandwheelController")
METHOD_NAMES = {"_update_success_checker", "_update_settle_gate", "_twist_angle_about_z"}
METHODS = [node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES]
assert {node.name for node in METHODS} == METHOD_NAMES
TEST_CLASS = ast.ClassDef(name="ControllerMethods", bases=[], keywords=[], body=METHODS, decorator_list=[])
MODULE = ast.fix_missing_locations(ast.Module(body=[TEST_CLASS], type_ignores=[]))
NAMESPACE = {
    "np": np, "math": math, "time": time, "torch": torch,
    "REQUIRED_ROTATION": math.radians(60),
    "RESET_SETTLE_SECONDS": 1.0,
    "SETTLE_STILL_THRESHOLD_RAD_S": 0.05,
    "SETTLE_STILL_FRAMES": 15,
    "SETTLE_HANDSHAKE_BUDGET_SECONDS": 20.0,
}
exec(compile(MODULE, str(SOURCE), "exec"), NAMESPACE)
ControllerMethods = NAMESPACE["ControllerMethods"]


class Task05ProtocolTests(unittest.TestCase):
    def base(self, phase="reset_settling"):
        c = ControllerMethods()
        c._phase = phase
        c._benchmark = SimpleNamespace(active=True, begin_settle=self.record_begin_settle(c))
        c._pending_settle = None
        c._settle_elapsed = 0.0
        c._settle_ignored_rotation = 0.0
        c._accumulated_rotation = 0.0
        c._counting_elapsed = 0.0
        c._settle_started_at = None
        c._settle_still_frames = 0
        c._settle_zeroings = 0
        c._settle_last_speed = 0.0
        c._settle_still_seconds = 0.0
        c._settle_last_diag = 0.0
        c._settle_budget_warned = False
        c._last_twist_angle = None
        c._last_wheel_angular_speed = 0.0
        c.success_latched = False
        c._failure_latched = False
        c.success_payload = None
        c.begin_settle_calls = []
        c.zero_calls = 0
        c.env = SimpleNamespace(num_envs=1, device=torch.device("cpu"))
        c.device = torch.device("cpu")
        c.handwheel = SimpleNamespace(
            write_root_velocity_to_sim=lambda velocity: setattr(c, "zero_calls", c.zero_calls + 1)
        )
        c._wheel_axis_angular_speed = lambda: c.wheel_speed
        c._current_relative_twist = lambda: c.angle
        c._update_success_checker_legacy = lambda: None
        c._mark_success = lambda: (
            setattr(c, "success_latched", True),
            setattr(c, "_phase", "success"),
            setattr(c, "success_payload", {
                "rotation_deg": math.degrees(c._accumulated_rotation),
                "counting_seconds": c._counting_elapsed,
            }),
        )
        c.wheel_speed = 0.0
        c.angle = 0.0
        return c

    @staticmethod
    def record_begin_settle(c):
        def begin_settle(request, details, steps=10):
            c.begin_settle_calls.append((request, dict(details), steps))
        return begin_settle

    def tick_gate(self, c, wheel_speed=None, elapsed=None):
        if wheel_speed is not None:
            c.wheel_speed = wheel_speed
        if elapsed is not None:
            c._settle_started_at = time.monotonic() - elapsed
        c._update_settle_gate()

    def tick_checker(self, c, delta, dt=1 / 30):
        c.angle += delta
        c._update_success_checker(dt)

    def test_gate_zeroes_drift_and_holds_handshake(self):
        c = self.base()
        c._pending_settle = ({"reset_id": "r1"}, {"protocol": "settled_rotation_v3"})
        c._settle_started_at = time.monotonic()
        for _ in range(10):
            self.tick_gate(c, wheel_speed=0.2, elapsed=0.0)
        self.assertEqual(c.zero_calls, 10)
        self.assertEqual(c._settle_zeroings, 10)
        self.assertEqual(c._settle_still_frames, 0)
        self.assertEqual(c.begin_settle_calls, [])

    def test_gate_waits_for_minimum_settle_time(self):
        c = self.base()
        c._pending_settle = ({"reset_id": "r1"}, {})
        c._settle_started_at = time.monotonic()
        for _ in range(20):
            self.tick_gate(c, wheel_speed=0.0, elapsed=0.2)
        self.assertEqual(c.begin_settle_calls, [])

    def test_gate_completes_after_still_frames(self):
        c = self.base()
        request = {"reset_id": "r1"}
        details = {"protocol": "settled_rotation_v3"}
        c._pending_settle = (request, details)
        for _ in range(30):
            self.tick_gate(c, wheel_speed=0.2, elapsed=1.2)
        self.assertEqual(c._settle_zeroings, 30)
        for _ in range(15):
            self.tick_gate(c, wheel_speed=0.0, elapsed=2.5)
        self.assertEqual(len(c.begin_settle_calls), 1)
        seen_request, seen_details, seen_steps = c.begin_settle_calls[0]
        self.assertEqual(seen_request, request)
        self.assertEqual(seen_steps, 1)
        self.assertEqual(seen_details["settle_zeroings"], 30)
        self.assertGreaterEqual(seen_details["settle_still_seconds"], 1.0)
        self.assertIsNone(c._pending_settle)

    def test_checker_stays_settling_until_gate_passes(self):
        c = self.base(phase="reset_settling")
        c._pending_settle = ({"reset_id": "r1"}, {})
        self.tick_checker(c, 0.0, dt=0.2)
        self.tick_checker(c, math.radians(55), dt=0.2)
        self.tick_checker(c, math.radians(20), dt=0.2)
        self.assertEqual(c._phase, "reset_settling")
        self.assertAlmostEqual(c._settle_ignored_rotation, math.radians(75))
        self.tick_checker(c, 0.0, dt=0.2)
        self.assertEqual(c._phase, "reset_settling")
        c._pending_settle = None
        self.tick_checker(c, math.radians(59), dt=0.2)
        self.assertEqual(c._phase, "counting")
        self.assertAlmostEqual(c._accumulated_rotation, 0.0)

    def test_counting_attributes_all_rotation_to_policy(self):
        c = self.base(phase="counting")
        c._last_twist_angle = 0.0
        for _ in range(7):
            self.tick_checker(c, math.radians(10))
        self.assertTrue(c.success_latched)
        self.assertAlmostEqual(c.success_payload["rotation_deg"], 70.0, places=5)

    def test_counting_accepts_either_direction(self):
        c = self.base(phase="counting")
        c._last_twist_angle = 0.0
        for _ in range(7):
            self.tick_checker(c, -math.radians(10))
        self.assertTrue(c.success_latched)
        self.assertAlmostEqual(c.success_payload["rotation_deg"], -70.0, places=5)

    def test_counting_net_rotation_below_threshold_keeps_going(self):
        c = self.base(phase="counting")
        c._last_twist_angle = 0.0
        for _ in range(4):
            self.tick_checker(c, math.radians(10))
        for _ in range(3):
            self.tick_checker(c, -math.radians(10))
        self.assertFalse(c.success_latched)
        self.assertAlmostEqual(c._accumulated_rotation, math.radians(10), places=5)

    def test_twist_angle_and_wrap_convention(self):
        method = ControllerMethods._twist_angle_about_z
        self.assertAlmostEqual(method(np.array([1.0, 0.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(method(np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])), math.pi / 2)
        self.assertAlmostEqual(method(np.array([math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)])), -math.pi / 2)

    def test_protocol_contract_and_config(self):
        text = SOURCE.read_text()
        for token in ("settled_rotation_v3", "_update_settle_gate", "SETTLE_STILL_THRESHOLD_RAD_S", "G1_TASK_FAILURE_FILE"):
            self.assertIn(token, text)
        for stale in ("rotation_without_active_manipulation", "lost_handwheel_grasp", "acquiring_grasp", "active_rotation_v2"):
            self.assertNotIn(stale, text)
        evaluator = (Path(__file__).resolve().parent / "evaluator.py").read_text()
        self.assertIn("settled_rotation_v3", evaluator)
        self.assertIn("task05_rotation_thresholds", evaluator)
        self.assertIn("settle_still_threshold_rad_s", evaluator)


if __name__ == "__main__":
    unittest.main()
