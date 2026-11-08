"""Exercise actual Task04 methods without importing Isaac/Omni or starting simulation."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch


def controller_methods():
    source = Path(__file__).resolve().parents[2] / "third_party/unitree_sim_isaaclab/tasks/g1_tasks/hand_cart_push_g1_29dof_dex1/hand_cart_push_controller.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {"_update_success_checker", "override_action"}
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    cls = ast.ClassDef(name="ControllerMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "DUAL_GRASP_CONFIRM_FRAMES": 5, "POST_PUSH_HOLD_SECONDS": 0.5}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["ControllerMethods"]


ControllerMethods = controller_methods()


class Task04GraspStateTests(unittest.TestCase):
    def make_controller(self, phase="pushing", valid=True, motion=False):
        c = ControllerMethods()
        c._phase = phase; c._dual_grasp_valid_now = valid; c._dual_grasp_frames = 5
        c._post_push_hold_elapsed = 0.0; c._gripper_hold_targets = None
        c.success_latched = False; c._failure_latched = False
        c.stops = 0; c.captures = 0; c.failure_reason = None
        c._success_geometry = lambda: (motion, 0.5 if motion else 0.2, 0.0, 1.0)
        c._maybe_print_grasp_diagnostics = lambda *args: None
        def fail(reason):
            c._phase = "failure"; c._failure_latched = True; c.failure_reason = reason
        def success():
            c._phase = "success"; c.success_latched = True
        def stop(): c.stops += 1
        def capture(): c.captures += 1
        c._mark_failure = fail; c._mark_success = success
        c._stop_base_for_terminal_phase = stop; c._capture_gripper_hold_targets = capture
        return c

    def test_acquiring_cannot_succeed_even_at_target(self):
        c = self.make_controller(phase="acquiring_grasp", motion=True)
        c._update_success_checker(0.1, True)
        self.assertEqual(c._phase, "acquiring_grasp")
        self.assertFalse(c.success_latched)

    def test_one_bad_push_cycle_is_terminal_failure(self):
        c = self.make_controller(valid=False)
        c._update_success_checker(1 / 30, True)
        self.assertEqual(c.failure_reason, "grasp_lost_during_push")
        c._dual_grasp_valid_now = True
        c._success_geometry = lambda: (True, 0.6, 0.0, 1.0)
        c._update_success_checker(0.5, True)
        self.assertEqual(c._phase, "failure")
        self.assertFalse(c.success_latched)

    def test_target_requires_full_post_push_hold(self):
        c = self.make_controller(motion=True)
        c._update_success_checker(0.1, True)
        self.assertEqual(c._phase, "post_push_hold")
        self.assertEqual(c.captures, 1)
        self.assertFalse(c.success_latched)
        c._update_success_checker(0.25, True)
        self.assertFalse(c.success_latched)
        c._update_success_checker(0.25, True)
        self.assertTrue(c.success_latched)
        self.assertGreaterEqual(c.stops, 3)

    def test_one_bad_hold_cycle_fails(self):
        c = self.make_controller(phase="post_push_hold", valid=False, motion=True)
        c._update_success_checker(0.5, True)
        self.assertEqual(c.failure_reason, "grasp_lost_during_post_hold")
        self.assertFalse(c.success_latched)

    def test_override_changes_only_four_gripper_joints(self):
        c = self.make_controller()
        c._gripper_hold_targets = {1: 0.1, 3: 0.3, 5: 0.5, 7: 0.7}
        for action in (torch.arange(10, dtype=torch.float32), torch.arange(20, dtype=torch.float32).reshape(2, 10)):
            original = action.clone()
            result = c.override_action(action)
            torch.testing.assert_close(action, original)
            for index in (0, 2, 4, 6, 8, 9):
                torch.testing.assert_close(result[..., index], original[..., index])
            for index, target in c._gripper_hold_targets.items():
                torch.testing.assert_close(result[..., index], torch.full_like(result[..., index], target))
        c._gripper_hold_targets = None
        self.assertIs(c.override_action(action), action)


if __name__ == "__main__":
    unittest.main()
