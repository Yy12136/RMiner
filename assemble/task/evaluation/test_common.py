from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from policy_runtime import CompatibleCheckpoint, SyncPolicyRunner, validate_checkpoint_features
from safety import apply_right_only_lock, project_mobile_action, sanitize_joint_action, synchronized_displacement_error
from supervisor import terminate_group
from task_specs import SPECS
from utils import summarize


class Resettable:
    def reset(self):
        pass
    def __call__(self, value):
        return value


class FakePolicy(Resettable):
    config = SimpleNamespace(use_relative_actions=True, use_amp=False, n_action_steps=2)
    def predict_action_chunk(self, _batch):
        return torch.tensor([[[1.0] * 16, [2.0] * 16, [3.0] * 16]])


class CountingPost(Resettable):
    def __init__(self): self.calls = 0
    def __call__(self, chunk):
        self.calls += 1
        return chunk + 10.0


class CommonTests(unittest.TestCase):
    def test_declared_dimensions(self):
        self.assertEqual((SPECS["task04"].state_dim, SPECS["task04"].action_dim), (30, 19))
        self.assertEqual((SPECS["task06"].state_dim, SPECS["task06"].action_dim), (23, 16))
        self.assertTrue(SPECS["task02"].right_arm_only)

    def test_feature_validation(self):
        spec = SPECS["task04"]
        feature = lambda shape: SimpleNamespace(shape=shape)
        cfg = SimpleNamespace(
            input_features={key: feature(shape) for key, shape in spec.expected_inputs.items()},
            output_features={key: feature(shape) for key, shape in spec.expected_outputs.items()},
        )
        validate_checkpoint_features(cfg, spec)
        cfg.output_features["action"] = feature((16,))
        with self.assertRaises(ValueError): validate_checkpoint_features(cfg, spec)

    def test_relative_chunk_is_anchored_once(self):
        post = CountingPost()
        runner = SyncPolicyRunner(FakePolicy(), Resettable(), post, torch.device("cpu"), SPECS["task02"])
        with mock.patch("policy_runtime.prepare_observation_for_inference", return_value={}):
            first, _ = runner.action({}); second, _ = runner.action({})
        self.assertEqual(post.calls, 1)
        np.testing.assert_allclose(first, np.full(16, 11.0))
        np.testing.assert_allclose(second, np.full(16, 12.0))

    def test_checkpoint_compatibility_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "checkpoint"; source.mkdir()
            original = {"type": "pi05", "rtc_training_max_delay": 3}
            (source / "config.json").write_text(json.dumps(original))
            (source / "model.safetensors").write_bytes(b"x")
            compat = CompatibleCheckpoint(source)
            self.assertNotEqual(compat.path, source)
            self.assertNotIn("rtc_training_max_delay", json.loads((compat.path / "config.json").read_text()))
            self.assertEqual(json.loads((source / "config.json").read_text()), original)
            compat.close()

    def test_right_arm_lock_and_joint_limits(self):
        locked = np.arange(8, dtype=float)
        action = apply_right_only_lock(np.full(16, 99.0), locked)
        np.testing.assert_allclose(action[:7], locked[:7]); self.assertEqual(action[14], locked[7])
        safe = sanitize_joint_action(action, np.zeros(16), np.full(14, -1.0), np.ones(14), 3.0, 1 / 30, 0.18)
        self.assertLessEqual(float(np.max(np.abs(safe[:14]))), 0.100001)
        self.assertLessEqual(float(np.max(np.abs(safe[14:]))), 0.180001)

    def test_mobile_projection_tie_stops(self):
        projected, changed = project_mobile_action(np.asarray([0.6, 0.6, 0.0]))
        np.testing.assert_allclose(projected, np.zeros(3)); self.assertTrue(changed)
        projected, changed = project_mobile_action(np.asarray([1.2, 1.2, 0.0]))
        np.testing.assert_allclose(projected, [1.2, 1.2, 0.0]); self.assertFalse(changed)

    def test_task04_synchronization_metric(self):
        before, after = np.zeros(30), np.zeros(30)
        after[16], after[23] = 0.01, 0.01
        self.assertLessEqual(synchronized_displacement_error(before, after), 1e-12)
        after[23] = 0.02
        self.assertGreater(synchronized_displacement_error(before, after), 0.009)

    def test_summary_counts_only_valid(self):
        summary = summarize([
            {"valid": True, "status": "success", "elapsed_seconds": 2},
            {"valid": True, "status": "timeout", "elapsed_seconds": 3},
            {"valid": False, "status": "infrastructure_error", "elapsed_seconds": 0},
        ], [0.01, 0.02])
        self.assertEqual(summary["valid_episodes"], 2); self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["failure_episodes"], 0)

    def test_summary_counts_grasp_failure_as_valid(self):
        summary = summarize([
            {"valid": True, "status": "success", "elapsed_seconds": 2},
            {"valid": True, "status": "failure", "elapsed_seconds": 1},
            {"valid": True, "status": "timeout", "elapsed_seconds": 3},
            {"valid": False, "status": "infrastructure_error", "elapsed_seconds": 0},
        ], [0.01, 0.02])
        self.assertEqual(summary["valid_episodes"], 3)
        self.assertEqual(summary["successful_episodes"], 1)
        self.assertEqual(summary["failure_episodes"], 1)
        self.assertEqual(summary["timeout_episodes"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)


    def test_robot_controller_action_override(self):
        control_root = Path(__file__).resolve().parents[2] / "third_party/unitree_sim_isaaclab"
        sys.path.insert(0, str(control_root))
        from layeredcontrol.robot_control_system import ControlConfig, RobotController

        class RobotData:
            joint_names = [f"j{i}" for i in range(4)]

        class FakeEnv:
            device = torch.device("cpu")
            scene = {"robot": SimpleNamespace(data=RobotData())}
            def __init__(self): self.seen = None
            def step(self, action): self.seen = action.clone()

        class Provider:
            name = "unit-test"
            def start(self): pass
            def stop(self): pass
            def cleanup(self): pass
            def get_action(self, _env): return torch.zeros(4)

        env = FakeEnv()
        controller = RobotController(env, ControlConfig(step_hz=1000))
        controller.set_action_provider(Provider())
        controller.set_action_override(lambda action: action + torch.tensor([1.0, 2.0, 3.0, 4.0]))
        controller.start(); controller.step(); controller.stop()
        self.assertIsNotNone(env.seen)
        np.testing.assert_allclose(env.seen.numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_task04_grasp_protocol_contract(self):
        source = Path(__file__).resolve().parents[2] / "third_party/unitree_sim_isaaclab/tasks/g1_tasks/hand_cart_push_g1_29dof_dex1/hand_cart_push_controller.py"
        text = source.read_text()
        for phase in ("acquiring_grasp", "pushing", "post_push_hold", "success", "failure"):
            self.assertIn(f'"{phase}"', text)
        self.assertIn("grasp_lost_during_push", text)
        self.assertIn("grasp_lost_during_post_hold", text)
        self.assertIn("POST_PUSH_HOLD_SECONDS", text)
        self.assertIn("def override_action", text)
        self.assertIn("def is_terminal_frozen", text)
        evaluator_text = (Path(__file__).resolve().parent / "evaluator.py").read_text()
        self.assertIn("--failure-marker-file", evaluator_text)
        self.assertIn('status = "failure"', evaluator_text)
        self.assertIn('status in ("success", "failure", "timeout")', evaluator_text)

    def test_supervisor_kills_orphaned_process_group(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import os,time\npid=os.fork()\nif pid: os._exit(0)\ntime.sleep(60)"],
            start_new_session=True,
        )
        process.wait(timeout=5)
        time.sleep(0.05)
        os.killpg(process.pid, 0)  # the orphaned child still owns the PGID
        terminate_group(process, grace=0.2)
        with self.assertRaises(ProcessLookupError):
            os.killpg(process.pid, 0)


if __name__ == "__main__": unittest.main()
