from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

import policy_eval


class FakeRelativePreprocessor:
    def __init__(self) -> None:
        self.calls = 0
        self.state = None
        self.resets = 0

    def __call__(self, observation):
        self.calls += 1
        self.state = observation["observation.state"]
        return observation

    def reset(self):
        self.resets += 1
        self.state = None


class FakeRelativePostprocessor:
    def __init__(self, preprocessor) -> None:
        self.preprocessor = preprocessor
        self.calls = 0
        self.resets = 0

    def __call__(self, chunk):
        self.calls += 1
        return chunk + self.preprocessor.state.unsqueeze(1)

    def reset(self):
        self.resets += 1


class FakeRelativePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            use_relative_actions=True,
            use_amp=False,
            n_action_steps=3,
        )
        self.predict_calls = 0
        self.resets = 0

    def predict_action_chunk(self, _observation):
        self.predict_calls += 1
        return torch.tensor([[[1.0], [2.0], [3.0]]])

    def reset(self):
        self.resets += 1


class SyncPolicyRunnerTest(TestCase):
    def test_relative_chunk_is_anchored_once(self):
        prepare = lambda observation, *_args: {
            "observation.state": torch.tensor(
                [[float(observation["observation.state"][0])]]
            )
        }
        with patch.object(policy_eval, "prepare_observation_for_inference", prepare):
            policy = FakeRelativePolicy()
            preprocessor = FakeRelativePreprocessor()
            postprocessor = FakeRelativePostprocessor(preprocessor)
            runner = policy_eval.SyncPolicyRunner(
                policy, preprocessor, postprocessor, torch.device("cpu")
            )

            outputs = [
                runner.action({"observation.state": np.array([state], dtype=np.float32)})[0]
                for state in (10.0, 20.0, 30.0)
            ]

            self.assertEqual([float(output[0]) for output in outputs], [11.0, 12.0, 13.0])
            self.assertEqual(policy.predict_calls, 1)
            self.assertEqual(preprocessor.calls, 1)
            self.assertEqual(postprocessor.calls, 1)

            # The next chunk uses the new observation as its own fixed anchor.
            self.assertEqual(
                float(runner.action({"observation.state": np.array([40.0])})[0][0]),
                41.0,
            )
            self.assertEqual(policy.predict_calls, 2)
            self.assertEqual(preprocessor.calls, 2)

            runner.reset()
            self.assertFalse(runner._absolute_action_queue)
            self.assertEqual(policy.resets, 1)
            self.assertEqual(preprocessor.resets, 1)
            self.assertEqual(postprocessor.resets, 1)

    def test_absolute_policy_keeps_existing_single_step_path(self):
        calls = []

        def fake_predict_action(**kwargs):
            calls.append(kwargs["observation"])
            return torch.tensor([[7.0]])

        with patch.object(policy_eval, "predict_action", fake_predict_action):
            policy = SimpleNamespace(
                config=SimpleNamespace(use_relative_actions=False, use_amp=False),
                reset=lambda: None,
            )
            processor = SimpleNamespace(reset=lambda: None)
            runner = policy_eval.SyncPolicyRunner(
                policy, processor, processor, torch.device("cpu")
            )

            output, _ = runner.action({"observation.state": np.array([1.0])})
            self.assertEqual(float(output[0]), 7.0)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    main()
