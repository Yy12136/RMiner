"""Policy loading, checkpoint compatibility, and inference-time invariants."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from task_specs import TaskSpec

LEGACY_CONFIG_FIELDS = {"rtc_training_max_delay"}

# Checkpoints trained through dataset conversions that renamed the three G1
# cameras. X-VLA training used head -> image, left_wrist -> image2,
# right_wrist -> image3 (xvla_training_audit_20260908.md).
CHECKPOINT_CAMERA_ALIASES: dict[str, dict[str, str]] = {
    "xvla": {
        "observation.images.head": "observation.images.image",
        "observation.images.left_wrist": "observation.images.image2",
        "observation.images.right_wrist": "observation.images.image3",
    },
}


def camera_key_map(policy_type: str) -> dict[str, str] | None:
    return CHECKPOINT_CAMERA_ALIASES.get(policy_type)


class IllegalPolicyAction(RuntimeError):
    pass


def feature_shapes(features: dict) -> dict[str, tuple[int, ...]]:
    return {key: tuple(int(value) for value in feature.shape) for key, feature in features.items()}


def validate_checkpoint_features(config: PreTrainedConfig, spec: TaskSpec) -> None:
    inputs, outputs = feature_shapes(config.input_features), feature_shapes(config.output_features)
    key_map = camera_key_map(config.type)
    expected_inputs = spec.expected_inputs
    if key_map is not None:
        expected_inputs = {key_map.get(key, key): value for key, value in expected_inputs.items()}
    if inputs != expected_inputs:
        raise ValueError(f"Checkpoint input mismatch for {spec.key}: expected {expected_inputs}, got {inputs}")
    if outputs != spec.expected_outputs:
        raise ValueError(f"Checkpoint output mismatch for {spec.key}: expected {spec.expected_outputs}, got {outputs}")


class CompatibleCheckpoint:
    """Read old checkpoints without ever mutating their config.json."""

    def __init__(self, original: Path) -> None:
        self.original = original.resolve()
        self._temporary: tempfile.TemporaryDirectory | None = None
        self.path = self.original
        raw = json.loads((self.original / "config.json").read_text(encoding="utf-8"))
        removed = sorted(set(raw).intersection(LEGACY_CONFIG_FIELDS))
        if removed:
            self._temporary = tempfile.TemporaryDirectory(prefix="lerobot-eval-compat-")
            compat = Path(self._temporary.name)
            for item in self.original.iterdir():
                if item.name == "config.json":
                    continue
                os.symlink(item, compat / item.name, target_is_directory=item.is_dir())
            for key in removed:
                raw.pop(key, None)
            (compat / "config.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            (compat / "COMPATIBILITY.json").write_text(
                json.dumps({"source": str(self.original), "removed_fields": removed}, indent=2) + "\n",
                encoding="utf-8",
            )
            self.path = compat

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


class SyncPolicyRunner:
    def __init__(self, policy, preprocessor, postprocessor, device: torch.device, spec: TaskSpec, key_map: dict[str, str] | None = None) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.spec = spec
        self.key_map = key_map
        self._absolute_action_queue: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        self._absolute_action_queue.clear()
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _rename_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.key_map:
            return observation
        return {self.key_map.get(key, key): value for key, value in observation.items()}

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        observation = self._rename_observation(observation)
        if bool(getattr(self.policy.config, "use_relative_actions", False)):
            if not self._absolute_action_queue:
                with torch.inference_mode(), (
                    torch.autocast(device_type="cuda")
                    if self.device.type == "cuda" and bool(self.policy.config.use_amp)
                    else nullcontext()
                ):
                    prepared = prepare_observation_for_inference(
                        dict(observation), self.device, self.spec.instruction, "unitree_g1_29_dex1"
                    )
                    chunk = self.policy.predict_action_chunk(self.preprocessor(prepared))
                    steps = int(self.policy.config.n_action_steps)
                    # Important: convert the complete relative chunk once, with
                    # the state that produced it. Cached actions are already absolute.
                    absolute = self.postprocessor(chunk[:, :steps])
                if absolute.ndim != 3 or absolute.shape[0] != 1 or absolute.shape[2] != self.spec.action_dim:
                    raise IllegalPolicyAction(f"Invalid relative action chunk {tuple(absolute.shape)}")
                self._absolute_action_queue.extend(absolute.transpose(0, 1))
            result = self._absolute_action_queue.popleft().squeeze(0)
        else:
            result = predict_action(
                observation=observation, policy=self.policy, device=self.device,
                preprocessor=self.preprocessor, postprocessor=self.postprocessor,
                use_amp=bool(self.policy.config.use_amp), task=self.spec.instruction,
                robot_type="unitree_g1_29_dex1",
            ).squeeze(0)
        # X-VLA inference runs in bfloat16; numpy has no bf16 dtype.
        action = result.detach().float().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if action.shape != (self.spec.action_dim,) or not np.isfinite(action).all():
            raise IllegalPolicyAction(f"Illegal policy action shape={action.shape}")
        return action, time.perf_counter() - started

    def close(self) -> None:
        return


def load_policy(checkpoint: Path, spec: TaskSpec):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal policy evaluation")
    compat = CompatibleCheckpoint(checkpoint)
    try:
        config = PreTrainedConfig.from_pretrained(str(compat.path))
        validate_checkpoint_features(config, spec)
        config.device = "cuda"
        policy = get_policy_class(config.type).from_pretrained(
            pretrained_name_or_path=str(compat.path), config=config
        )
        policy.to("cuda").eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(compat.path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        runner = SyncPolicyRunner(
            policy, preprocessor, postprocessor, torch.device("cuda"), spec,
            key_map=camera_key_map(config.type),
        )
        return config, runner, compat
    except Exception:
        compat.close()
        raise
