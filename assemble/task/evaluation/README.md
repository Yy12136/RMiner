# Task02–06 formal evaluation framework

Task01 remains the regression baseline. Task02–06 use this shared, configuration-driven evaluator.

Run from the corresponding task directory:

```bash
./benchmark.sh eval \
  --checkpoint experiments/weights/diffusion_policy/030000/pretrained_model \
  --episodes 20 --seed 42 --timeout 60 --inference sync
```

Add `--headless` for offscreen rendering. A one-episode smoke test uses `--episodes 1`; `Ctrl-C` requests a safe stop and the supervisor terminates every owned process group.

The supervisor loads the CUDA policy first, then starts Isaac Sim with CPU physics and GPU/Vulkan cameras, then the DDS bridge. Results are written below each task's `experiments/results/<model>/<step>/` directory. Each run contains `config.json`, `episodes.jsonl`, `summary.json`, deterministic reset payloads, and three-camera LeRobot v3 rollout/video output.

Key invariants:

- PI0.5 relative action chunks are postprocessed exactly once at chunk creation.
- Legacy checkpoint fields are removed only in a temporary compatibility view; source checkpoints are not modified.
- Right-arm-only tasks lock the measured left arm and left gripper.
- Task04 projects mobile actions onto stop or straight-forward and has a 0.25 s command timeout.
- Task06 compares the seven object-pose observation values with fresh simulator telemetry every frame.
- Only success and timeout are valid episodes; infrastructure errors retry the same seed once.

Run unit tests with:

```bash
cd <BENCH_ROOT>/assemble/task/evaluation
<BENCH_ROOT>/policy/lerobot/.venv/bin/python -m unittest -v test_common.py
```

Task05 uses a true kinematic target for the valve under formal CPU physics and falls back to a generic pose write only for manual CUDA simulation, whose rigid-body backend does not implement kinematic targets. Isaac Sim 5.1 can still emit one non-fatal `wakeUp: Body must be non-kinematic` diagnostic for each constrained valve/handwheel assembly reset; reset pose, joint anchor, determinism, and rollout remain valid.
