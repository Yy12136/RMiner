# LeRobot v3 video-only augmentation

Entry point:

```text
<BENCH_ROOT>/assemble/task/augmentation/run_lerobot_augmentation.sh
```

The command never edits the input dataset and never re-simulates the robot or
task objects. Every variant is generated from the successful source video and
written to new MP4 files. State/action are copied exactly. Low/warm/cool light
uses deterministic photometric curves; the directional headlamp uses a smooth
spatial beam; dusty variants add temporally coherent neutral-gray haze plus
5000 depth-layered sub-pixel particles/short streaks. Isaac Sim and DDS are not
started.

## Default 50-to-250 layout

```text
episode 000-049: original (copy)
episode 050-074: low illumination (video transform)
episode 075-099: low illumination + dust (video transform)
episode 100-124: warm work lamp (video transform)
episode 125-149: warm work lamp + dust (video transform)
episode 150-174: cool LED (video transform)
episode 175-199: cool LED + dust (video transform)
episode 200-224: directional headlamp (video transform)
episode 225-249: directional headlamp + dust (video transform)
```

Each lighting profile uses all 50 source trajectories exactly once across its
clean/dust pair. The two 25-episode halves alternate between profiles, so every
source trajectory receives four augmented versions, including two clean-air
and two dusty versions.

For a 50-to-250 run, 50 originals are copied and all 200 augmented episodes use
the video path. The manipulation remains successful because its pixels,
state, and action come from the accepted demonstration rather than open-loop
physics replay.

## Current task example

Preview and validate the plan without creating data:

```bash
cd <BENCH_ROOT>/assemble/task/augmentation
./run_lerobot_augmentation.sh \
  --input <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion_50_20260822 \
  --output <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion_250_video_only_20260823 \
  --source-start 0 \
  --source-count 50 \
  --mode dry-run
```

Generate or resume all 250 episodes:

```bash
cd <BENCH_ROOT>/assemble/task/augmentation
./run_lerobot_augmentation.sh \
  --input <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion_50_20260822 \
  --output <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion_250_video_only_20260823
```

Those two paths are sufficient for any compatible LeRobot v3 dataset. The
augmenter reads FPS, instruction, state/action names, camera keys, resolution,
and robot type from the input; it does not need a Gym ID, task launcher, reset
protocol, or robot-specific replay adapter.

Rerun the same command after an interruption. The writer validates the plan and
continues at the next incomplete episode. It refuses to resume if the input,
output, source range, particle count, seed, or episode plan differs.
An older output that contains only the copied originals can be migrated safely
to this video-only plan. If it already contains augmented episodes generated
by a replay/hybrid pipeline, the command refuses to mix them and asks for a new
output.

## Inputs

- `--input PATH` (required): source LeRobot Dataset v3.0 directory.
- `--output PATH` (required): new or resumable output directory; it must differ
  from the input.
- `--source-start N`: first source episode, default `0`.
- `--source-count N`: even number of source episodes, default `50`. The output
  contains `5 * N` episodes.
- `--particle-count N`: gray spatial-particle count, default `5000`, range
  `1..10000`.
- `--seed N`: deterministic reset/effect seed, default `260823`.
- `--mode dry-run`: print the complete episode layout and generation path
  without writing data.

## Outputs

The output is a directly loadable LeRobot v3 dataset containing:

- `data/chunk-XXX/file-XXX.parquet`
- `videos/{camera_key}/chunk-XXX/file-XXX.mp4`
- `meta/episodes/chunk-XXX/file-XXX.parquet`
- `meta/tasks.parquet`, `meta/info.json`, and `meta/stats.json`
- `meta/augmentation_manifest.json`, mapping every output episode to its source,
  lighting profile, dust flag, and completion state

The language instruction, state/action names, FPS, camera keys, resolution, and
robot type are read from the input dataset instead of being hard-coded.

Because no physical replay occurs, per-frame object/rope/base transforms are
not required. The output keeps the accepted source trajectory intact.
