# G1 Tabletop Grasp Task

This task keeps the tracked G1 upper body, official Unitree Quest 3/3S IK,
Dex1 grippers, and all three cameras from the mobile-drilling setup.  It adds a
static worktable with an upright `leiguan` and a flexible `chaganshengzi`
assembly standing in a small static holder.

## Unified collect-train-eval benchmark

The recommended entry point is `benchmark.sh`. The simulator remains in its
own terminal so collection and evaluation can attach to the same local DDS and
camera streams without changing the existing task behavior.

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./benchmark.sh sim
```

Terminal 2, choose one command:

```bash
# Existing Quest 3/3S collection state machine (B/A/s/q)
./benchmark.sh collect

# All 51 local v3.0 demonstrations, no validation split
./benchmark.sh train \
  --policy act \
  --steps 100000 \
  --batch-size 16 \
  --save-freq 10000

# Automatic simulated evaluation
./benchmark.sh eval \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --episodes 20 \
  --timeout 60 \
  --seed 42 \
  --initial-gripper 4.0 \
  --inference sync
```

Training accepts any policy registered by this LeRobot checkout. Add `--wandb`
to enable W&B using credentials already present in the environment; the script
does not contain or save an API key. Arguments after `--` are forwarded to
`lerobot-train`. PyAV uses zero worker processes by default because that is the
stable setting verified on this machine; override it with `--num-workers`.

Evaluation validates the checkpoint's exact inputs before issuing DDS commands:
16 state values, three `[3,480,640]` RGB images, and one 16-dimensional action.
After loading, press `r` once in the terminal. Episodes then run, score, save,
and deterministically reset without further input. Press `s` to abort the
current episode without adding it to the denominator, or `q` to return both
arms home and stop the run. `sync` supports matching LeRobot policies through
`select_action()` and the checkpoint's own processors. `rtc` is accepted only
when a policy explicitly advertises the LeRobot RTC action-chunk interface.

Each run is local-only and is written below:

```text
outputs/eval/<policy>_<checkpoint-step>_<timestamp>/
```

`config.json` records the model and control settings, `episodes.jsonl` records
every valid/invalid attempt, and `summary.json` contains success rate, Wilson
95% interval, timing, timeout counts, and inference latency. `rollouts/` is a
loadable LeRobot Dataset v3.0 containing the executed (safety-clipped) action,
state, instruction, and all three H.264 camera streams for both successes and
timeouts. Infrastructure failures retry once; a second failure stops the run.
An illegal/NaN action immediately holds the last safe command and stops.

The evaluation bridge is local-only (a mode-0600 Unix socket), always uses DDS
domain 1 for Isaac Sim, clamps arms to the official G1 URDF hard limits and
3 rad/s, and clamps Dex1 to `[0,5.4]` with at most `0.18 rad` change per 30 Hz
tick. The tracked base and lift remain fixed during policy rollout.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./run_task.sh
```

Terminal 2, after the Isaac Sim window is ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./run_quest3s.sh
```

Open <https://vuer.ai?grid=False> in Quest 3/3S.  In the teleoperation
session, press the right controller `B` button to start arm tracking and a new
LeRobot Dataset v3.0 episode.

## Controls

- `W` / `S`: tracked base forward / reverse
- `A` / `D`: tracked base turn left / right
- `T` / `Y`: lift up / down
- `R` in the Isaac Sim window: reset the G1, tracked base/lift, `leiguan`,
  rod, and all 26 rope links
- Right controller `B`: start teleoperation and begin a new
  episode; recording never restarts automatically after a reset
- Right controller `A`: accept and save the current episode,
  perform a full reset, and wait for the next right-controller `B`
- `s` in the teleoperation terminal: discard the current episode, perform a
  full reset, and wait for the next right-controller `B`
- `q` in the teleoperation terminal: discard any unfinished episode, stop,
  and return the arms home
- `P` in the Isaac Sim window: save the current rod pose relative to Leiguan,
  together with the complete rod/rope/holder world poses, to
  `saved_inserted_object_pose.json` (this does not reset or move anything)
- Quest controller poses: control the two G1 arms through the official IK
- Quest triggers: control the two Dex1 grippers
- The official XR object-reset action resets `leiguan`, the rod, and all 26
  rope links; reset-all also restores the robot and tracked base.
- On startup and every object/full reset, `leiguan` and the rod assembly receive
  independent uniform XY offsets in `[-5 cm, +5 cm]` around their saved centers.
  The rod, all rope links, and the kinematic holder move together through
  PhysX, followed by a short contact-settling window; Z positions and
  orientations are unchanged. Candidates are resampled if Leiguan or any
  rod/rope/holder sample point is within 6 cm of either gripper in its home or
  pre-reset pose. Set `G1_OBJECT_GRIPPER_CLEARANCE_M` before `run_task.sh` to
  adjust this clearance.

`run_task.sh` owns the Isaac Sim process group. Pressing `Ctrl+C` in that
terminal closes Isaac Sim, its DDS/camera-server children, and the separate
three-camera viewer; if normal termination exceeds 10 seconds, the script
forces the remaining simulator process group to exit.

## Success and collection progress

Success is detected only when the portion of the rigid rod inside Leiguan is
fully within its cylindrical cavity and its insertion depth is greater than
`0.12 m`. On success, physics, robot commands, new camera frames, and new
recorded samples are frozen. The three-camera window keeps the final frame and
shows `SUCCESS`. Communication remains active: right-controller `A` still
accepts/saves the pending episode, while terminal `s` still discards it; either
choice performs the existing reset and clears the freeze.

The default collection target is 50 accepted episodes. The teleoperation
terminal prints `Saved episodes: N/50` at startup and after every successful
`A` save. Discarded or interrupted staging episodes do not increment this
count. Existing v3 data is resumed by default, so restarting the task continues
from the next episode rather than overwriting prior data. Set
`G1_LEROBOT_RESUME=0` to require a new/empty dataset path, or set
`G1_LEROBOT_TARGET_EPISODES` to override the display target.

## Assets

- Scene: `<BENCH_ROOT>/assemble/assets-new/scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
- Robot: `<BENCH_ROOT>/assemble/assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- Static table trolley: `<BENCH_ROOT>/assemble/assets-new/object/TableTrolley_B/TableTrolley_B01_01.usd` (top surface aligned to 0.84 m)
- Upright object: `<BENCH_ROOT>/assemble/assets-new/object/leiguan.usd`
- Flexible object: `<BENCH_ROOT>/assemble/assets-new/object/chaganshengzi_standing.usd`
- Kinematic holder: `<BENCH_ROOT>/assemble/assets-new/object/chaganshengzi_holder.usd`
- Previous flat version (kept as a fallback): `<BENCH_ROOT>/assemble/assets-new/object/chaganshengzi_tabletop.usd`

The simulator starts a local window containing the head, left-wrist, and
right-wrist camera streams.  Set `G1_SHOW_CAMERA_VIEWPORTS=0` to disable it.

## Isaac Sim 3D mine-dust preview

The optional dust preview is disabled during normal teleoperation and data
collection. It uses RTX global volumetric scattering with animated 3D density
noise, depth-aware fog absorption, a head-camera beam, and USD spatial points.
The particles therefore have camera-relative parallax and scene occlusion;
there is no 2D Gaussian-noise or line-particle overlay.

Dust selection is explicit and opt-in:

```text
G1_DUST_MODE=off       # default; no dust
G1_DUST_MODE=legacy    # RTX haze plus animated USD points
G1_DUST_MODE=flow      # RTX ambient haze plus smoke-only NVIDIA Flow
G1_DUST_PRESET=low|medium|heavy
```

The legacy `G1_ENABLE_3D_DUST=1` flag remains equivalent to
`G1_DUST_MODE=legacy` when `G1_DUST_MODE` is not set. To render the fixed-seed
Task01 episode-0 comparison POC, run `./run_flow_dust_poc.sh`; it refuses to
start while another `sim_main.py` process or a formal evaluation supervisor is
active. Flow uses the locally installed NVIDIA Fire USD only as a complete
known-good prim hierarchy. In this Isaac Sim 6.0.1/Flow 110 headless renderer,
direct smoke with combustion disabled produced no visible volume, so the POC
uses the local Fire solver as a compatibility fallback, lowers its dynamics,
and replaces the fire map with a neutral brown-gray dust colormap. This is a
documented visual approximation rather than a strict smoke-only simulation;
the installed preset file itself is never modified.

Generate a standalone 15-second head-camera preview with:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./run_3d_dust_preview.sh
```

The script launches an offscreen task, records the simulator camera, and then
closes its Isaac Sim process. Set `G1_DUST_PREVIEW_OUTPUT` or
`G1_DUST_PREVIEW_SECONDS` to override the output path or duration. Renderer
strength can be tuned with `G1_DUST_VOLUME_DENSITY`,
`G1_DUST_TRANSMITTANCE_DISTANCE`, `G1_DUST_PARTICLE_COUNT`, and
`G1_DUST_HEADLAMP_INTENSITY`. The default replay uses 5000 spatial particles;
`G1_DUST_PARTICLE_COUNT` accepts values from 1 through 10000.

To replay a recorded LeRobot v3 episode through G1/Dex1 DDS control and render
neutral-gray dust inside Isaac Sim, run:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./run_3d_dust_replay.sh 0
```

The argument is the episode index. The replay program reads state/action data
from the Parquet shard, never decodes or edits the source MP4 files, and records
fresh head, left-wrist, right-wrist, and three-camera-grid videos from the
simulator. Override the input and output roots with `G1_DUST_REPLAY_DATASET` and
`G1_DUST_REPLAY_OUTPUT_ROOT`.

New collections started with the current `run_quest3s.sh` save only robot
state/action per frame. At episode start, a replay snapshot containing the
Leiguan/rod/holder initial poses is captured from `rt/sim_state`. Existing datasets made
before this change do not contain those per-frame object poses and must not be
used as the output directory for the new schema; use the default
`detonator_wire_insertion_replay_ready` directory (or set
`G1_LEROBOT_DATASET_DIR` to an empty path).
Each episode also stores a replay snapshot sidecar under `meta/replay/`,
captured at episode start, containing the initial Leiguan, rod, and holder poses.

For balanced 50-to-250 LeRobot v3 augmentation across four lighting profiles,
with clean/dust pairs rendered during Isaac Sim replay, see
`AUGMENTATION.md` and run `./run_lerobot_augmentation.sh --help`.

## LeRobot Dataset v3.0 data

By default, accepted episodes are stored in:

```text
<BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion_with_object_pose
```

The dataset records synchronized 30 Hz samples containing 14 G1 arm joint
positions, two Dex1 gripper positions, matching commanded actions, and the
head/left-wrist/right-wrist RGB videos. Object initial poses are stored in the
episode replay snapshot rather than repeated in every frame.
Its fixed language instruction is:
`Insert the detonator lead wire into the detonator opening.`

Set `G1_LEROBOT_DATASET_DIR` before running `run_quest3s.sh` to choose another
output directory. Right controller `A` commits an episode immediately as
LeRobot v3 data/video/episode Parquet shards. Terminal `s`, `q`, or an
interrupted process discards the pending episode. A transaction journal repairs
an interrupted commit the next time the writer starts.

The default continuation settings are equivalent to:

```bash
G1_LEROBOT_RESUME=1 G1_LEROBOT_TARGET_EPISODES=50 ./run_quest3s.sh
```

For this task, the simulator enables GPU-compatible speculative CCD on both
Leiguan and the Dex1 finger links, with two PhysX substeps and increased
contact iterations to prevent the thin tube wall from tunneling through a
closing gripper.

The original v2.1 dataset remains unchanged at:

```text
<BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v21/detonator_wire_insertion
```

## LeRobot ACT smoke training

The recorder now writes the v3.0 training dataset directly at:

```text
<BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task/data/lerobot_v30/detonator_wire_insertion
```

Run the 500-step local ACT smoke test with:

```bash
cd <BENCH_ROOT>/assemble/task/g1_tabletop_grasp_task
./run_lerobot_act_smoke.sh
```

The script uses the independent Python 3.12 environment in
`<BENCH_ROOT>/policy/lerobot/.venv`, disables Hub and Weights &
Biases uploads, and writes checkpoints and logs below the LeRobot repository's
`outputs/train` directory.  It starts with batch size 8 and only retries with
4 or 2 if CUDA reports an out-of-memory error.  PyAV decoding stays in the
main data-loader process because this machine's multi-process PyAV workers can
abort while Python is shutting down after a completed training run.

To start a new 1000-step online Weights & Biases run:

```bash
LEROBOT_TRAIN_STEPS=1000 \
LEROBOT_WANDB_ENABLE=true \
LEROBOT_WANDB_PROJECT=g1-tabletop-grasp \
./run_lerobot_act_smoke.sh
```
