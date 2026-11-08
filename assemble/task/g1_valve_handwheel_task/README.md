# G1 Valve Handwheel Rotation Task

This task uses the tracked G1 upper body and official Quest 3/3S controller to
rotate an already-installed blue handwheel on a fixed three-way ball valve by
one complete revolution.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_valve_handwheel_task
./run_task.sh
```

Terminal 2, after the simulator reports ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_valve_handwheel_task
./run_quest3s.sh
```

Open <https://vuer.ai?grid=False> in Quest 3/3S. The robot does not follow the
controllers until the right-controller `B` button is pressed.

## Controls

- `W` / `S`: tracked base forward / reverse
- `A` / `D`: tracked base turn left / right
- `T` / `Y`: lift up / down
- `I`: bend the G1 waist forward (up to 25 degrees)
- `K`: return the G1 waist toward upright
- `R` in Isaac Sim: reset robot, base/lift, and the installed valve assembly
- Right controller `B`: start teleoperation and a new episode
- Right controller `A`: accept/save the episode, return home, and reset all
- `s` in the teleoperation terminal: discard, return home, and reset all
- `q` in the teleoperation terminal: discard the unfinished episode and exit
- Quest triggers: control the two Dex1 grippers

The head, left-wrist, and right-wrist camera streams are displayed together by
default. Set `G1_SHOW_CAMERA_VIEWPORTS=0` to disable the viewer.

## Task behavior

There is no task table. The three-way valve is fixed to the RealScene04 wall
that was behind the previous robot pose. Its former floor-contacting face is
now against the wall at about `(1.186, 1.077, 1.25) m`. The tracked G1 is
rotated 180 degrees and moved close to that wall, leaving the upper-body mount
about `0.85 m` from the wall. The handwheel starts seated on the valve stem. A
physical revolute joint connects the two assets, so the valve is a fixed base
and the handwheel can move only by rotating about their shared horizontal wall
normal. Connected-body collision is disabled to avoid bore/stem jitter, while
both assets retain collision against the robot and scene.

At startup and every full reset, the complete valve-handwheel assembly shares
one random wall-plane translation: `±5 cm` horizontally along the wall and
`±5 cm` vertically. The valve and handwheel never randomize independently,
so their initial seated relationship and revolute joint remain intact.

The task-specific tracked lift initializes at `0.1218405274 m`, captured from
the live simulator pose on 2026-08-22. This places the G1 root at approximately
`1.112308514 m`. Other G1 tasks retain their own saved lift heights.

Both task assets are uniformly enlarged to 1.8 times their original size. The
task-specific handwheel has a 46.8 mm visual and collision bore around the 28.8 mm
valve stem. Success requires a net handwheel rotation of 60 degrees from the
reset orientation in
either direction. Reversing direction subtracts from the accumulated angle.

On success, the simulator prints `SUCCESS` and freezes physics, commands, new
camera frames, and recording samples. Right-controller `A` still saves the
episode; terminal `s` still discards it. Either action resets the task.

## LeRobot Dataset v3.0

Accepted episodes are stored by default at:

```text
<BENCH_ROOT>/assemble/task/g1_valve_handwheel_task/data/lerobot_v30/valve_handwheel_rotation_replay_ready
```

Collection resumes an existing dataset and displays a 50-episode target. The
recorded instruction is:

```text
Rotate the blue handwheel on the three-way ball valve by one full turn.
```

Use `G1_LEROBOT_DATASET_DIR`, `G1_LEROBOT_TARGET_EPISODES`, or
`G1_LEROBOT_RESUME=0` to override these defaults.

Each accepted episode stores an initial replay snapshot under `meta/replay/`
with the world XYZ/WXYZ poses of the valve and handwheel. These poses are not
repeated in every data frame.

## Assets

- Scene: `<BENCH_ROOT>/assemble/assets-new/scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
- Robot: `<BENCH_ROOT>/assemble/assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- Task handwheel: `<BENCH_ROOT>/assemble/assets-new/object/valve_handwheel_task/blue_valve_handwheel_task.usd`
- Task valve: `<BENCH_ROOT>/assemble/assets-new/object/valve_handwheel_task/three_way_ball_valve_task.usd`

The original files in `<BENCH_ROOT>/assemble/objects` are never
modified. Rebuild the task-specific assets with:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  <ISAACSIM_ROOT>/python.sh \
  <BENCH_ROOT>/assemble/task/g1_valve_handwheel_task/tools/build_valve_handwheel_assets.py
```
