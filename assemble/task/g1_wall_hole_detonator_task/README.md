# G1 Wall-Hole Loaded Detonator Task

This task places the tracked G1 in front of the authored hole in
`realscene04_hole.usd`. A pre-loaded detonator assembly rests horizontally on
the worktable beside the right gripper. The right arm grasps the assembly and
inserts the 262 mm detonator body at least 200 mm into the fixed wall hole.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_wall_hole_detonator_task
./run_task.sh
```

Terminal 2, after the Isaac Sim window is ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_wall_hole_detonator_task
./run_quest3s.sh
```

Open <https://vuer.ai?grid=False> in Quest 3/3S. Right controller `B` starts
right-arm teleoperation and a new LeRobot v3 episode.

## Task and controls

- The left arm and left Dex1 gripper remain locked.
- Right `B`: start teleoperation and recording.
- Right `A`: accept/save the episode, then perform a full reset.
- Terminal `s`: discard the episode and perform a full reset.
- Terminal `q`: discard any unfinished episode, return the arms home, and exit.
- Isaac `W/S`, `A/D`: drive/turn the tracked base.
- Isaac `T/Y`: raise/lower the lift.
- Isaac `R`: reset robot, base/lift, loaded detonator, and all lead links.

The default instruction is:

```text
Pick up the loaded detonator assembly with the right gripper and insert it into the wall hole.
```

Accepted data is stored by default in:

```text
data/lerobot_v30/wall_hole_loaded_detonator_insertion
```

## Scene and object layout

- Scene: `assets-new/scene/RealScene04/real2sim_intermediate/realscene04_hole.usd`
- Robot: tracked G1 29-DoF with Dex1.
- Table: `TableTrolley_B01_01.usd`, collision enabled, top at 1.09 m.
- Loaded object: `assets-new/object/loaded_detonator_assembly.usd`.
- Wall target: the fixed kinematic hollow cylinder already authored in the
  scene.  Its outer size and position are preserved; at spawn, only its bore
  is enlarged to approximately 67 mm for reliable teleoperation insertion.
- Ground: invisible collision-only GroundPlane authored in the scene.

The task reads the authored hole center and axis at startup. The default base
pose was captured from the calibrated running task at XYZ-yaw
`(-3.0386274361, 3.2230474961, 0, -0.6097617192)`. The table is 0.85 m outward
and 0.65 m to the robot's right. Its XY
dimensions are 50% of the Task 2 trolley (about 0.37 x 0.61 m). The loaded
detonator is placed near the right gripper. The complete table/object layout is
rotated 90 degrees clockwise around the table center, then shifted 0.20 m
toward the robot's left arm. The flexible lead follows the trolley's rotated
long edge. The table alone is then moved 0.05 m robot-right and 0.10 m forward
toward the wall; this final table-only adjustment does not move the detonator.

Every object/full reset applies a common random XY offset in `[-5 cm, +5 cm]`
to the loaded body and all 26 lead links. Candidates outside the tabletop or
too close to either gripper are rejected. The hole, table, and robot placement
do not randomize.

The loaded detonator's base reset position is shifted `15 cm` along the wall's
outward axis (the previous `5 cm` plus the requested additional `10 cm`). This
backoff moves only the loaded detonator; the table remains at its existing
position. The flexible lead may consequently overhang the outward table edge
by up to `17 cm`; the rigid detonator body must remain on the tabletop.

## Loaded assembly physics

The 262 mm shell is 10 mm longer than the internal 252 mm connector. The shell
and inserted connector are one 0.10 kg dynamic rigid body, so the
pre-loaded state cannot separate. The existing 26-link lead remains flexible
and is connected to the compound body by a fixed joint. The connector/lead
insertion end protrudes 1 mm past the rear shell opening so the complete lead
is visible. The asset uses
speculative CCD, 16/4 solver iterations, and 0.5 mm contact offsets.

Rebuild and statically validate the asset with:

```bash
python \
  tools/build_loaded_detonator_assembly.py
```

## Success

Success requires only these two conditions:

- axial overlap inside the wall bore is at least five eighths of the detonator
  length (`0.16375 m`);
- centerline radial error is strictly less than one detonator diameter
  (`0.045 m`).

On success, the simulator writes
`/tmp/g1_wall_hole_detonator_${UID}.success`, freezes physics/commands/new
camera frames, and keeps DDS alive so `A` can save or `s` can discard/reset.
At the start of every recorded episode, LeRobot v3 stores replay metadata with
the exact loaded-detonator XYZ/WXYZ pose for that reset, together with the
captured tracked-base pose and lift height.
The same current XYZ/WXYZ object pose is appended to every recorded
`observation.state` frame, increasing this task's state from 16 to 23 values;
the action remains 16 values.

## Tunable environment variables

- `G1_WALL_HOLE_BASE_X_M` (default `-3.0386274361`)
- `G1_WALL_HOLE_BASE_Y_M` (default `3.2230474961`)
- `G1_WALL_HOLE_BASE_YAW_RAD` (default `-0.6097617192`)
- `G1_WALL_HOLE_ROBOT_FORWARD_SHIFT_M` (default `0.10`; moves the initial
  tracked base 10 cm forward toward the wall)
- `G1_WALL_HOLE_TABLE_OUTWARD_M` (default `0.85`)
- `G1_WALL_HOLE_TABLE_RIGHT_M` (default `0.65`)
- `G1_WALL_HOLE_TABLE_OBJECT_LEFT_SHIFT_M` (default `0.20`)
- `G1_WALL_HOLE_TABLE_ONLY_RIGHT_SHIFT_M` (default `0.05`)
- `G1_WALL_HOLE_TABLE_ONLY_FORWARD_SHIFT_M` (default `0.10`)
- `G1_WALL_HOLE_TABLE_XY_SCALE` (default `0.005`)
- `G1_WALL_HOLE_OBJECT_OUTWARD_M` (default `0.85`)
- `G1_WALL_HOLE_OBJECT_RIGHT_M` (default `0.40`)
- `G1_WALL_HOLE_OBJECT_WALL_BACKOFF_M` (default `0.15`)
- `G1_WALL_HOLE_OBJECT_RANDOM_XY_M` (default `0.05`)
- `G1_WALL_HOLE_LIFT_HEIGHT_M` (default `0.058038`, captured from the calibrated
  running pose whose robot-root Z was `1.048506 m`)
- `G1_LEROBOT_DATASET_DIR`, `G1_LEROBOT_TARGET_EPISODES`,
  `G1_LEROBOT_RESUME`, and `G1_TASK_INSTRUCTION` retain the Task 2 behavior.
