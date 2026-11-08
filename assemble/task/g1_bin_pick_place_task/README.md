# G1 Randomized-Object/Bin Pick-and-Place Task

This task uses the RealScene04 environment, the tracked G1 upper body, the
official Unitree Quest 3/3S arm IK controller, two Dex1 grippers, the tracked
base and lift controls, and the head/left-wrist/right-wrist cameras. The left
arm is held in a side raise while the right arm performs the task. One of
five SAM3D metal components starts on the right side of the trolley and must
be placed fully inside the randomly selected sorting bin on the left.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_bin_pick_place_task
./run_task.sh
```

Terminal 2, after the simulator reports that its controller is ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_bin_pick_place_task
./run_quest3s.sh
```

Open <https://vuer.ai?grid=False> in Quest 3/3S. No additional page is needed
in the headset. `run_task.sh` opens one desktop window containing all three
camera streams.

## Controls

- `W` / `S`: tracked base forward / reverse
- `A` / `D`: tracked base turn left / right
- `T` / `Y`: lift up / down
- `R` in the active Isaac Sim window: reset robot, both arms and grippers,
  base/lift, and the randomized metal component
- Right controller `B`: enable only the right arm and right gripper, then start
  recording a new episode; the left arm and gripper remain locked
- Right controller `A`: accept/save the current episode and reset everything
- Terminal `s`: discard the current episode and reset everything
- Terminal `q`: discard unfinished data, return both arms home, and exit
- Quest triggers: close/open the corresponding Dex1 gripper

Every reset selects one of five metal components with equal probability. The
component and bin are randomized independently by `±5 cm` left/right and by
`0–5 cm` only toward the robot; neither can move farther away from the robot.
Their nominal reset centers are both shifted 4 cm toward the robot's right arm,
and all randomized poses are sampled around these shifted centers.
The component also receives a random `-180°..+180°` yaw while remaining flat,
so reset never introduces roll or pitch. Its Z position is computed from the
selected component's height. The other four
components are hidden and physically isolated on a dedicated parking shelf
outside the workspace. Every reset also selects one of three static open bins with
equal probability; the other two are hidden and their collisions are disabled.
Independently, the selected bin is tinted
industrial blue, safety yellow, warning orange, graphite gray, or equipment
green with equal probability. Its albedo texture is desaturated before tinting,
so surface detail, normal maps, roughness, and metal parts remain unchanged.
The task-only compound colliders do not alter the source USD files. The table
does not move. The robot reset pose is base XY `(-1.762346, 2.192210)`, yaw
`-0.524169 rad`, and lift height `0.20 m`.

The left-arm task-home joint pose is
`[0.15, 0.675, 1.50, 0, 0, 0, 0]` radians (shoulder pitch/roll/yaw, elbow,
wrist roll/pitch/yaw), which places it in an approximately horizontal side
raise. This pose is restored after reset/save/discard/quit and remains locked
when right-controller `B` starts teleoperation.

## Layout and physics

The trolley is centered about 0.509 m directly in front of the G1 mount and its top
surface remains at 1.09 m. Their approximate task sizes are
`0.261 × 0.325 × 0.066 m` for the original blue bin,
`0.300 × 0.200 × 0.075 m` for Tote B08, and `0.405 × 0.279 × 0.183 m` for
Container F26. Each bin keeps its original render mesh but uses a robust
five-box compound collider (one floor and four walls) under an independent,
unscaled kinematic collision root, leaving the top and calibrated interior
open. Startup verifies all five named colliders and the exact floor height for
all three bins before control is enabled. Each selected metal component is a 0.25 kg
dynamic body with native PhysX gravity, high friction, speculative CCD, 16/4
solver iterations, linear/angular damping of `0.8/1.5`, and one convex-hull
collider. No GroundPlane is added.

The randomized metal component starts 0.10 m closer to the robot than in the
original layout. Its reset clearance above the tabletop is 0.5 mm. Object and
bin contact offsets are 1 mm; the thick five-box bin collision remains robust
without the earlier visible hover gap.

## Success

Success requires the active component's center to be inside the selected
bin's calibrated opening, the lowest point of its actual convex-hull collision
mesh to contact the physical floor (no more than 3 mm penetration or 6 mm
separation), and its center to be below the rim. Requiring every long-component
corner to fit was
removed because it rejected valid diagonal placements. Linear speed must be
below 0.08 m/s and angular speed below 0.5 rad/s continuously for 0.5 seconds.
Success prints `SUCCESS` and freezes physics,
commands, and new camera frames. Right `A` still saves; terminal `s` still
discards and resets.

## LeRobot Dataset v3.0

The default dataset is:

```text
<BENCH_ROOT>/assemble/task/g1_bin_pick_place_task/data/lerobot_v30/sam3d_metal_component_to_randomized_bin
```

The recorded instruction is:

```text
Pick up the metal component from the right side of the table and place it into the sorting bin on the left.
```

The target is 50 accepted episodes and existing data is resumed by default.
Set `G1_LEROBOT_DATASET_DIR` to select another dataset. Use a new empty
directory together with `G1_LEROBOT_RESUME=0` for a fresh collection run.

### Replay snapshots

Pressing right-controller `B` first freezes a fresh DDS simulator snapshot and
only starts the episode if that snapshot belongs to this task and is no more
than 0.5 seconds old. Each accepted episode transactionally stores its snapshot
at:

```text
meta/replay/chunk-XXX/file-XXX.json
```

The snapshot contains the complete Isaac scene `init_state`, robot base/lift
and G1 root pose, selected object type/asset/exact pose/velocity, selected bin
type/color/exact pose, all reset offsets, the object's horizontal random yaw,
and the object/bin catalogs. Replay coverage is discovered directly from the
sidecars, so standard LeRobot v3 metadata remains unchanged and official
loaders can ignore this optional extension cleanly. Saving commits the sidecar
together with data, videos and episode metadata; discarding creates no sidecar.
`LeRobotV30Writer.load_replay_metadata(episode_index)` validates and loads the
immutable snapshot for a replay tool.

## Assets

- Scene: `<BENCH_ROOT>/assemble/assets-new/scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
- Robot: `<BENCH_ROOT>/assemble/assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- Static trolley: `<BENCH_ROOT>/assemble/assets-new/object/TableTrolley_B/TableTrolley_B01_01.usd`
- Blue bin: `<BENCH_ROOT>/assemble/assets-new/object/BlueSortingBin/blue_sorting_bin.usd`
- Tote B08: `<BENCH_ROOT>/assemble/assets-new/object/Tote_B08/Tote_B08_60x40x15cm_PR_V_NVD_01.usd`
- Container F26: `<BENCH_ROOT>/assemble/assets-new/object/Container_F26/Container_F26_116x80x52cm_PR_V_NVD_01.usd`
- Metal nozzle: `<BENCH_ROOT>/assemble/assets-new/object/SAM3DMetalNozzle/SAM3D_Metal_Nozzle_GraspReady.usd`
- Four randomized metal components: `<BENCH_ROOT>/assemble/assets-new/object/SAM3DRandomMetalComponents`

The previous nozzle-only dataset and the existing roof-bolt sleeve task/data
are not modified. Five object shapes, three bin shapes, and five bin colors
provide 75 independent appearance combinations.
