# G1 Roof-Bolt Sleeve Installation Task

This task keeps the existing tracked G1 upper body, official Unitree Quest
3/3S IK, Dex1 grippers, base/lift controls, RealScene04 environment, and the
head/left-wrist/right-wrist cameras. The right gripper picks a coupling sleeve
from the worktable on the robot's right, places it over the fixed upright roof
bolt on the robot's left, then
rotates and lowers it.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_roof_bolt_sleeve_task
./run_task.sh
```

Terminal 2, after the simulator is ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_roof_bolt_sleeve_task
./run_quest3s.sh
```

Open `https://192.168.2.161:8012/?ws=wss://192.168.2.161:8012` in Quest 3/3S.

## Controls

- `W` / `S`: tracked base forward / reverse
- `A` / `D`: tracked base turn left / right
- `T` / `Y`: lift up / down
- `R` in Isaac Sim: reset robot, base/lift, randomized sleeve, and fixed bolt
- Right controller `B`: start teleoperation and recording
- Task 2 uses right-only XR control: after `B`, the right arm and right Dex1
  gripper follow the controller while the left arm and left gripper stay locked.
- Right controller `A`: accept/save the episode and reset all
- Terminal `s`: discard the episode and reset all
- Terminal `q`: discard unfinished data, return arms home, and exit
- Quest triggers: control the two Dex1 grippers

The sleeve starts upright and is randomized independently within `±5 cm` in
world XY. The kinematic roof bolt is independently randomized within `±3 cm`
in world XY while its height and orientation remain fixed; it has 15 cm below
the floor and a top height of 1.30 m. The raised tabletop is 1.09 m high. The
sleeve remains seated on it and has a 53 mm bore, 55 mm outside diameter, and
250 mm length. Its dynamics copy the effective properties of the previous
task's `leiguan.usd`: 0.283436 kg mass, matching principal inertia, gravity,
default damping/velocity limits/contact material, speculative CCD, 16/4 body
solver iterations, 1 m/s depenetration limit, and a 0.5 mm contact envelope. Its
collision is an explicit 16-piece convex ring: 55 mm collision bore, 65 mm
collision OD, and 250 mm
length. Together with the roof bolt's 0.5 mm contact envelope, this preserves
4 mm effective radial insertion clearance. The collision bore is 1 mm larger
per side than the visible bore to make Quest alignment practical. The extra
invisible 5 mm outer radius improves gripper contact while the visible sleeve is ID 53 /
OD 55 / L 250 mm.

The robot, table, sleeve, and roof bolt are shifted together by 0.50 m toward
the robot's left arm to clear the scene furnishing. Override this group offset
at launch with `G1_ROOF_BOLT_GROUP_LEFT_SHIFT_M`, if needed. After that group
placement, the table, sleeve, and roof bolt are shifted together by 0.20 m
toward the robot's right arm while the robot stays fixed. Override this second
offset with `G1_ROOF_BOLT_OBJECT_RIGHT_SHIFT_M`.
The roof bolt then receives a final 0.15 m offset toward the robot, aligning it
with the sleeve in the robot's forward direction. This offset is adjustable
with `G1_ROOF_BOLT_TOWARD_ROBOT_M`.
The table alone receives an additional 0.03 m offset toward the robot's right
arm, adjustable with `G1_ROOF_BOLT_TABLE_RIGHT_SHIFT_M`.

The tracked robot reset pose was captured from the live simulator on
2026-08-21: base XY `(-1.553482, 2.071463)`, yaw `-0.524169 rad`, and lift
height `0.200000 m` (live G1 root Z `1.190468 m`). Isaac Sim `R`, accepted
episodes, and discarded episodes return to this pose.

## Success

Success requires the 45 mm roof bolt centerline to be inside the sleeve's 55 mm
collision bore, relative axis tilt no greater than 12 degrees, and at least
0.20 m insertion depth (80% of the 250 mm sleeve). Once insertion reaches
0.10 m, the simulator terminal prints the live depth, radial-error, and tilt
checks so any failed condition can be identified directly.
Rotation is part of the language
instruction but is not scored. On success, physics, robot commands, and new
camera frames freeze. Right `A` saves; terminal `s` discards and resets.

## Data

The default LeRobot Dataset v3.0 directory is:

```text
<BENCH_ROOT>/assemble/task/g1_roof_bolt_sleeve_task/data/lerobot_v30/roof_bolt_sleeve_installation_replay_ready
```

The recorded instruction is:

```text
Pick up the coupling sleeve with the right gripper, place it over the upright roof bolt, and rotate it while lowering it onto the bolt.
```

Existing datasets are resumed by default. Use `G1_LEROBOT_RESUME=0` with a new
`G1_LEROBOT_DATASET_DIR` to start another collection run.

Each accepted episode writes one replay snapshot under `meta/replay/`. It
contains the initial world XYZ/WXYZ poses for the coupling sleeve and roof
bolt; their poses are not repeated in every recorded frame.

## Assets

- Scene: `<BENCH_ROOT>/assemble/assets-new/scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
- Robot: `<BENCH_ROOT>/assemble/assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- Static table trolley: `<BENCH_ROOT>/assemble/assets-new/object/TableTrolley_B/TableTrolley_B01_01.usd` (top surface aligned to 1.09 m)
- Sleeve: `<BENCH_ROOT>/assemble/assets-new/object/steel_mining_roof_bolt_coupling_sleeve_ID53_OD55_COD65_L250.usd`
- Roof bolt: `<BENCH_ROOT>/assemble/assets-new/object/M20_Bolt_Chain_D45_L1450.usda`

The source sleeve and the previous tabletop task are not modified.
