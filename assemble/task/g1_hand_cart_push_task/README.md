# G1 Hand Cart Push Task

This task reuses the reconstructed `RealScene04_Orin` scene and the tracked
G1 upper-body robot. It adds only the non-articulated two-wheel hand cart; no
task table, prior manipulation objects, or GroundPlane is spawned.

## Start

Terminal 1:

```bash
cd <BENCH_ROOT>/assemble/task/g1_hand_cart_push_task
./run_task.sh
```

Terminal 2, after the simulator reports ready:

```bash
cd <BENCH_ROOT>/assemble/task/g1_hand_cart_push_task
./run_quest3s.sh
```

Open `https://vuer.ai?grid=False` in the Quest browser.

## Controls

- Quest controllers: official G1 dual-arm IK, wrist rotation, and both Dex1 grippers
- Quest left `Y`: drive forward; Quest left `X`: drive backward (release to stop)
- Isaac Sim `W/S`: drive; `A/D`: turn; `T/Y`: lift; `R`: full reset
- Right controller `B`: start teleoperation and a new episode
- Right controller `A`: accept/save and reset all
- Terminal `s`: discard and reset all
- Terminal `q`: discard pending data, return both arms home, and exit

The cart starts with independent XY randomization of `±5 cm` and yaw
randomization of `±3 degrees`. Success is forward progress of at least `0.50 m`
with no more than `0.20 m` lateral drift while upright, and both grippers must
match the user-calibrated dual-handle geometry in
`config/dual_grasp_template.json`. The template stores the two centerline
stations of each gripper relative to the cart, derived from 30 confirmed push
frames. The checker removes motion along the long local-Z handle axis before
measuring cross-handle error, so the gripper may use different points along a
handle. It still constrains lateral-X error, usable longitudinal range and
finger separation. The default cross-handle limits are 7 cm RMS and 8 cm
maximum, with at most 4 cm lateral-X error. Five good frames confirm the grasp;
isolated solver fluctuations decay the confirmation count instead of clearing
it immediately. Success freezes physics,
commands, and new camera frames until the episode is accepted or discarded.
Because the two visual wheels are fused into one collider while the tracked
base is pose-driven, the cart uses synchronized forward motion. While `W` is
held (including the short acceleration/deceleration tail), the cart receives
exactly the same world-XY displacement as the base, so the grasped handles do
not move through the grippers. At other times the cart holds its current pose.

## Physics and assets

- Scene: `<BENCH_ROOT>/assemble/assets-new/scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
- Robot: `<BENCH_ROOT>/assemble/assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- Cart: `<BENCH_ROOT>/assemble/objects/two_wheel_steel_hand_cart.usd`

The source cart remains unchanged. At spawn, the task overrides it to 12 kg,
enables gravity and speculative CCD, uses static/dynamic friction `0.25/0.20`,
and locks roll/pitch. Its authored SDF and visual wheels remain one rigid body,
so it slides rather than rolling with wheel joints. Six invisible analytic
capsules are added along the two scanned handle tubes at runtime. They provide
continuous handle collision without changing the cart's visual appearance.

Useful tuning variables are `G1_HAND_CART_MASS_KG`,
`G1_HAND_CART_STATIC_FRICTION`, `G1_HAND_CART_DYNAMIC_FRICTION`,
`G1_HAND_CART_ROBOT_BACK_OFFSET_M`, `G1_HAND_CART_DISTANCE_M`, and
`G1_HAND_CART_INITIAL_LIFT_M`. `G1_HAND_CART_RANDOM_XY_M` overrides the
default `0.05 m` cart reset range.
`G1_HAND_CART_HANDLE_COLLIDER_RADIUS_M` and
`G1_HAND_CART_HANDLE_CONTACT_OFFSET_M` tune the task-only handle collision.
Grasp-check calibration can be adjusted with
`G1_HAND_CART_GRIPPER_MAX_GAP_M`, `G1_HAND_CART_GRASP_PROBE_RADIUS_M`, and
`G1_HAND_CART_GRASP_CONFIRM_FRAMES`. The simulator prints a full grasp
diagnostic once per second by default; set
`G1_HAND_CART_GRASP_DEBUG_INTERVAL_S=0` to disable it. The same records are
persisted as JSON Lines in `/tmp/g1_hand_cart_push_${UID}_grasp.jsonl` for the
current run, so the measured joint, gap, and center-probe values remain
available after Isaac Sim exits.

## LeRobot Dataset v3.0

The default dataset is:

`data/lerobot_v30/hand_cart_push`

It resumes by default and targets 50 accepted episodes. Each 30 Hz sample has:

- 30-dimensional `observation.state`: 14 arm joints, two grippers, relative
  base forward/lateral position, relative yaw, lift height, left/right actual
  track velocity, lift velocity, and the cart's world XYZ plus WXYZ quaternion
- 19-dimensional `action`: 14 arm targets, two gripper targets, left/right
  track target velocity, and lift target velocity
- head, left-wrist, and right-wrist H.264 RGB video
- instruction: `Grasp both handles of the hand cart and push it straight forward with the mobile base.`

`G1_LEROBOT_DATASET_DIR`, `G1_LEROBOT_TARGET_EPISODES`, and
`G1_LEROBOT_RESUME` override the dataset path, target, and append behavior.
The mobile schema is enabled only by this task's `--lerobot-mobile-base` flag;
the previous tasks remain 16-dimensional for both state and action.
