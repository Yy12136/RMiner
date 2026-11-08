# Tracked G1 Upper-Body Mobile Drilling Scene

The default task keeps the original W1 tracked chassis, lift, support geometry,
collision proxies, saved base pose, and upper-body mounting frame. The W1 upper
body is removed and replaced by the Unitree G1 pelvis/waist/torso, dual arms,
head, cameras, and Dex1 grippers. G1 leg visuals and collisions are disabled,
while their joint coordinates remain internal so the official 29-DOF DDS and
Quest arm IK protocol stays unchanged.

## 1. Start the drilling simulation

```bash
cd <BENCH_ROOT>/assemble/task/w1_mobile_drilling_task
./run_task.sh
```

Wait for Isaac Sim to open, then click the simulator's **Sim** button if the
terminal asks for it. A separate three-panel window automatically displays the
head, left-wrist, and right-wrist camera streams. Leave this terminal and Isaac
Sim running.

## 2. Start the unchanged official Quest teleoperation

In a second terminal, use the task wrapper:

```bash
cd <BENCH_ROOT>/assemble/task/w1_mobile_drilling_task
./run_quest3s.sh
```

Open the printed `https://vuer.ai?grid=False` page on Quest 3/3S and enter VR.
Starting the program only connects XR, cameras, and DDS. It does not begin arm
tracking automatically. After entering VR, press `r` once in the
`run_quest3s.sh` terminal to begin teleoperation. The controls are unchanged:

- Move the left/right controller: move the corresponding arm
- Left/right index trigger: close/open the corresponding Dex1 gripper
- Terminal `r`: enable/start teleoperation
- Terminal `q`: stop teleoperation and return the arms home

The tracked-base controls from the earlier W1 task are also retained in the
Isaac Sim window:

- `W/S`: tracked base forward/reverse
- `A/D`: tracked base differential turn
- `T/Y`: raise/lower the lift and mounted G1 upper body
- `R`: reset the tracked base and lift to the saved W1 placement

The G1 root follows the moving lift kinematically, matching the original W1
mount behavior, while its arm joints and Dex1 grippers remain under the
official Unitree controller.

## Preserved launchers

- `./run_task_x7s.sh`: previous X7S mobile-manipulator implementation
- `./run_task_w1.sh`: earlier W1 implementation

## Assets and implementation

- Combined scene: `assets/environment/RealScene04/real2sim_intermediate/RealScene04_Orin.usd`
  (contains the NuRec visual and invisible collision layers)
- No additional fallback GroundPlane is spawned; scene contact uses the
  collision layer authored inside `RealScene04_Orin.usd`.
- Task object: `assets/objects/gasoline_powered_post_driver.usd`
- Final combined robot asset: `../../assets-new/robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd`
- IsaacLab task: `../../third_party/unitree_sim_isaaclab/tasks/g1_tasks/mobile_drilling_g1_29dof_dex1`

Optional simulator arguments may be appended to `./run_task.sh`; for example,
`./run_task.sh --headless`. Set `G1_DRILLING_DEVICE=cpu` only when CUDA is not
available. Set `G1_SHOW_CAMERA_VIEWPORTS=0` to disable the three-camera window.
