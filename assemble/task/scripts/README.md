# Shared W1 task scripts

`w1_mobile_manipulator_teleop.py` is the reusable controller for config-driven
W1 mobile-manipulation tasks. Keyboard input controls the mobile base/lift;
`w1_quest3s.py` and `w1_quest3s_server.py` provide reusable Quest 3S WebXR
control for arms, grippers, and head. A task package sets `W1_TASK_ROOT` and
provides:

- `config/task.json` for asset paths and task metadata;
- `config/placement.json` for persistent placement;
- the robot, environment, and task-object assets named by the task config.

The current drilling task is:

```text
../w1_mobile_drilling_task/
```

Recommended launch:

```bash
cd <BENCH_ROOT>/assemble/task/w1_mobile_drilling_task
./run_task.sh
```

The old command remains compatible:

```bash
cd <BENCH_ROOT>/assemble/task/scripts
./run_task001.sh
```

Keyboard-only control of the same complete drilling task is available through:

```bash
./run_task002.sh
```

Task002 disables Quest, enables keyboard control for both arms, grippers, head,
and waist pitch, and defaults to normal PD joint targets so physics contacts can
oppose arm motion. Set `W1_DRILLING_STRICT_JOINT_HOLD=1` explicitly only when
deterministic joint holding is more important than contact response.

`task001_teleop.py` is now only a compatibility wrapper. New tasks should reuse
`w1_mobile_manipulator_teleop.py` through their own small `run_task.sh`.

`setup_w1_quest3s.sh` installs the pinned Vuer and official RealMan
`Robotic_Arm` dependencies into shared `.quest_deps`. Task entrypoints invoke
it automatically when Quest support is enabled and either dependency is absent.
Quest arm control defaults to a Pinocchio/Pink constrained-QP backend whose
two exact 7-DOF models are constructed directly from the composed W1 USD joint
frames. The RealMan RM75 and previous Jacobian DLS implementations remain
available as `W1_QUEST_IK_BACKEND=realman` and `W1_QUEST_IK_BACKEND=dls`.
