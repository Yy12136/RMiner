# Quest 3/3S 遥操作：Galbot 与 Astribot S1 履带底盘本体

本目录为两条履带底盘上装本体提供与宇树 G1 一致的 Quest 遥操作：

- `galbot_tracked_teleop/` — Tracked Galbot One Golf（银河通用）
- `astribot_tracked_teleop/` — Astribot S1 上身 + Wuji Hand 2（星尘智能）

方案与仓库内已验证的 W1/X7S 遥操作一致：Isaac Sim 直接加载组装 USD，
`w1_quest3s.py` 提供 Vuer WebXR Quest 桥，`tracked_upperbody_ik.py`
从组装 USD 的关节坐标系构建 Pinocchio/Pink 约束 QP IK（7 自由度/臂）。

## 官方遥操作调研结论

- **Astribot S1**：官方仓库 `Astribot-Dev/astribot_descriptions`（URDF/MJCF/USD）
  与 `astribot_simulation`（MuJoCo/Genesis/ManiSkill/IsaacLab）。未发现公开的
  Quest 遥操作程序；官网所述“VR 遥操作”为商业版 Astribot UI，不公开 IK 代码。
- **Galbot**：官方 `GalbotSDK`（真机 SDK）与 `galbot_one_golf_description`
  （URDF/MJCF/USD）。TM01 为物理主从臂，非 Quest；未发现公开 Quest 遥操作。
- **Lw-benchhub**：支持 `G1-Controller`、`X7S-Abs`、PandaOmron 等 `vr-controller`
  设备（OpenTeleVision），不含 Galbot 或 Astribot。

因此两条机器人没有现成官方 Quest 遥操作程序，采用仓库既有的
“QuestBridge + Pinocchio/Pink IK”参考宇树方案。

## 启动

先启动仿真遥操作（本脚本同时是仿真与遥操作入口）：

```bash
# Galbot
cd <BENCH_ROOT>/assemble/task/galbot_tracked_teleop
./run_quest.sh

# Astribot S1 tracked
cd <BENCH_ROOT>/assemble/task/astribot_tracked_teleop
./run_quest.sh
```

启动后终端会打印证书地址与 Vuer 地址：

1. 在 Quest 3/3S 浏览器打开 `https://<本机IP>:8012`，接受自签名证书；
2. 打开 `https://vuer.ai?ws=wss://<本机IP>:8012`，进入 VR；
3. 手柄握持 Grip 驱动对应手臂，Trigger 控制夹爪/Wuji 手抓握。

## 手柄操作

| 操作 | 功能 |
| --- | --- |
| 按住左 Grip | 左臂跟随左手柄（相对 IK，先锚定到当前位姿） |
| 按住右 Grip | 右臂跟随右手柄 |
| 左/右 Trigger | 关闭对应夹爪（Galbot）或卷曲 Wuji 手（Astribot） |
| 右 B | 双臂复位到初始位姿 |
| ESC | 退出 |

IK 为“夹持离合”式相对控制：第一次按住 Grip 时把控制器位姿锚定到当前
末端，之后控制器相对位移按 `TRACKED_QUEST_POSITION_SCALE`（默认 0.7）缩放
映射到末端，并用 Pink 约束 QP 求解 7 自由度臂。

## 无手柄测试（烟测 / headless）

```bash
# headless 烟测（不加载 Quest，验证装配、驱动、IK、关节跟踪）
cd <BENCH_ROOT>/assemble/task/galbot_tracked_teleop
TRACKED_SMOKE_TEST=1 TRACKED_SMOKE_FRAMES=60 TRACKED_QUEST_ENABLED=0 ./run_quest.sh

cd <BENCH_ROOT>/assemble/task/astribot_tracked_teleop
TRACKED_SMOKE_TEST=1 TRACKED_SMOKE_FRAMES=60 TRACKED_QUEST_ENABLED=0 ./run_quest.sh
```

无 Quest 时可用键盘驱动：
- `W/S A/D`、`I/K J/L`：左右臂关节
- `T/Y U/O`：左右夹爪/手
- `R`：复位，`ESC`：退出

## 关键文件

- `assemble/task/scripts/tracked_upperbody_ik.py` — 通用 USD→Pinocchio/Pink IK
- `assemble/task/scripts/tracked_upperbody_teleop.py` — 通用 Quest 遥操作控制器
- `assemble/task/galbot_tracked_teleop/config.json` — Galbot 装配/关节/驱动配置
- `assemble/task/astribot_tracked_teleop/config.json` — Astribot 装配/关节/驱动配置
- `assemble/task/scripts/w1_quest3s.py` / `w1_quest3s_server.py` — Vuer WebXR 桥

## 说明与限制

- 两条机器人共用 Tracked G1 履带底盘，装夹关节固定、不参与遥操作；驱动
  配置在 `_configure_drives` 统一按每弧度（N·m/rad）覆写，避免 Galbot 的
  每度/每弧度混用导致跟踪抖动。
- 履带与上装全部 `disableGravity`（与 W1 一致），否则整体会坠落。
- Astribot 的 Wuji 手共 20 DOF/只，Trigger 触发协调卷曲（flex 关节）+ 收拢
  （abd 关节归零）；如需手势级单指控制可在此基础上扩展。