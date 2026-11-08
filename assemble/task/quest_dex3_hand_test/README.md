# Quest + G1/Dex3-1 灵巧手仿真测试

这是独立于现有任务的测试环境：Unitree G1（29 DoF）使用 Dex3-1 三指灵巧手，Quest 3/3S 使用自身 Hand Tracking，不依赖手柄控制器。

## 安装

```bash
cd <BENCH_ROOT>/assemble/task/quest_dex3_hand_test
./setup_env.sh
```

默认环境名为 `xr_dex3`，可用 `XR_DEX_ENV=自定义名称` 覆盖。

## 启动

先确保 Isaac Sim 已安装，并设置其目录（目录内应有 `python.sh`）：

```bash
export ISAAC_SIM_ROOT=<ISAACSIM_ROOT>
./run_sim_hand_test.sh
```

在 Quest 3/3S 浏览器中打开终端打印的 HTTPS/Vuer 地址，接受证书并启用 Hand Tracking。先在仿真中确认手臂跟随，再做捏合、握拳、张开等动作，观察 Dex3-1 三指关节是否同步。

## 选择理由与限制

官方 `unitreerobotics/xr_teleoperate` 已提供 `G1_29 + dex3 + --input-mode=hand + --sim` 路径，这是当前最少改动、最接近真实 G1 的方案。真实机器人还需要 Dex3-1 硬件、DDS/SDK 网络和安全限位；本目录先只做仿真验证。Quest 的 Hand Tracking 输出的是 26 个手部关节，需由 retargeting 映射到 Dex3-1 的 7 DoF，不能把每根手指直接当成同名关节。

## 已下载的备用五指手资产

`assets/brainco-description` 是 BrainCo 官方开源 Revo2/Revo3 资产，含左右手的 USD、URDF 和 MJCF。当前 Unitree XR 遥操已支持 `--ee=brainco`，但现有 `unitree_sim_isaaclab` 尚未提供它的 G1 场景和 DDS 桥接；在它可被 Quest 测试前，仍需把 Revo2 双手挂接到 G1 手腕并实现 `rt/brainco/*` DDS 适配。

为了先验证 Quest 手势到 BrainCo Revo2 的重定向，本目录提供 `brainco_inspire_dds_bridge.py`：它将 BrainCo 的左右六关节 DDS 命令桥接到现有 G1-Inspire 仿真场景。运行时令 `XR_EE=brainco`，并启动 `Isaac-PickPlace-Cylinder-G129-Inspire-Joint --enable_inspire_dds`。该桥接只验证控制链路；画面中仍是 Inspire 五指手，原生 Revo2 USD 挂接是后续独立阶段。
