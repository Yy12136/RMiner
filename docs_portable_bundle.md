# 可迁移 Benchmark 任务包 (portable_Benchmark)

把 `Benchmark/assemble/task` 下所有可运行的仿真任务，连同它们真正依赖的代码、资产和
运行环境配置，打包成这个可复制到别的机器直接使用的目录。

目标机假设：**已装好 NVIDIA 驱动 + CUDA + Isaac Sim 5.1**，只缺任务代码。

---

## 1. 这是什么 / 包含了什么

```
portable_Benchmark/
├── README.md                      # 本文档
├── env.sh                         # 统一的路径/conda 配置（各启动脚本会自动 source）
├── setup/                         # 环境重建与依赖清单
│   ├── unitree_sim_env.yml        # 源机 conda 环境导出（仿真端，精确快照）
│   ├── tv.yml                     # 源机 conda 环境导出（Quest 遥操作端，精确快照）
│   ├── requirements-unitree_sim_env.txt  # pip 依赖清单（已去掉本机绝对路径）
│   ├── requirements-tv.txt               # pip 依赖清单（已去掉本机绝对路径）
│   ├── build_cyclonedds.sh        # 编译 CycloneDDS（两个环境都需要）
│   ├── install_editable.sh        # 把 editable 包重新指向本目录（关键！）
│   └── setup_envs.sh              # 从零重建两个 conda 环境（需联网）
└── assemble/                      # 与源机完全一致的目录层次（不能改名/移动层级）
    ├── task/                      # 任务包 + 共享控制器 + 评估框架
    ├── third_party/               # IsaacLab / unitree_sim_isaaclab / xr_teleoperate / ...
    ├── assets-new/                # 任务场景/机器人/物体 USD
    └── objects/                   # 额外 USD 物体（手推车、阀门、钻杆等）
```

为什么必须保持 `assemble/{third_party,assets-new,task}` 这个层次？
因为 Isaac Lab 的任务配置用 `Path(__file__).resolve().parents[5]` 定位 `assemble/`
根目录（例如 `third_party/unitree_sim_isaaclab/tasks/g1_tasks/<task>/..._env_cfg.py`），
再从这里找 `assets-new/` 和 `task/`。层级一旦改变，资产路径就会失效。

### 已包含的任务

| 任务目录 | 说明 | 仿真入口 | 遥操作入口 | 评估 |
|---|---|---|---|---|
| `g1_tabletop_grasp_task` | Task01 桌面抓取（回归基线） | `./benchmark.sh sim` | `./run_quest3s.sh` | `./run_policy_eval.sh` |
| `g1_roof_bolt_sleeve_task` | Task02 顶板螺栓套筒 | `./benchmark.sh sim` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_bin_pick_place_task` | Task03 料箱分拣 | `./benchmark.sh sim` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_hand_cart_push_task` | Task04 手推车推动 | `./benchmark.sh sim` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_valve_handwheel_task` | Task05 阀门手轮旋转 | `./benchmark.sh sim` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_wall_hole_detonator_task` | Task06 炮孔装药 | `./benchmark.sh sim` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_tabletop_grasp_inspire_task` | Inspire 手版桌面抓取 | `./run_task.sh` | `./run_quest3s.sh` | — |
| `g1_hand_cart_rolling_demo_task` | 手推车滚动 demo | `./run_rolling_demo.sh` | `./run_quest3s_rolling_demo.sh` | — |
| `w1_mobile_drilling_task` | W1 履带底盘+G1 打钻（主线） | `./run_task.sh` | `./run_quest3s.sh` | — |
| `quest_dex3_hand_test` | G1 + Dex3 手套测试 | `./run_sim_hand_test.sh` | — | — |

其它目录（`galbot_tracked_teleop`、`astribot_tracked_teleop`、`valve_dynamic_validation_demo_*` 等）
也一并保留，属实验/校验脚本。

---

## 2. 没有包含的东西（按“无用数据先算了”原则剔除）

以下内容体积巨大且可用原流程重新生成/下载，未打包：

- 每个任务的 `experiments/`（策略权重 + 评估结果，单项就 90–138G）
- `data/`（LeRobot 采集数据集，1.9–11G/任务）
- 渲染/录像产物：`*.mp4`、`*_frames/`、`artifacts/`、`outputs/`、`camera_demo_*`、
  `robot_exports/`、`replay_augmented/`、`head_camera_first3*`、`all_task_asset_renders/`
- `archive/`、各类 `.backup_*`/`.bak_*` 备份
- `policy/lerobot`（14G 的 LeRobot + uv venv，仅评估/训练用）
- `scripts/.x7s_ik_env`（1.2G 冻结环境）和 `scripts/.quest_deps`（可用 `setup_w1_quest3s.sh` 重建）
- `simulator/`（另一份未被任何任务调用的 IsaacLab 3.0，死代码）

需要用评估/训练时，参考第 4 节补齐。

---

## 3. 在目标机上使用

### 3.1 拷贝

把整个 `portable_Benchmark/` 拷到目标机任意路径（建议放在数据盘，例如
`/data/portable_Benchmark`）。**保持目录内部结构不变**，路径随便换。

```bash
rsync -a --info=progress2 portable_Benchmark/ user@newhost:/data/portable_Benchmark/
```

### 3.2 准备 conda 环境（二选一）

**情况 A：目标机已有可用的 `unitree_sim_env` 和 `tv`（推荐）**

只需把 editable 包重新指向本 bundle 即可（否则它们还指着旧机器的绝对路径）：

```bash
cd /data/portable_Benchmark
./setup/build_cyclonedds.sh                 # 生成 CYCLONEDDS_HOME
source /path/to/miniconda3/etc/profile.d/conda.sh

conda activate unitree_sim_env && ./setup/install_editable.sh unitree_sim_env
conda activate tv              && ./setup/install_editable.sh tv
```

**情况 B：目标机没有这两个环境，需要从零建**

```bash
cd /data/portable_Benchmark
./setup/setup_envs.sh            # 需要联网（PyTorch / NVIDIA / conda 源）
```

> `setup/*.yml` 是源机环境的**精确快照**，可直接参考版本号；
> 但 `conda env create -f unitree_sim_env.yml` 会因为 isaacsim 需要
> `--extra-index-url https://pypi.nvidia.com` 而失败，所以清理/重建请用
> `setup_envs.sh`。

### 3.3 运行任务

每个任务都在**自己的目录里**运行，启动脚本会自动定位 conda 和任务资产：

```bash
cd /data/portable_Benchmark/assemble/task/g1_valve_handwheel_task
./run_task.sh                 # 终端 1：启动 Isaac Sim（等窗口出现）
./run_quest3s.sh              # 终端 2：Quest 3/3S 遥操作
```

纯键盘版（无需 Quest），例如：

```bash
cd /data/portable_Benchmark/assemble/task/w1_mobile_drilling_task
./run_task.sh --headless      # 无头运行；或直接 ./run_task.sh 可视化
```

评估（需要 LeRobot venv 与策略权重，见第 4 节）：

```bash
cd ../g1_valve_handwheel_task
./benchmark.sh eval --checkpoint /path/to/pretrained_model --episodes 5 --seed 42
```

---

## 4. 可选：评估 / 训练相关依赖（默认未带）

若需要跑 `benchmark.sh eval` 或训练，额外准备：

1. **LeRobot + venv**（评估器必须用它启动）：
   ```bash
   git clone https://github.com/huggingface/lerobot.git /data/portable_Benchmark/policy/lerobot
   cd /data/portable_Benchmark/policy/lerobot && uv sync    # 或按 README 建 .venv
   ```
2. **策略权重 / 数据集**：从原机器 `experiments/weights/...`、`data/lerobot_v30/...`
   拷回对应任务的同名子目录。
3. 覆盖环境变量（见下）指向你的实际路径。

---

## 5. 环境变量开关（都可在调用前覆盖）

| 变量 | 作用 | 默认 |
|---|---|---|
| `CONDA_SH` | conda 初始化脚本路径 | 自动探测（PATH 上的 conda 或常见安装目录） |
| `LEROBOT_ROOT` | LeRobot 仓库根 | `<bundle>/policy/lerobot` |
| `POLICY_PYTHON` | 评估/训练用 python | `$LEROBOT_ROOT/.venv/bin/python` |
| `ISAACSIM_ROOT` | 独立 Isaac Sim 安装（仅个别工具用） | `$HOME/isaacsim` |
| `ISAACSIM_ASSET_ROOT` | Isaac Sim 内置资产库（仅个别构建工具用） | `<bundle>/assets/isaacsim_assets` |
| `G1_IMAGE_SERVER_IP` | Quest 图像服务 IP | `192.168.2.161` |
| `G1_DRILLING_DEVICE` / `G1_EVAL_SIM_DEVICE` | 仿真设备 | `cuda` |
| `G1_SHOW_CAMERA_VIEWPORTS` | 是否弹三路相机窗口 | `1`（设 `0` 关闭） |
| `G1_LEROBOT_DATASET_DIR` | 遥操作录制数据集目录 | 各任务 `data/lerobot_v30/...`（本包未带，需自建或覆盖） |

`env.sh` 会把 `CONDA_SH`、`LEROBOT_ROOT`、`POLICY_PYTHON`、`ISAACSIM_ASSET_ROOT`
导出，启动脚本会自动 source 它。

---

## 6. 已知仍需手动处理的点

1. **`scripts/.quest_deps` 未带**：W1/X7S 的 Quest 控制（`w1_quest3s*.py` 路径）会在
   首次运行时自动调用 `assemble/task/scripts/setup_w1_quest3s.sh` 安装 pin 版本的
   Vuer / RealMan 依赖；如目标机离线，请手动预装。
2. **`augmentation/`、各 `tools/`、`demo_mining_conditions/` 等创作类脚本**：属高级
   工具，部分仍可能引用源机绝对路径。已尽量改为 `BENCH_ASSEMBLE_ROOT` /
   `ISAACSIM_ROOT` 环境变量回退，必要时设置这些变量或手工核对。
3. **LeRobot 数据集默认目录**：遥操作录制脚本默认写到各任务 `data/lerobot_v30/...`，
   该目录未打包。首次录制前请确认目录存在或设置 `G1_LEROBOT_DATASET_DIR`。
4. **端口/锁**：仿真端独占 TCP `60000`（图像服务）并使用 `/tmp/g1_*_${UID}.lock`
   加锁；同时只跑一个任务实例。

---

## 7. 源机使用环境（供核对）

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 22.04 |
| conda base | `<miniconda3>` |
| `unitree_sim_env` | Python 3.11.15，torch 2.7.0+cu128，isaacsim 5.1.0.0 |
| `tv` | Python 3.10.20，torch 2.3.0，pinocchio 3.1.0，vuer 0.0.60 |
| Isaac Lab | `third_party/IsaacLab` @ `80094be32` (2.3.2) |
| unitree_sim_isaaclab | @ `e30c25b` |
| xr_teleoperate | @ `845b25a` |

精确包清单见 `setup/unitree_sim_env.yml`、`setup/tv.yml`。
