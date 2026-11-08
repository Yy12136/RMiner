# RMiner

**RMiner** 是一个面向矿业 / 施工场景的机器人操作 Benchmark。任务在
**NVIDIA Isaac Sim 5.1** 中构建，使用 **Unitree G1（29 DoF + Dex1 夹爪）**
履带式移动底盘，通过 **Meta Quest 3 / 3S** 遥操作采集示教数据，并提供
LeRobot 格式的数据集、策略评估框架和完整的可迁移运行环境。

> 本仓库只包含**代码、任务定义与环境配置**。
> 3D 资产（USD/USDZ/贴图等，约 8 GB）和全部采集数据集（约 24 GB）托管在
> Hugging Face：[`lyang116/RMiner`](https://huggingface.co/datasets/lyang116/RMiner)。

---

## 目录结构

```
RMiner/
├── README.md                     # 本文件
├── setup/                        # conda 环境快照、依赖清单、重建脚本
│   ├── setup_envs.sh             # 从零重建 unitree_sim_env + tv
│   ├── install_editable.sh       # 把 editable 包指向本仓库
│   ├── build_cyclonedds.sh       # 编译 CycloneDDS
│   ├── unitree_sim_env.yml       # 仿真端精确环境快照
│   ├── tv.yml                    # Quest 遥操作端精确环境快照
│   └── requirements-*.txt        # pip 依赖
├── scripts/                      # 资产/数据集下载与安装检查
│   ├── download_assets.sh
│   ├── download_datasets.sh
│   └── check_install.sh
├── assemble/                     # 与 Isaac Lab 任务配置约定的固定层级
│   ├── env.sh                    # 统一路径 / conda 配置
│   ├── task/                     # 任务包、共享控制器、评估框架
│   ├── third_party/              # IsaacLab / unitree_sim_isaaclab / xr_teleoperate / CycloneDDS
│   ├── assets-new/               # （下载）共享场景/机器人/物体 USD
│   └── objects/                  # （下载）额外 USD 物体
├── datasets/                     # （下载）LeRobot 数据集
└── docs_portable_bundle.md       # 原始可迁移包说明（环境版本、排错）
```

`assemble/{third_party,assets-new,task}` 的层级**不能改变**：Isaac Lab 的任务
配置通过 `Path(__file__).resolve().parents[5]` 定位 `assemble/` 根目录，再从这里
引用 `assets-new/`、`objects/` 和 `task/`。

---

## 快速开始

### 0. 前置条件

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA，驱动支持 CUDA 12.8 |
| Isaac Sim | 5.1.0 |
| Python（仿真） | 3.11（`unitree_sim_env`） |
| Python（遥操作） | 3.10（`tv`） |

### 1. 克隆代码

```bash
git clone https://github.com/Yy12136/RMiner.git
cd RMiner
```

### 2. 下载 3D 资产

```bash
# 需要 huggingface_hub（pip install -U huggingface_hub）
./scripts/download_assets.sh
```

脚本会把资产放到 `assemble/assets-new/`、`assemble/objects/`、
`assemble/task/*/assets/` 和 `assemble/third_party/unitree_sim_isaaclab/assets/`。

### 3. 准备 conda 环境（二选一）

**情况 A：机器上已有可用的 `unitree_sim_env` 和 `tv`（推荐）**

```bash
./setup/build_cyclonedds.sh
conda activate unitree_sim_env && ./setup/install_editable.sh unitree_sim_env
conda activate tv              && ./setup/install_editable.sh tv
```

**情况 B：从零重建**

```bash
./setup/setup_envs.sh          # 需要联网，安装 Isaac Sim 5.1 / PyTorch / DDS
```

### 4. 下载数据集（按需）

```bash
# 全部数据集（约 24 GB）
./scripts/download_datasets.sh

# 或只下载某一个任务
./scripts/download_datasets.sh g1_valve_handwheel_task
```

### 5. 运行任务

每个任务都在自己的目录中启动：

```bash
cd assemble/task/g1_valve_handwheel_task

# 终端 1：启动 Isaac Sim
./run_task.sh

# 终端 2：Quest 3/3S 遥操作
./run_quest3s.sh
```

纯键盘 / 无头模式：

```bash
cd assemble/task/w1_mobile_drilling_task
./run_task.sh --headless
```

策略评估（需要 LeRobot 虚拟环境与策略权重）：

```bash
cd assemble/task/g1_valve_handwheel_task
./benchmark.sh eval --checkpoint /path/to/pretrained_model --episodes 5 --seed 42
```

---

## 任务列表

| 任务目录 | 说明 | 仿真入口 | 遥操作入口 | 评估 |
|---|---|---|---|---|
| `g1_tabletop_grasp_task` | Task01 桌面抓取（回归基线） | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_roof_bolt_sleeve_task` | Task02 顶板螺栓套筒 | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_bin_pick_place_task` | Task03 料箱分拣 | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_hand_cart_push_task` | Task04 手推车推动 | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_valve_handwheel_task` | Task05 阀门手轮旋转 | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `g1_wall_hole_detonator_task` | Task06 炮孔装药 | `./run_task.sh` | `./run_quest3s.sh` | `./benchmark.sh eval` |
| `w1_mobile_drilling_task` | W1 履带底盘 + G1 打钻（主线） | `./run_task.sh` | `./run_quest3s.sh` | — |
| `g1_tabletop_grasp_inspire_task` | Inspire 手版桌面抓取 | `./run_task.sh` | `./run_quest3s.sh` | — |
| `g1_hand_cart_rolling_demo_task` | 手推车滚动 demo（数据集仅有 staging 原始视频） | `./run_rolling_demo.sh` | `./run_quest3s_rolling_demo.sh` | — |
| `quest_dex3_hand_test` | G1 + Dex3 手套测试 | `./run_sim_hand_test.sh` | — | — |

其它目录（`galbot_tracked_teleop`、`astribot_tracked_teleop`、
`valve_dynamic_validation_demo_*` 等）为实验 / 校验脚本，一并保留。

---

## 数据集

全部数据托管在 Hugging Face：
<https://huggingface.co/datasets/lyang116/RMiner>

仓库内布局：

```
datasets/<task_directory>/<lerobot_version>/<dataset_name>/
```

所有数据集均为 **LeRobot v3.0**（`g1_tabletop_grasp_task/lerobot_v21` 为遗留
v2.1 样本），30 FPS，包含头部 + 左右腕部三路相机、29 DoF 本体状态与动作。

| 任务 | 主数据集 | Episodes | Frames |
|---|---|---|---|
| Task01 桌面抓取 | `g1_tabletop_grasp_task/lerobot_v30/detonator_wire_insertion` | 250 | 221,315 |
| Task02 顶板螺栓套筒 | `g1_roof_bolt_sleeve_task/lerobot_v30/roof_bolt_sleeve_installation_final` | 250 | 238,015 |
| Task03 料箱分拣 | `g1_bin_pick_place_task/lerobot_v30/sam3d_metal_component_to_randomized_bin_final` | 250 | 228,875 |
| Task04 手推车推动 | `g1_hand_cart_push_task/lerobot_v30/hand_cart_push_final` | 250 | 73,750 |
| Task05 阀门手轮 | `g1_valve_handwheel_task/lerobot_v30/valve_handwheel_rotation_final` | 250 | 182,300 |
| Task06 炮孔装药 | `g1_wall_hole_detonator_task/lerobot_v30/wall_hole_loaded_detonator_insertion_final` | 250 | 158,190 |

每个任务还包含 `*_replay_ready`（回放增强版，50 episodes）和 `_final` 之外的
基础版本。完整清单、字段说明与下载方式见
[Hugging Face 数据卡](https://huggingface.co/datasets/lyang116/RMiner)。

下载后脚本会把数据放回：

```
assemble/task/<task>/data/<lerobot_version>/<dataset_name>/
```

---

## 环境变量

都可在调用前覆盖：

| 变量 | 作用 | 默认 |
|---|---|---|
| `CONDA_SH` | conda 初始化脚本 | 自动探测 |
| `LEROBOT_ROOT` | LeRobot 仓库根 | `<repo>/policy/lerobot` |
| `POLICY_PYTHON` | 评估/训练 Python | `$LEROBOT_ROOT/.venv/bin/python` |
| `ISAACSIM_ROOT` | 独立 Isaac Sim 安装 | `$HOME/isaacsim` |
| `ISAACSIM_ASSET_ROOT` | Isaac Sim 内置资产库 | `<repo>/assets/isaacsim_assets` |
| `G1_IMAGE_SERVER_IP` | Quest 图像服务 IP | `192.168.2.161` |
| `G1_DRILLING_DEVICE` / `G1_EVAL_SIM_DEVICE` | 仿真设备 | `cuda` |
| `G1_SHOW_CAMERA_VIEWPORTS` | 三路相机窗口 | `1`（`0` 关闭） |
| `G1_LEROBOT_DATASET_DIR` | 遥操作录制目录 | 各任务 `data/lerobot_v30/...` |

完整环境与排错说明见 [`docs_portable_bundle.md`](docs_portable_bundle.md)。

---

## 已知注意事项

1. 仿真端独占 TCP `60000`（图像服务），并使用 `/tmp/g1_*_${UID}.lock` 加锁，
   同一时间只能运行一个任务实例。
2. `scripts/.quest_deps` 未打包。W1/X7S 的 Quest 控制首次运行会自动调用
   `assemble/task/scripts/setup_w1_quest3s.sh` 安装 pin 版本的 Vuer / RealMan
   依赖，离线机器请手动预装。
3. 评估 / 训练需要额外准备 LeRobot 与策略权重；本仓库默认不带权重。
4. `astribot_tracked_teleop` 使用的 `assemble/robot001/` 资产未包含在公开资产包中，
   需要时请单独准备。

---

## 许可证

本仓库代码采用 Apache-2.0。`third_party/` 下的 IsaacLab、unitree_sim_isaaclab、
xr_teleoperate、CycloneDDS 等组件保留其原始许可证；3D 资产与数据集请遵循
各自来源的许可条款。
