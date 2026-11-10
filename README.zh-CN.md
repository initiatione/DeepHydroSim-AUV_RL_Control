# DeepHydroSim — 基于强化学习的水下机器人控制

**简体中文** | [English](README.md)

面向自主水下航行器（AUV）的 sim-to-real 强化学习框架，把高保真水动力仿真与可扩展的
RL 训练连接起来。

**核心思路.** [Stonefish](https://github.com/patrykcieslak/stonefish) 提供物理精确的
GIRONA500 级 AUV 六自由度水动力数据，用 MLP（`DeepHydroMLP`）从这些数据中学习非线性
流体动力；训练好的 MLP 以自定义插件形式嵌入 MuJoCo，从而在不付出完整 CFD 仿真代价的
前提下，用真实水动力做快速 RL 训练。

<p align="center">
  <img src="assets/teacher_student_simulators.png" width="88%">
</p>
<p align="center"><sub>(a) <b>教师</b> — Stonefish（NED 坐标系）：高保真数据采集 &nbsp;|&nbsp; (b) <b>学生</b> — MuJoCo（ENU 坐标系）：用学习到的水动力做可扩展 RL 训练</sub></p>

> **本仓库是一个带完整复现工作的 fork。**
> 在原发布代码基础上补充了：修正后的水动力 plant、论文提出的 `PhysicsFeature-MLP`、
> 评估与回放工具链，以及一份完整的复现报告。下面每一个数字都可以从当前代码树重新生成——
> 协议、命令以及**未能复现**的部分见 [`REPRODUCTION.md`](REPRODUCTION.md)。

---

## 复现结果概览

| 项目 | 论文 | 本仓库 |
|---|---|---|
| 水动力 MLP 精度（pooled RMSE） | 11.81 | **2.25**（Baseline）/ **2.06**（PhysFeature）¹ |
| 相对 Stonefish 的开环回放（surge RMSE） | 1.784 → 0.996 m/s | **2.391 → 0.095 m/s**² |
| Stage 1 — 定点航行 | 100% | **100%**（85 步） |
| Stage 2 — 声呐避障 | 100% | **100% / 80% / 80%**³ |
| Stage 3 — A\* 域内导航 | 95% | **90%**（786 步） |

¹ 论文的数字来自一处标签错误：`Thruster::getSetpoint()` 返回的是目标转速（rad/s），
却被当成推力（N）相加。改为记录 `Thruster::getThrust()` 并重新推导标签后，误差下降 84%。
详见 `REPRODUCTION.md` §3.6。

² 九段 × 10 s（共 90 s）在 MuJoCo 中开环回放；两列分别是「无水动力模型」与「学习到的 plant」。
脚本：`scripts/replay_table3.py`。

³ curriculum 的 level 2 / 3 / 4（level 4 = 固定 8 柱 slalom）。论文的 100% 是在旧 plant 上
取得的——旧 plant 把 surge 力硬编码为 `30|u| + 12u²`，满推力终速 20.6 m/s；修正后只有
1.36 m/s，因此同一个任务客观上难得多。

### 哪些结论没能复现

论文主张 `PhysicsFeature-MLP` 能避免闭环发散，理由是普通 MLP 会把加速度通道当作非线性
阻力的「捷径」。但这个发散实际上是实现 bug：估计加速度的卡尔曼滤波器用的是物理步长
（`0.002 s`）而非控制周期（`0.1 s`），使加速度估计被放大 50 倍。修好时间基之后，普通
Baseline 同样能到 100%。因此 PhysFeature 在闭环（95% vs 100%，单 seed）和开环回放
（0.095 vs 0.095 m/s）中都没有可测的优势，尽管离线精度确实好了 8.8%。

---

## 架构

```
┌─────────────────────┐      CSV 日志       ┌─────────────────────┐
│   Stonefish C++     │ ─────────────────▶  │    hydro_MLP        │
│   (cpp_sim/)        │                     │  (训练 DeepHydro    │
│  GIRONA500 AUV      │                     │   MLP)              │
│  8 推进器布局       │                     └──────────┬──────────┘
│  HybridDataCollect  │                                │  .pth 权重
└─────────────────────┘                                ▼
                                            ┌─────────────────────┐
                                            │    mujoco_sim       │
                                            │  HydroInference     │
                                            │  插件 (ENU↔NED)     │
                                            │  SAC · curriculum   │
                                            │  A* 路径规划        │
                                            └─────────────────────┘
```

教师端仿真 **GIRONA500**（`sf::FeatherstoneRobot`，8 推进器）；学生端使用
`mujoco_sim/assets/yuyuan.xml`，其质量、惯量与推进器布局与之对应。

---

## 仓库结构

```
.
├── cpp_sim/                        # Stonefish 数据采集程序
│   └── src/
│       ├── MyAUVManager.{h,cpp}    # 仿真管理器：机器人、场景、数据循环
│       ├── MyAUVApp.{h,cpp}        # OpenGL 界面 / HUD
│       ├── controllers/
│       │   ├── MotionController.h  # 6-DOF → 8 推进器混控
│       │   └── MotionStrategy.h    # PID、MaxThrust、HybridDataCollection、OU 噪声
│       └── utils/
│           ├── DataCollector.h     # CSV 记录（状态 + 指令 + 真实推力）
│           └── analysing_damping.py
│
├── hydro_MLP/                      # MLP 训练流水线
│   ├── config.yaml                 # 模型 / 训练配置
│   └── src/
│       ├── model.py                # DeepHydroMLP、PhysFeatureMLP、build_model()
│       ├── dataset.py              # 连续分块划分（消除时间泄漏）
│       ├── data_process.py         # 反演动力学标签提取
│       ├── prepare_data.py         # 合并原始 Stonefish 日志
│       ├── downsample.py           # 200 Hz → 50 Hz
│       ├── data_analysis.py
│       ├── train.py
│       ├── evaluate.py             # 汇总指标
│       └── evaluate_detail.py      # 逐轴 / 逐样本分析
│
├── mujoco_sim/                     # MuJoCo RL 环境
│   ├── configs/
│   │   ├── config.yaml             # SAC 超参、Hydra 默认项
│   │   ├── env/default.yaml        # 旧 plant（转速当推力标签）
│   │   ├── env/thrust.yaml         # 修正 plant（真实推力标签）
│   │   └── task/
│   │       ├── stage1_navigate.yaml
│   │       ├── stage2_avoidance.yaml
│   │       └── stage3_domain_navigation.yaml
│   ├── scripts/
│   │   ├── train.py                # SAC 训练（curriculum + WandB）
│   │   ├── eval_policy.py          # 确定性 N 回合评估
│   │   ├── bench_control.py        # 闭环跟踪 / 终速基准
│   │   ├── replay_stonefish.py     # 回放记录推力
│   │   ├── replay_table3.py        # 论文 Table III 协议（9 × 10 s）
│   │   ├── evaluate.py             # 被动 viewer 评估
│   │   ├── enjoy_rl.py             # 交互式 rollout
│   │   └── check_env.py
│   └── src/
│       ├── core/
│       │   ├── hydro_plugin.py     # 嵌入 MuJoCo 步进循环的 MLP 推理
│       │   ├── models.py           # MLP 结构的在线镜像
│       │   ├── robot.py            # 运动学、推进器混控
│       │   ├── sensors.py          # IMU / DVL / 声呐封装
│       │   └── scene_builder.py
│       ├── envs/
│       │   ├── auv_base_env.py     # Gymnasium 基础环境
│       │   └── tasks/
│       │       ├── navigation_task.py
│       │       ├── avoidance_task.py
│       │       └── domain_navigation_task.py
│       └── utils/
│           ├── kalman_filter.py    # KalmanFilter6D 加速度估计
│           ├── astar_planner.py    # 占据栅格 A* → 弧长路径
│           └── ...
│
├── REPRODUCTION.md                 # 完整复现报告
├── requirements.txt
├── README.md                       # English
└── README.zh-CN.md                 # 本文件
```

---

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1（建议 CUDA） |
| MuJoCo | ≥ 3.1 |
| Stonefish | 源码编译 — 见 [Stonefish 文档](https://stonefish.readthedocs.io) |
| C++ 编译器 | GCC ≥ 11 或 Clang ≥ 14，CMake ≥ 3.16 |

## 安装

```bash
git clone https://github.com/initiatione/DeepHydroSim-AUV_RL_Control.git
cd DeepHydroSim-AUV_RL_Control
pip install -r requirements.txt
```

编译 Stonefish 数据采集程序：

```bash
cd cpp_sim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 使用流程

### 第 1 步 — 采集水动力数据（Stonefish）

`HybridDataCollectionStrategy` 会依次执行九个确定性机动（surge、heave、roll、pitch、
yaw、两个圆周、三个耦合模式），随后进入 Ornstein–Uhlenbeck 随机激励：

```bash
./cpp_sim/build/MyAUVApp
```

日志写入 `cpp_sim/logs/GeneralMission_<timestamp>.csv`。每行同时记录指令转速（`M_*`）
与**真实螺旋桨推力**（`T_*`，单位 N）。

`SF_MAX_TIME` 可按仿真时间精确停机，让长时间采集确定性地结束，而不是靠外部超时：

```bash
SF_MAX_TIME=17151 ./cpp_sim/build/MyAUVApp
```

### 第 2 步 — 训练水动力 MLP

```bash
cd hydro_MLP
python src/prepare_data.py        # 合并原始日志
python src/data_process.py        # 反演动力学标签（存在 T_* 时优先使用）
python src/downsample.py          # 200 Hz → 50 Hz（4× 抽取）
python src/train.py               # model.type = baseline | physfeature
python src/evaluate.py            # 汇总指标
python src/evaluate_detail.py     # 逐轴分析
```

权重与 scaler 写入 `hydro_MLP/models/`。

**模型：** `DeepHydroMLP` — 12 → 128 → 256 → 128 → 6（71K 参数）。
`PhysFeatureMLP` 先把 12 维输入展开成 33 维基函数
`[ν, ν̇, ν|ν|, νᵢνⱼ]`，再送入同样的 hourglass 结构。

**标签生成：**

```
F_fluid = τ_prop − M_total · a̅
```

其中 `τ_prop` 由记录的真实推力计算，`M_total` 为刚体 + 附加质量对角阵，`a̅` 由
Savitzky–Golay 平滑 + 有限差分得到。

### 第 3 步 — 训练 RL 策略（MuJoCo）

把训练好的 MLP 放到 `mujoco_sim/weights/`（或 `weights_thrust_baseline/`），并把
`configs/env/<profile>.yaml → weights` 指向它。**权重不随仓库发布**（`*.pth`、`*.pkl`
已被 gitignore），需先完成第 2 步。

```bash
cd mujoco_sim

# 修正 plant（真实推力权重）
python scripts/train.py env=thrust task=stage1_navigate
python scripts/train.py env=thrust task=stage2_avoidance
python scripts/train.py env=thrust task=stage3_domain_navigation

# 旧 plant（原发布行为）
python scripts/train.py env=default task=stage1_navigate
```

训练使用 **SAC**（Stable Baselines3），配合 `VecNormalize`、基于滚动成功率的 curriculum
推进、WandB + TensorBoard 日志以及定期 checkpoint。

**跨阶段热启动不等于只加载权重。** 在奖励尺度不同的任务上续训，必须同时继承
`VecNormalize` 统计量与 replay buffer，否则 critic 的 TD target 会按回报尺度之比跳变，
把 actor 直接带飞：

```bash
python scripts/train.py env=thrust task=stage2_avoidance total_timesteps=600000 \
  hyperparams.learning_rate=1e-4 \
  pretrained.model_path=outputs/<stage1>/final_model.zip \
  pretrained.vecnorm_path=outputs/<stage1>/vec_normalize.pkl \
  pretrained.load_buffer=true \
  pretrained.buffer_path=outputs/<stage1>/final_replay_buffer.pkl
```

### 第 4 步 — 评估与回放

```bash
cd mujoco_sim

# 确定性评估
python scripts/eval_policy.py task=stage2_avoidance env=thrust \
  pretrained.model_path=outputs/<run>/final_model.zip \
  +eval_episodes=20 +eval_stage=4

# plant 保真度：把记录的 Stonefish 推力回放到 MuJoCo
python scripts/replay_stonefish.py

# 论文 Table III 协议（九段 × 10 s，五种 plant 变体）
python scripts/replay_table3.py

# 闭环跟踪与终速对比
python scripts/bench_control.py
```

---

## 配置

所有超参由 **Hydra** 管理。`mujoco_sim/configs/config.yaml` 中的关键项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total_timesteps` | 5 000 000 | SAC 总训练步数 |
| `num_envs` | 1 | 并行环境数 |
| `batch_size` | 512 | SAC mini-batch |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `eval_freq` | 10 000 | 评估间隔步数 |
| `gradient_steps` | `${num_envs}` | 保持 UTD = 1:1 |

命令行可覆盖任意参数：

```bash
python scripts/train.py total_timesteps=10000000 hyperparams.batch_size=256
```

> **关于评估噪声.** `EvalCallback` 使用 Stable Baselines3 默认的 **5 回合**评估。当真实
> 成功率为 0.8 时，5 回合估计的标准误是 0.18——同一个策略跑出 0.60 或 1.00 都属正常，
> `best_model` 也可能被噪声选中。判断收敛请用 20 回合以上（`eval_policy.py`），
> 不要看训练曲线。

### Plant 配置档

| 配置档 | 权重目录 | 说明 |
|---|---|---|
| `env/default` | `weights/` | 原发布行为：转速当推力、硬编码 `30\|u\| + 12u²` surge 覆盖、`mlp_drag_scale=0.7` |
| `env/thrust` | `weights_thrust_baseline/` | 修正版：真实推力标签、无覆盖、`mlp_drag_scale=1.0`、`force_clip=2000`、包络外 `(v/v_max)²` 外推、中性浮力 1806.9 N |

---

## Curriculum 阶段

| 阶段 | 任务 | 主要难点 |
|------|------|----------|
| 1 — `navigate` | 定点航行（固定起终点） | 稳定的 6-DOF 控制 |
| 2 — `avoidance` | 带静态障碍的航行 | 碰撞惩罚塑形，仅靠声呐 |
| 3 — `domain_navigation` | 复杂域内 A\* 引导导航 | 弧长进度奖励 + PBRS |

Stage 3 沿 A\* 航点路径建立弧长坐标系，得到单调递增的进度信号，避免奖励平台。

<p align="center">
  <img src="assets/auv_navigation_demo.jpg" width="55%">
</p>
<p align="center"><sub>Stage 3 — GIRONA500 在复杂域内导航。橙色实线：实际轨迹；橙色虚线：A* 参考路径。</sub></p>

---

## 坐标系

MLP 训练数据来自 Stonefish 的 **NED**（北-东-地）坐标系，MuJoCo 使用 **ENU**（东-北-天）。
`HydroInference` 插件在推理前后各施加一次符号翻转：

```python
COORD_MASK = [+1, -1, -1, +1, -1, -1]   # [Surge, Sway, Heave, Roll, Pitch, Yaw]
```

---

## 消融

把 `mlp_drag_scale` 设为 `0.0` 可以关掉学习到的阻力项，同时保留附加质量、浮力等所有其他
动力学项——得到一个纯刚体 + 附加质量的基线。`simplified_mode: true` 通过「仅附加质量」
分支实现同样的效果。

---

## 已知问题

- **权重未纳入版本管理。** `*.pth` / `*.pkl` 被 gitignore，新克隆的仓库没有训练好的 plant，
  需先跑第 2 步，或另行获取权重。
- **`EvalCallback` 只评估 5 回合。** 见上文说明；对外报告的数字请用 `eval_policy.py`。
- **Stage 2 level 4 对回合预算敏感。** 成功轨迹需要 589–1188 个控制步，650 步的训练预算会
  把正确行为截断成 timeout 惩罚。训练与评估都应使用 `env.max_steps=1200`。
- **单环境、CPU 受限训练。** `mj_step` 约占单步的 52%，SAC 更新再占墙钟的 65%，GPU 利用率
  长期在 4% 左右。单纯并行环境收益有限，因为更新开销随样本数线性增长。

---

## 引用

如果使用本工作，请引用原始项目：

```bibtex
@misc{deephyrosim2025,
  author  = {Ray Wu},
  title   = {AUV Deep RL Navigation — DeepHydroSim},
  year    = {2025},
  url     = {https://github.com/RayH-Wu/DeepHydroSim-AUV_RL_Control}
}
```

## 致谢

- [Stonefish](https://github.com/patrykcieslak/stonefish) — 高保真水下仿真器
- [MuJoCo](https://mujoco.org) — 用于 RL 训练的物理引擎
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — SAC 实现
- 来自 University of Girona 的 GIRONA500 AUV
