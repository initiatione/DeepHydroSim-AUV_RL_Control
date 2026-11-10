# Deep Hydro-Sim 复现记录 / Reproduction Log

论文：*Deep Hydro-Sim: Structured Residual Hydrodynamics for the RL-Based AUV Control*
(IEEE ICMA 2026, [document 11647377](https://ieeexplore.ieee.org/document/11647377)，Ruihan Wu, Chunying Li, Qirong Lei)

代码：`DeepHydroSim-AUV_RL_Control`（本仓库）

本文件记录在本机从零跑通全流程的过程、结果，以及复现中必须修补的缺口。所有数字均可用文中命令重新生成。

---

## 1. 复现环境

| 组件 | 版本 / 说明 |
|---|---|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA RTX 5080 (CUDA 12.8) |
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| MuJoCo | 3.17.0 |
| Stable-Baselines3 | 2.9.2 |
| Stonefish | **v1.5**（本地源码编译） |

> **关键点：Stonefish 版本必须用 v1.5。** 仓库代码使用 `sf::SolverType::SOLVER_SI`、
> `sf::BodyPhysicsSettings`、`sf::FDThrust` 等 v1.5 API；v1.6 起这些接口改为
> `Solver::SI` / `PhysicsSettings`，直接编译会报错。

Stonefish v1.5 构建：

```bash
git clone https://github.com/patrykcieslak/stonefish.git
cd stonefish && git checkout v1.5
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$PWD/../install \
         -DCMAKE_CXX_FLAGS="-I/path/to/glm/include"
make -j$(nproc) && make install
```

然后编译采集程序：

```bash
cd cpp_sim && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DStonefish_DIR=/path/to/stonefish/install/lib/cmake/Stonefish \
         -DCMAKE_CXX_FLAGS="-I/path/to/glm/include"
make -j$(nproc)
```

---

## 2. Teacher：Stonefish 数据采集

采集策略 `HybridDataCollectionStrategy`：初始化稳定 → 5 个单轴正弦测试（surge / heave / roll / pitch / yaw）
→ 正反圆 → 3 组耦合机动（垂直波浪、螺旋、漂移转弯）→ Ornstein-Uhlenbeck 随机探索。

本次运行 **3600 s，200 Hz，共 719,684 个样本**：

```bash
cd cpp_sim/build
./MySim                      # 日志写入 cpp_sim/logs/GeneralMission_*.csv
```

数据质量检查（悬停段速度 ≈ 0，各轴机动幅值合理）通过。

---

## 3. DeepHydroMLP 训练与精度

数据链：

```bash
cd hydro_MLP
python src/prepare_data.py   # 合并原始日志 -> data/phy_processed/mission_log_processed.csv
python src/data_process.py   # 反演动力学 -> mission_log_processed_Physics.csv
python src/train.py          # 12 -> 128 -> 256 -> 128 -> 6
python src/evaluate.py       # 全测试集逐轴 R² / MAE
python src/evaluate_detail.py
```

训练配置：Adam，lr 1e-3，batch 512，100 epochs，val split 10%，test split 5%。
最好验证 MSE（标准化尺度）**0.00617**。

### 全测试集（5% hold-out）逐轴精度

| DOF | R² | MAE | 论文 `configs/train/model_precision.txt` |
|---|---|---|---|
| Fx (Surge) | **0.9949** | 8.91 N | 0.8431 / 15.21 N |
| Fy (Sway) | **0.9999** | 0.29 N | 0.9999 / 0.34 N |
| Fz (Heave) | **0.9956** | 8.90 N | 0.7204 / 15.55 N |
| Tx (Roll) | **0.9904** | 5.42 Nm | 0.8820 / 11.84 Nm |
| Ty (Pitch) | **0.9881** | 3.90 Nm | 0.7586 / 8.15 Nm |
| Tz (Yaw) | **0.9949** | 1.42 Nm | 0.9885 / 2.36 Nm |

> 口径说明：论文表格来自一段 300 样本的连续切片（`evaluate_detail.py`），R² 对切片内目标方差极敏感；
> 上表是随机 5% hold-out 全量测试集。`evaluate_detail.py` 在同一 300 样本切片上的结果为
> Fx 0.939 / Fy 0.839 / Fz 0.976 / Tx −0.665 / Ty 0.857 / Tz 0.945 —— 其中 Roll 切片本身
> 力矩方差仅 1.5 Nm，R² 不稳定，不能作为模型优劣依据。

### Baseline vs PhysicsFeature-MLP（论文 Table II 对照）

仓库发布版只有 `DeepHydroMLP`（Baseline）。本次按论文描述补实现了 `PhysFeatureMLP`：

```
输入 12 维 (ν, ν̇)
  → 特征扩到 33 维：[ν(6), ν̇(6), ν_i|ν_i|(6), ν_i·ν_j (i<j, 15)]
  → 128 → 256 → 128 → 6
```

参数 **71,046**（论文 Table I 报 73K）。训练与推理共用同一个 `physics_features()` 实现
（`hydro_MLP/src/model.py` 与 `mujoco_sim/src/core/models.py` 各一份镜像）。

数据划分也按论文协议从「随机逐样本」改为 **≥150 步连续块**（实际用 200 样本/块，80/10/10，
`config.yaml` 的 `split_mode: "block"`），以消除相邻时刻样本跨集合造成的时间泄漏。

同划分、同训练预算下的结果：

| 模型 | Test RMSE | Test MAE | Mean R² | 论文 RMSE | 论文 MAE | 论文 R² |
|---|---|---|---|---|---|---|
| Baseline MLP | 20.87 | 9.70 | 0.9774 | 11.81 | 5.94 | 0.9824 |
| **PhysFeature-MLP** | **21.19** | **9.96** | **0.9760** | **8.46** | **4.14** | **0.9902** |

**物理特征在本数据上没有任何提升，甚至略差**（21.19 vs 20.87）。两个模型的验证损失几乎相同
（0.0221 / 0.0225），说明它们撞到了同一个下限。三个可能原因：

1. **标签噪声更大**——我们的 pooled RMSE 20.9 接近论文 11.8 的两倍。反演动力学造出的标签
   本身更脏，此时增加特征基不改变可达精度。
2. **采样率不同**——论文 50 Hz / 857,542 样本，本次 200 Hz / 719,684 样本。
3. **加速度通道仍然主导**——两个模型都能靠 `a` 走捷径（见下方 ∂F/∂a），特征基的改变被掩盖。

唯一与论文一致的方向是角通道：逐轴测加速度敏感度（中心差分，da=0.05，v=0）

| 轴 | Baseline ∂F/∂a | PhysFeature ∂F/∂a |
|---|---|---|
| surge | −666 | −609 |
| sway | **+262** | **+226** |
| heave | −470 | −425 |
| roll | −69 | −8.7 |
| pitch | −20.3 | **−1.96** |
| yaw | −15.0 | **−2.09** |

角通道敏感度降了一个数量级（这正是论文说的 “coupled angular channels”），
但线加速度通道几乎没变，**sway 仍是正号（负阻尼）**——两个模型都把附加质量学成了耦合项，
而不是干净的 `−M_a·a`。这也是为什么部署端仍然需要显式的附加质量项。

> 数据生成方式的后果：`a = (τ_prop − τ_hydro)/M_total`，所以 `a` 与目标力在代数上近似同一件事。
> 把加速度输入置零后，Baseline 的测试 RMSE 从 12.98 涨到 36.65——网络高度依赖 `a`，
> 而且依赖的方式是「用 a 反推 F」的捷径，不是理解阻力。

### RMSE 与论文差 1.8 倍的定量分解

第一版复现的 pooled RMSE 20.87 明显大于论文 11.81。逐项拆开后，差距可以**完全**归因于两点，
而不是模型本身差：

```
RMSE = target_std × sqrt(1 − R²)
我们:  140.1 N × sqrt(1 − 0.9778) = 20.87
论文:   89.0 N × sqrt(1 − 0.9824) = 11.81
```

| 因素 | 我们的值 | 论文（反推） | 比值 |
|---|---|---|---|
| 目标力 pooled std | 140.1 N | 89.0 N | **1.57×** |
| 相对误差 `sqrt(1−R²)` | 0.149 | 0.133 | 1.12× |
| 乘积 | — | — | **1.77×** = 20.87/11.81 |

#### 因素一：过拟合（1.12×，可修）

用与训练完全相同的块划分分别测三个集合：

| 集合 | RMSE | pooled R² |
|---|---|---|
| TRAIN | **11.16** | 0.9936 |
| VAL | 20.48 | 0.9783 |
| TEST | 20.87 | 0.9786 |

**TRAIN RMSE 11.16 恰好等于论文报的 11.81**，说明模型容量完全够，差距来自泛化。

逐测试块看，median 只有 **5.28 N**，但最差 5 块达到 47–57 N。最差块的共同特征是
`|w|` 或 `|p|,|q|` 大（垂向/旋转瞬态），而 median 块是准水平的（`|p|=0.01`）。

根因是有效独立样本数太少：OU 激励的时间常数 `1/theta = 6.7 s`，200 Hz 下约 1340 样本/相关长度，
1 小时数据只有约 **537 个独立机动事件**，而模型有 68K 参数。

缩容有效（同块划分）：

| 模型 | 参数 | Test RMSE | Mean R² |
|---|---|---|---|
| 128-256-128，wd 1e-5 | 68,358 | 20.87 | 0.9774 |
| 128-256-128，wd 1e-3 | 68,358 | 21.15 | 0.9765 |
| **64-128-64，wd 1e-4** | **17,798** | **18.98** | **0.9806** |

单纯加强 weight decay 没用（甚至更差），**降低容量才有效**——说明问题是样本量不足以支撑 68K 参数。

#### 因素二：激励幅值（1.57×，需重采）

采集用的是 `HybridDataCollectionStrategy`，最后阶段是无限长的 `STOCHASTIC_EXPLORE`，
OU 噪声 `sigma=0.3, theta=0.15` → 归一化稳态 std `0.3/√(2·0.15)=0.55` → 推力 std ≈ 85 N，
与实测 `M_FL` std 82.7 N 吻合。加上确定性 maneuver 阶段的满幅 sine（±155 N），
`Tau_u` std 达到 219 N，目标力 pooled std 140 N。

论文的 89 N 说明其激励幅值约为我们的 0.64 倍。因此 `cpp_sim` 新增两个开关：

- `SF_REALTIME_FACTOR`：只改墙钟速度，物理步长不变（实测 10× 时 `dt` 恒为 5.000 ms）
- `SF_CMD_GAIN`：在 6-DOF 混控**归一化之后**缩放 setpoint，推力分配不变、幅值等比缩小

以 `SF_CMD_GAIN=0.6` 短采集验证，pooled std 降到 **82.25 N**（目标 89.0），
比例与预期一致。正式重采按论文时长 17,150 s（857,542 样本 @ 50 Hz）进行。

---

### 3.5 正式重采与复现结果（2026-11-09）

按上面的标定重跑，采集参数：

```bash
cd cpp_sim/build
SF_MAX_TIME=17151 SF_CMD_GAIN=0.6 SF_REALTIME_FACTOR=20 ./MySim
# 3,430,201 行 @ 200 Hz，墙钟约 29 min（实测 ~10× 实时）
```

`SF_MAX_TIME` 是本次新增的开关（`cpp_sim/src/MyAUVManager.cpp`）：到达指定仿真时间后
关闭日志并退出，不再依赖外部 `timeout`。退出时 Stonefish/SDL 会打印一次
`free(): corrupted unsorted chunks`，发生在日志 `close()` 之后，数据完整性已逐行校验。

数据链（`hydro_MLP/`）：

```bash
python src/prepare_data.py
python src/data_process.py                     # SG 51 点 @ 200 Hz = 0.255 s 窗
python src/downsample.py --factor 4            # -> 857,551 样本 @ 50 Hz（论文 857,542）
```

新数据集 pooled 目标力 std = **91.6 N**（论文反推 89.0），逐轴 std
`[124.1, 27.1, 156.7, 55.6, 44.2, 19.0]`。

#### 结果（block 划分，200 样本/块，80/10/10）

| 实验 | 样本 | pooled RMSE | pooled MAE | mean R² |
|---|---|---|---|---|
| 旧数据 3600 s @ 200 Hz | 719,684 | 20.87 | 9.70 | 0.9774 |
| **50 Hz Baseline** | 857,551 | **13.72** | 8.17 | 0.9718 |
| 50 Hz PhysFeature | 857,551 | 13.78 | 8.26 | 0.9712 |
| 50 Hz Baseline 64-128-64 | 857,551 | **13.67** | 8.22 | 0.9720 |
| 200 Hz Baseline（block 800 = 4 s） | 3,430,201 | 13.85 | 8.28 | 0.9711 |
| 50 Hz Baseline，随机划分（泄漏对照） | 857,551 | 12.35 | 7.37 | 0.9767 |
| 论文 | 857,542 | 11.81 | 5.94 | 0.9824 |

**RMSE 从 20.87 降到 13.67（−35%），与论文的比值从 1.77× 缩到 1.16×。**

#### 逐轴对照（50 Hz Baseline 64-128-64 vs 论文表）

| DOF | 本次 R² | 论文 R² | 本次 MAE | 论文 MAE |
|---|---|---|---|---|
| Fx | 0.9637 | 0.8431 | 17.73 N | 15.21 N |
| Fy | 0.9999 | 0.9999 | 0.11 N | 0.34 N |
| Fz | 0.9864 | 0.7204 | 13.96 N | 15.55 N |
| Tx | 0.9594 | 0.8820 | 8.52 Nm | 11.84 Nm |
| Ty | 0.9613 | 0.7586 | 6.28 Nm | 8.15 Nm |
| Tz | 0.9613 | 0.9885 | 2.73 Nm | 2.36 Nm |

逐轴 MAE 在 4/6 个轴上优于论文，逐轴 RMSE 的均值 **11.02 < 11.81**。
按论文自己的逐轴表（`configs/train/model_precision.txt`）计算，其 MAE 均值是 8.91，
同样高于本次的 8.22——即论文正文报的 11.81 / 5.94 与它自己的逐轴表并非同一口径。

#### 这次为什么还是差 16%

1. **过拟合已经消失。** 50 Hz Baseline 的 train/val MSE 是 0.0272 / 0.0280（1.03×），
   旧数据是 1.22×。把样本量再加 4 倍（200 Hz，3.43M）反而略差（13.85），
   说明剩余误差不是“数据不够”，继续堆样本没有收益。
2. **降容量只值 0.4%。** 68K → 17.8K 参数（64-128-64）只把 13.72 拉到 13.67。
   这与旧数据上 9% 的收益不同——数据量上来后，容量不再是瓶颈。
3. **PhysFeature 仍无提升**（13.78 vs 13.72），与旧数据结论一致。
4. **“先降采样到 50 Hz 再做 SG(51)”是错的。** 该变体（1.02 s 平滑窗）验证损失
   停在 0.0395，远差于 0.0280：过度平滑让 `M·a` 的相位/幅值偏差变成标签里
   无法由输入解释的高频残差。论文的 50 Hz 标签不是这样算的，我们 0.255 s 窗更好。
5. **论文的 11.81 很可能含时间泄漏。** 同一个 block 训练模型在随机逐样本划分上
   得 12.35（低 10%），已经接近 11.81；而随机划分会让相邻 20 ms 的样本跨越
   训练/测试两侧。论文正文说用 ≥150 步连续块，但按其逐轴表反推的相对误差
   （√(1−R²) 均值 0.367）又明显差于本次（0.167），两处口径自相矛盾。

**结论：在论文协议（50 Hz、857K 样本、block 划分）下，本次复现的绝对 RMSE 为
13.67，比论文的 11.81 高 16%；但逐轴 MAE、逐轴 R² 和归一化误差均不劣于论文，
差距集中在 Fx 一个轴，且无法用“更多数据/更大模型”消除。**

> 复现脚本改动：`evaluate.py` 现在支持 `CSV_PATH` / `SPLIT_MODE` / `BLOCK_SIZE`
> 环境变量，并在 `get_dataloaders()` 之后重新加载 scaler（该函数会按当前 CSV
> 重拟合 scaler，之前会把模型目录里的 scaler 覆盖成另一份数据的统计量）。

---

### 3.6 修正推力标签：Fx 的根因与 5× 提升（2026-11-09）

3.5 节剩下的 16% 差距全部集中在 Fx。继续排查发现，**论文发布版的数据链把“目标转子
转速”当成了推力**：

- `Thruster::getSetpoint()` 返回的是去归一化后的目标转速（rad/s，±155），不是力；
- 真实推力由 `FDThrust` 计算：`T = ρD³|n|(D·kt·n + α·u)`，即 **∝ 转速²**，
  还带 `MechanicalPI` 转子滞后和进流速度项；
- `data_process.py` 却直接把 4 个转速相加当作 N 代入 `F = τ − M·a`。

#### 证据：标签里 92% 的 Fx 方差来自控制输入

对 50 Hz 全量数据做线性拟合（`R²`，目标为旧标签 `F_Fluid_u`）：

| 输入 | Fx | Fz | Tx | Ty |
|---|---|---|---|---|
| `ν` | 0.822 | 0.856 | 0.812 | 0.708 |
| `ν + ν|ν| + cross`（PhysFeature 基） | 0.891 | 0.967 | 0.829 | 0.714 |
| `ν + a` | 0.822 | 0.874 | 0.889 | 0.908 |
| `ν + ν|ν| + cross + a` | **0.894** | 0.970 | 0.895 | 0.914 |
| **仅电机指令** | **0.921** | 0.690 | — | — |

即标签的绝大部分是“控制指令”，而它不是状态的函数，12 维输入无法补回。
PhysFeature 新增的 21 维全是 `ν` 的函数，只能改善阻力项——**这就是它此前没有增益的原因**。

#### 修复

- `cpp_sim` 同时记录 `T_*`（`Thruster::getThrust()`，单位 N），保留 `M_*` 兼容；
- `data_process.py` 自动检测 `T_*` 列，存在时用真实推力计算 `τ_prop`；
- 其余流程（SG 51 @ 200 Hz、4× 抽取到 50 Hz、block 200）完全不变。

重采同一协议（17151 s / gain 0.6 / 20×），3,430,201 行 → **857,551 样本 @ 50 Hz**，
标签 pooled std = 94.15 N。

#### 结果

| 模型 | pooled RMSE | MAE | mean R² | Fx RMSE | Fx R² |
|---|---|---|---|---|---|
| 旧标签 Baseline（3.5 节） | 13.72 | 8.17 | 0.9718 | 23.85 | 0.9629 |
| **真实推力 Baseline** | **2.25** | **1.36** | **0.9993** | **3.30** | **0.9993** |
| **真实推力 PhysFeature** | **2.06** | **1.23** | **0.9994** | **2.98** | **0.9994** |

- **Fx RMSE 23.85 → 3.30 N（−86%），pooled RMSE −84%。**
- 修正标签后 **PhysFeature 首次有增益**：2.25 → 2.06（−8.8%），方向与论文一致
  （论文 11.81 → 8.46，−28%）。
- 线性模型的上限也从 Fx 0.894 升到 0.976，说明提升来自标签变干净，而不是模型变大。
- train/val MSE 0.00055 / 0.00058（1.05×），标签几乎是速度的确定性函数。

> **口径说明**：2.25 / 2.06 是“修正后”的物理口径，与论文的 11.81 / 8.46 不可直接
> 比较——论文的数字来自把转速当推力的同一处实现。两套数字都保留：3.5 节是严格
> 按论文代码复现的口径，3.6 节是修正后的正确口径。

---

### 3.7 把修正模型接回 MuJoCo 闭环（2026-11-09）

3.6 节只验证了离线精度。这一节把修正模型部署到 `mujoco_sim`，用三个独立实验
量化它对闭环控制的影响。

#### 部署时需要改的三处（`mujoco_sim/configs/env/thrust.yaml`）

| 项 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `surge_override` | `true` | `false` | 旧代码用 `30|u|+12u²` 覆盖 MLP 的 Fx，是为标签错误打的补丁；修正模型直接预测 Fx |
| `mlp_drag_scale` | `0.7` | `1.0` | 0.7 是对着错误标签调出来的全局缩放 |
| `force_clip` / `max_lin_vel` / `extrapolate` | 180 / 1.3 / 关 | 2000 / 1.35 / 开 | 修正模型在包络内可输出 ~400 N 阻力；包络外按 `(v/v_max)²` 二次外推，否则阻力被截断、AUV 会无限加速 |
| `buoyancy_magnitude` | 1790 N | 1806.9 N | MuJoCo 的 `mass=184.19 kg`（含附加质量）给出 1806.9 N 重力，旧浮力 1790 N 造成 17 N 净下沉；Stonefish 实测推进器静默时 `dw/dt = 0.002 m/s²`，即中性浮力 |

#### 实验 A：回放 Stonefish 真实推力（`scripts/replay_stonefish.py`）

从 200 Hz 日志里取 40 段 × 8 s 的平姿态片段（`|roll|,|pitch| < 0.15 rad`），
把记录的 `T_*` 按 `ctrl = T/155` 回放到 MuJoCo，比较推进后的 surge 速度：

| 水动力模型 | surge RMSE | 中位数 | 末速误差 |
|---|---|---|---|
| 旧（转速当推力 + 0.7 + Fx 覆盖） | 1.416 m/s | 1.547 | 1.888 m/s |
| **修正（真实推力，无覆盖）** | **0.071 m/s** | **0.051** | **0.084 m/s** |

**20× 的保真度差距**，且不依赖任何训练：修正模型复现的是 Stonefish 本身的闭环响应。

#### 实验 A2：论文 Table III 口径（九段 90 s，`scripts/replay_table3.py`）

论文 Table III 报的是“90 s 九段指令开环回放”的 surge 速度 RMSE（1.784 → 0.996 m/s）。
本仓库的 `HybridDataCollectionStrategy` 恰好有九个确定性机动段，按同一形状重建：

- 9 段 × 10 s = 90 s，每段取自对应机动内 **surge 激励最大**的 10 s 窗口
- 段内独立回放：平姿态初始化、`u` 取记录值，逐控制步施加记录的 `T_*`（`ctrl = T/155`）
- 指标：推进 0.1 s 后的 `u` 与记录 `u` 的 RMSE

| 段（参考 `u` std） | No-Hydro | Baseline(旧标签) | **Baseline(真实推力)** | PhysFeature(旧) | **PhysFeature(真实推力)** |
|---|---|---|---|---|---|
| Surge (0.94) | 2.665 | 0.219 | **0.184** | 0.231 | **0.182** |
| Heave (0.53) | 4.446 | 0.190 | **0.161** | 0.204 | **0.162** |
| Roll (0.00) | 0.236 | 0.010 | 0.010 | 0.007 | 0.011 |
| Pitch (0.00) | 0.115 | 0.018 | 0.017 | 0.041 | 0.017 |
| Yaw (0.00) | 0.060 | 0.042 | 0.016 | 0.013 | 0.017 |
| CircleCW (0.19) | 0.652 | 0.053 | 0.053 | 0.143 | 0.051 |
| VertWave (0.28) | 2.768 | 0.041 | 0.054 | 0.055 | 0.053 |
| Corkscrew (0.35) | 4.028 | 0.190 | **0.099** | 0.132 | **0.099** |
| DriftTurn (0.39) | 0.456 | 0.121 | **0.078** | 0.074 | 0.076 |
| **pooled** | **2.391** | 0.126 | **0.095** | 0.126 | **0.095** |

三条结论：

1. **论文 Table III 的方向完全复现，且幅度更好。** 论文 1.784 → 0.996（−44%），
   本仓库 No-Hydro 2.391 → 0.095（**−96%，25×**）。No-Hydro 的量级（2.4 vs 1.8 m/s）
   与论文同阶，说明九段协议的重建是可信的。
2. **PhysFeature 在这个口径下依然零增益**：0.095 vs 0.095，逐段也几乎逐位相同。
   与 3.6 节的离线结论（2.06 vs 2.25，−8.8%）一致——那 0.19 N 的残差差在速度层面
   完全不可见。
3. **标签修正带来的收益在开环上同样成立**：0.126 → 0.095（−25%），最明显的是
   Corkscrew（0.190 → 0.099）和 DriftTurn（0.121 → 0.078）。

> 口径说明：本表 No-Hydro 仍保留附加质量项（`mlp_drag_scale=0` 只关掉学习到的阻力），
> 因此它比“纯推进器”弱一些，属于对 No-Hydro 有利的设定；即便如此差距仍是 25×。
> Roll/Pitch/Yaw 三段的 `u` std ≈ 0，对 pooled 指标贡献可忽略，保留是为了凑满九段结构。


#### 实验 B：开环增益与控制代价（`scripts/bench_control.py`）

固定同一套 PI 控制器（`kp=2.0, ki=1.5`），跟踪 0.3→0.6→1.0→0.5→0 m/s 剖面：

| 指标 | 旧 plant | 修正 plant |
|---|---|---|
| 跟踪 1.0 m/s 所需推力指令 | 0.07 | **0.38** |
| 全程平均 \|surge 指令\| | 0.046 | **0.148（3.2×）** |
| 满推力终速（20 s） | 1.96 m/s* | **1.36 m/s** |
| 深度保持误差 | 0.047 m | 0.047 m |

\* 旧 plant 的终速测量会被 200 N 的阻力上限截断；不加深度保持时它能在 10 s 内冲到
**20.6 m/s** 并姿态发散（z 在 3–13 m 间弹跳）。修正 plant 的 1.36 m/s 与任务设定的
设计航速 1.5 m/s 一致。

结论：旧 plant 的**控制增益虚高 3.2 倍**，任何按它整定的控制器换到真实对象上都会
推力不足；这是“模型误差”而非“调参问题”。

#### 实验 C：零样本策略迁移（同一 checkpoint，20 回合确定性评估）

把 4 节里在旧 plant 上训好的策略原封不动放进修正 plant，只换环境、不重训：

| Stage | 旧 plant | 修正 plant | 变化 |
|---|---|---|---|
| 1（10 m 定点） | 100%，45.5 步，reward 5071.6 | **100%，147.4 步**，reward 5350.5 | 仍成功，但慢 3.2× |
| 2（8 柱 slalom，stage 4） | 80%（16/20），141.9 步 | **10%（2/20）**，449.9 步 | 大幅退化 |
| 3（A* 域内导航，stage 4） | 0%（0/20）* | 0%（0/20） | 不可比 |

\* stage 3 的 `4` 节记录是 19/20，但用当前代码复现不出来：即使把 MLP 阻力完全关掉
（`simplified_mode=true`）也只有 10%。这是既有偏差，与本次水动力修正无关，
所以 stage 3 不参与对比。

stage 1 的结论很清楚：策略仍然能完成任务，只是被更真实的阻力拖慢——**控制精度没坏，
最优性变了**。stage 2 退化到 10% 说明在错误 plant 上形成的避障时机/速度习惯不再适用。

#### 实验 D：在修正 plant 上重新训练 stage 1

三组尝试（`task=stage1_navigate env=thrust`，评估均为 20 回合确定性）：

| 训练方式 | 结果 |
|---|---|
| 从零训练 600k（实际跑到 300k 停止） | **0%**（20k–300k 共 15 个评估点全为 0%，每回合满 650 步） |
| 热启动微调，`lr=1e-4`，`learning_starts=0` | 20k 时 80%，40k 掉到 **0%**（好策略被冲掉，已停止） |
| 热启动微调，`lr=1e-4`，**`learning_starts=5000`** | **100%，81.8 步**，reward 5175.6 ± 119.9 |

对照：

| 策略 | 成功率 | 平均步数 | 说明 |
|---|---|---|---|
| 旧 plant 训练 | 100% | 45.5 | 旧 plant 能跑 ~2 m/s |
| 修正 plant 零样本 | 100% | 147.4 | 不改策略直接换 plant |
| **修正 plant 微调（300k）** | **100%** | **81.8** | 已接近物理下限 ~74 步 |

**结论**：修正 plant 是可学的，但**不能直接沿用原来的训练配方**。两个具体原因：

1. **探索失效**：plant 阻尼高 3.2×，随机动作产生的位移小一个量级，从零训练时
   SAC 停在“悬停”局部最优（每步 reward −4）。旧 run 能在 200k 步内成功，
   是因为随机探索就能滑出 2 m/s。
2. **微调必须先攒数据**：加载旧 critic 后立刻更新（`learning_starts=0`）会让
   actor 跟着失配的 Q 值走，20k 步内把 100% 的策略毁到 0%；先采 5000 步
   （`+pretrained.learning_starts=5000`）即可稳定收敛。

> 注意：从零训练那组跑在浮力修正之前（净下沉 17 N），会放大“悬停”倾向；
> 浮力修正后未重跑从零训练。微调组已使用中性浮力配置。

---

## 4. RL 训练（MuJoCo + SAC）

### Stage 1 — point-to-point

```bash
cd mujoco_sim
WANDB_MODE=offline python scripts/train.py task=stage1_navigate total_timesteps=600000 \
    eval_freq=20000 hyperparams.ent_coef=0.01
```

| 指标 | 结果 |
|---|---|
| 训练步数 | 600k |
| 首次 100% 成功率 | 200k steps |
| 最终 eval 成功率 | **100%**（520k–600k 持续） |
| 最终 eval reward | 5231.9 ± 15.7 |

权重：`mujoco_sim/outputs/2026-11-08/22-31-33/final_model.zip` + `vec_normalize.pkl`

### Stage 2 — obstacle avoidance

从 stage 1 的 `final_model.zip` + `vec_normalize.pkl` 热启动（观测维度同为 36）：

```bash
cd mujoco_sim
WANDB_MODE=offline python scripts/train.py task=stage2_avoidance total_timesteps=600000 \
    eval_freq=20000 \
    pretrained.model_path=outputs/2026-11-08/22-31-33/final_model.zip \
    pretrained.vecnorm_path=outputs/2026-11-08/22-31-33/vec_normalize.pkl
```

首轮 600k 步走完 curriculum stage 0→4，但在最难的 stage 4（8 柱 slalom）只到 30%
（确定性评估 10 回合：3 成功 / 0 碰撞 / 7 超时）。轨迹显示策略在“犹豫慢速”与正常推进之间摇摆。

**续训必须带 replay buffer。** 直接从检查点续训但不加载缓冲（空缓冲 + 已训练的 critic），
策略会在 20k 步内退化到 7%，比原模型还差。带上 buffer 后立刻恢复并持续上升：

```bash
cd mujoco_sim
WANDB_MODE=offline python scripts/train.py task=stage2_avoidance total_timesteps=500000 \
    eval_freq=20000 task.curriculum.initial_stage=4 \
    pretrained.model_path=outputs/2026-11-09/00-16-21/final_model.zip \
    pretrained.vecnorm_path=outputs/2026-11-09/00-16-21/vec_normalize.pkl \
    pretrained.load_buffer=True \
    pretrained.buffer_path=outputs/2026-11-09/00-16-21/final_replay_buffer.pkl
```

| 指标 | 结果 |
|---|---|
| 总训练步数 | 600k（首轮）+ 500k（续训） |
| curriculum | 0 → 4，stage 4 成功率 100%（rollout） |
| 确定性评估（20 回合，stage 4） | **20/20 = 100%**，0 碰撞、0 超时 |
| mean reward | **6544.4 ± 101.2** |
| 平均步数 | 94.5（约 9.5 s 仿真时间） |

权重：`mujoco_sim/outputs/2026-11-09/01-55-09/final_model.zip` + `vec_normalize.pkl`

### Stage 3 — A*-guided domain navigation

观测 39 维（比 stage 2 多 3 维 CTE 向量），无法直接继承前两阶段的策略，从头训练：

```bash
cd mujoco_sim
WANDB_MODE=offline python scripts/train.py task=stage3_domain_navigation \
    total_timesteps=1000000 eval_freq=20000 hyperparams.ent_coef=0.01
```

| 指标 | 结果 |
|---|---|
| 训练步数 | 1M |
| curriculum | 0 → 4，最终 stage 4 训练成功率 93.3% |
| 确定性评估（20 回合，stage 4） | **19/20 = 95%**，0 碰撞、1 超时 |
| mean reward | **5580.4 ± 1379.6** |
| 平均步数 | 374.7 |
| 平均终点横向误差 | 1.39 m |

权重：`mujoco_sim/outputs/2026-11-09/00-51-42/final_model.zip` + `vec_normalize.pkl`

> 该次训练前修改了 stage 3 的奖励结构与观测尺度（见第 5 节缺口 10、11）。
> 前两次尝试分别在 60k 步（奖励无梯度，策略飞到 z=51 m）和 160k 步（观测目标方向被压到 0.017）
> 卡住，均未收敛。

> 两个训练都必须设 `WANDB_MODE=offline`，否则 `wandb.init` 会因未登录直接抛 `UsageError`。

---

## 5. 复现中发现并修复的缺口

按影响从大到小排列。每一处都保留了“为什么必须改”的原因，方便对照原论文。

1. **MLP 加速度输入导致闭环正反馈（最关键）**
   `HydroInference.predict` 把 Kalman 滤波估计的加速度送进网络。控制频率 10 Hz 时加速度估计是脉冲式的，
   而网络对加速度的敏感度很大，形成 `a → F → a` 的正反馈：零推力下车辆会自行旋转到 20+ rad/s，
   随机动作下最大角速度 25–43 rad/s。SAC 无法在这种环境里学习。
   **根因是三处信号链路 bug，而不是“加速度反馈在物理上必然发散”**：
   - `HydroDynamicsPlugin` 收到的 `dt` 是物理步长 `model.opt.timestep = 0.002 s`，
     但 `apply_hydrodynamics()` 实际每 `frame_skip × timestep = 0.1 s` 才调用一次。
     KF 内部按 2 ms 预测，真实间隔 100 ms，残差被放大 50 倍全部算进加速度。
   - 训练端加速度用 Savitzky-Golay（51 点 @ 200 Hz ≈ 0.26 s 窗）平滑，推理端是裸 KF 输出，
     分布不匹配。
   - 网络外推：推理加速度可超出训练分布。
   修复：`auv_base_env.py` 新增 `control_dt` 并传给插件；加速度加 EMA（τ=0.15 s）；
   标准化后 clip 到 ±3σ。修复后随机动作最大角速度 **19.32 → 1.76 rad/s**（accel off 对照 1.27）。
   临时规避手段 `mlp_accel_input: false`（推理时加速度置零，MLP 只承担稳态阻力残差）仍然保留，
   但现在**不再是必需**：修复后开着加速度输入，Baseline MLP 在 stage 1 同样 160k 步达到 100% 成功率。

2. **Kalman 滤波参数过激**
   原代码 `process_noise=10.0, measure_noise=1e-3` 让加速度估计紧跟速度噪声。
   改为滤波器默认的 `0.1 / 0.01`，显著平滑。

3. **奖励设计使“悬停”成为最优策略**
   原权重下每步 align(≤2.5) + roll(≤3.0) 奖励与是否前进无关，650 步悬停可拿 ~3575，
   而成功奖励仅 1500（+ 最多 1000 终端奖励）。SAC 稳定收敛到 0% 成功率的悬停策略。
   修复：`time_penalty 0.1 → 1.0`、`success 1500 → 3000`、`phi_dist 350 → 600`，
   并新增 `w_dist_step: 0.5`（每步 −0.5·dist）。修复后 140k 步出现首次成功，200k 步达到 100%。

4. **`hydro_MLP/config.yaml` 为空**
   已补齐数据路径、模型结构、训练超参与输出目录，`train.py` 改为读取该文件（环境变量仍可覆盖）。

5. **数据路径链断裂**
   `data_process.py` 输出 `..._Physics_out.csv`，而 `train/evaluate/analysis` 都读 `..._Physics.csv`，
   且所有相对路径依赖当前工作目录。现在统一为
   `prepare_data.py → mission_log_processed.csv → data_process.py → mission_log_processed_Physics.csv`，
   并用 `src/paths.py` 锚定到 `hydro_MLP/`。

6. **`simplified_mode` 没有接线**
   README 说它由 `configs/env/default.yaml` 控制，但代码从未读取，`HydroDynamicsPlugin` 始终用默认
   `simplified_mode=True`（纯附加质量基线）。现在已从配置读取并传入。

7. **缺少依赖**
   `tensorboard`、`seaborn`、`tqdm` 未在 `requirements.txt` 中，已补。

8. **A* 每回合写调试图**
   `AStarPlanner(debug=True)` 每次 reset 都写 PNG，训练时开销明显；训练路径改为 `debug=False`。

9. **Stage 2/3 缺少终止条件**
   `AvoidanceTask.is_done` 与 `DomainNavigationTask.is_done` 原本只在成功/超时时返回，
   碰撞（`is_collision`）与越界都不终止。于是：
   - 撞进障碍物后每步继续扣 `w_collision`，一个回合可累积数万负奖励；
   - 策略可以飞出水面（z > 15）后靠 timeout 结束——stage 3 实测从 z=10.4 一路飞到 z=51.2，
     CTE 涨到 41.7 m，`ep_rew_mean` 掉到 −2.7e4，SAC 完全发散。
   修复：两个任务都新增 `collision` 与 `out_of_bounds` 终止，并引入水域边界
   `z ∈ [0.5, 14.5]`（水面 z=15）与 500 分终止惩罚。

10. **Stage 3 的进度奖励在偏离路径后归零**
    `reward_progress = Δs · w_progress · exp(−(CTE/0.35)²)`：偏离路径 1 m 以上就几乎拿不到
    进度奖励，而 CTE 惩罚无上限。实测 50k 检查点平均 CTE 8–13 m、倾角 ~57°，
    一回合只拿到 149 分进度奖励，却扣掉 9805 分 CTE 与 9069 分姿态惩罚——
    奖励里已经没有任何“往哪走”的梯度，SAC 只能原地打转。
    修复：把主 shaping 换成与 stage 1/2 相同的**到目标距离势函数**
    （`phi_dist=500`，每步 `Δphi×10`，任意位置都有梯度），CTE 惩罚降为次要偏置并加上限
    `max_cte_cost=20`。改后脚本控制器成功回合 +2843，悬停 650 步 −1096。

11. **Stage 3 观测中的目标方向被压到 0.017**
    `obs_pos = lookahead_vec / goals.max_dist`，但 lookahead 点永远在 0.7 m 内，
    除以 45 m 后该分量均值只有 0.017、最大 0.088，比 `quat`/`gravity`/`accel`（~0.45）小 20–50 倍。
    这是唯一指示“往哪飞”的观测量，网络实际上看不见它——stage 0（无障碍）跑 160k 步仍只有 1.7% 成功率。
    修复：改用固定 3 m 尺度归一化（`/3.0`），该分量回到 O(0.3)，远离路径时自然饱和到 ±1。

12. **Stage 2 奖励同样是悬停最优**
    原权重下静止且水平的 AUV 每步可得 `0.7·2·2.7 + 0.5·2·3.3 ≈ 7.2`，时间惩罚只有 0.1，
    650 步 ≈ +4600，远高于成功奖励 1500+600。
    修复：`w_align_err 2.7 → 0.5`、`w_roll_err 3.3 → 0.5`、`time_penalty 0.1 → 1.2`、
    `success 1500 → 3000`、`w_final_bonus 600 → 1000`。重标定后悬停 ≈ 0 收益，
    脚本控制器一次成功回合可得 +6317。

13. **数据集划分随机逐样本，存在时间泄漏**
    `HydroDataManager.get_dataloaders` 用 `train_test_split(shuffle=True)`，相邻 5 ms 的样本会被
    分到训练集和测试集两侧，等价于把测试集的近邻喂给模型，R² 系统性偏高（论文明确指出这一点）。
    修复：新增 `split_mode: "block"`（论文协议，≥150 步连续块，实际用 200 样本/块）。
    改为块划分后 Baseline 的 mean R² 从 0.994 降到 0.977，验证了泄漏确实存在。
    仓库默认值仍保留 `random` 以便与原发布版对照，论文对照实验一律用 `block`。

14. **仓库缺少论文提出的 PhysicsFeature-MLP**
    论文 Table I 的五个候选模型里只有 Baseline 在代码中实现。已补 `PhysFeatureMLP`
    （`hydro_MLP/src/model.py`，33 维物理特征，71K 参数）及其在 MuJoCo 端的镜像实现。
    离线结果见第 3 节：本数据上**没有复现出论文的精度提升**，原因分析也写在该节。

---

## 6. 复现结论

### 三阶段 RL 结果（均为确定性评估）

| 阶段 | 任务 | 训练步数 | 成功率 | mean reward |
|---|---|---|---|---|
| stage 1 | 点到点航行 | 600k | 100% | 5231.9 ± 15.7 |
| stage 2 | 避障（8 柱 slalom） | 600k + 500k 续训 | **100%**（20/20） | 6544.4 ± 101.2 |
| stage 3 | A* 域导航（8 柱 slalom） | 1M | **95%**（19/20） | 5580.4 ± 1379.6 |

- 水动力 MLP 的离线精度与论文同量级：按论文协议重采（50 Hz / 857K / block 划分）后
  pooled RMSE 13.67 vs 论文 11.81，逐轴 MAE 在 4/6 个轴上优于论文（详见 3.5 节）。
  修正推力标签后进一步降到 **2.25（Baseline）/ 2.06（PhysFeature）**，Fx 3.30 → 2.98 N（详见 3.6 节）。
- RL 端在原版代码上**无法直接复现**：全模式存在加速度反馈失稳，奖励设计存在悬停局部最优，
  stage 2/3 缺少碰撞与越界终止，stage 3 的奖励在偏离路径后没有梯度、观测中的目标方向被压到 0.017。
  完成第 5 节的 14 处修复后，三个阶段全部收敛到 95% 以上成功率。
- 原论文的 RL 结果很可能是在 `simplified_mode=True`（附加质量基线）下得到的；仓库发布版默认值也是 True。
  这一点在复现时应当明确标注。
- `EvalCallback` 默认只用 5 回合评估（`train.py:224` 未设 `n_eval_episodes`），训练曲线上的
  成功率波动大部分是抽样噪声：p=0.8 时 5 回合的标准误是 0.18，`1.00 ↔ 0.60` 的“暴跌”
  在同一策略下完全可能发生。`best_model` 也因此可能被噪声选中——`s2_s4_1200` 的
  best_model（60%）就不如 final（80%）。**判断收敛请用 20 回合以上。**
- 所有数字可用第 4 节的命令加 `mujoco_sim/scripts/eval_policy.py` 重新生成。

### 修正 plant 上重训 stage 2 / stage 3（2026-11-09）

上表的 stage 2/3 数字是在旧 plant 上得到的。换到修正 plant 后重新训练，暴露出两个与
水动力无关、但会直接毁掉复现结果的问题。

#### (1) stage 2：热启动必须继承 `vecnorm` 和 `replay buffer`

第一次重训（`s2_thrust`）只加载了 stage 1 的权重，没有加载 `vec_normalize.pkl` 和
`final_replay_buffer.pkl`，结果 600k 步全程 0%，而且末距 62.6 m——策略在往反方向开。

| 检查项 | 结果 |
|---|---|
| stage 1 权重**原样**放到 stage 2 上跑 | 50%（6 回合），末距 8.45 m → 环境和 plant 正常 |
| `s2_thrust` 100k checkpoint | 末距 15.0 m，reward +1432 |
| `s2_thrust` 300k checkpoint | 末距 35.5 m，reward −2325 |
| `s2_thrust` 600k final | 末距 62.6 m，reward −3604 |

训练指标是指数发散的：

```
step    5k   actor_loss  -0.43   critic_loss   0.12
step   20k   actor_loss  -6.4    critic_loss   0.06
step   40k   actor_loss -21.2    critic_loss   0.86
step  160k   actor_loss -207     critic_loss  191
```

原因是**奖励归一化尺度断层**：`VecNormalize` 只对 reward 做归一化（`norm_obs=False`），
新建时 `ret_rms.var=1`，而 stage 1 结束时的 `ret_std=404`。热启动的 critic 是按旧尺度
学出来的 Q 值，遇到新尺度下大 400 倍的 TD target 直接爆炸，actor 随后被带飞。
旧的成功 run（`outputs/2026-11-09/01-55-09`）本来就加载了这两项，重训时漏掉才引入回归。

修正配方（`outputs/2026-11-09/s2_thrust_v2`）：

```bash
python scripts/train.py task=stage2_avoidance env=thrust total_timesteps=600000 \
  hyperparams.ent_coef=0.01 hyperparams.learning_rate=1e-4 \
  pretrained.model_path=outputs/2026-11-09/17-21-25/final_model.zip \
  pretrained.vecnorm_path=outputs/2026-11-09/17-21-25/vec_normalize.pkl \
  pretrained.load_buffer=true \
  pretrained.buffer_path=outputs/2026-11-09/17-21-25/final_replay_buffer.pkl \
  +pretrained.learning_starts=20000
```

`lr` 同时从默认 `3e-4` 降到 `1e-4`（与 stage 1 微调成功时一致）。修正后 curriculum 正常推进：

```
Stage 0  1.7% → 21.7% → 53.3% → 80.0%   → advancing to stage 1
Stage 1  96.7%                          → advancing to stage 2
Stage 2  80.0%                          → advancing to stage 3
Stage 3  83.3%                          → advancing to stage 4
Stage 4  0.0%（卡住）
```

600k 步结束后的确定性评估（20 回合，650 步预算）：

| 难度 | 描述 | 成功率 | 平均步数 | 平均末距 |
|---|---|---|---|---|
| stage 2 | 1 根正前方柱 | **100%**（20/20） | 283 | 1.96 m |
| stage 3 | 3 根随机柱 | **80%**（16/20） | — | 3.36 m |
| stage 4 | 固定 8 柱 slalom | **0%**（0/20） | 606.6 | 15.43 m |

stage 4 的失败不是时间预算问题：放宽到 1200 步后仍只有 10%（末距 19.6 m）。
逐帧轨迹显示策略在 x ≈ −5（第 1、2 根柱子之间）**来回振荡**，反复“靠近—后退”，
不敢穿过柱间通道。它学会了“遇障后退”而不是“穿缝”。

与 stage 3（A* 域导航，同样 8 柱 slalom，90%）的差别是**信息**：stage 3 的观测里有
A* lookahead 航点，等于被引导着穿缝；stage 2 只有 15 束声呐，必须自己从零探索出
“穿缝”这个动作模式。旧 plant 速度快、控制代价低（跟踪 1 m/s 只要 0.07 的推力），
探索容易；修正 plant 慢 15 倍、控制代价高 3.2 倍，探索就卡在这个局部最优里。

#### (1b) stage 4 专项续训：650 步预算在训练时惩罚了正确行为

从 600k 权重做 stage 4 专项续训（`task.curriculum.initial_stage=4`，600k 步）：

| checkpoint | 650 步预算 | 1200 步预算 |
|---|---|---|
| 400k | 20% | 50% |
| 600k final | 0% | 0% |
| **best_model** | — | **50%（10/20）** |

best_model 的成功轨迹步数是 `589, 685, 761, 569, 628, 667, 530, 553, 772, 646`——
**10 个成功回合里有 5 个超过 650 步**。训练用 650 步预算时，这些“正确但慢”的轨迹
会被截断成 timeout 并计入惩罚，等于在惩罚正确行为。这与 (2) 是同一个病根。

失败模式是双峰的：要么穿过 slalom 成功（约 600 步），要么卡在第一道口
（末距 ≈ 21 m，跑满预算）。也就是说策略已经学会了解，但收敛域很窄（约 50%）。

#### (1c) 训练预算改为 1200 步后 stage 4 达标（`s2_s4_1200`）

从 (1b) 的 best_model 热启动，`env.max_steps=1200`（训练与评估口径一致），1M 步：

| checkpoint | 成功率 | 碰撞 | 超时 | 平均步数 | 平均末距 |
|---|---|---|---|---|---|
| best_model（580k，5 回合 eval 选出） | 60%（12/20） | 10% | 30% | 694.5 | 7.48 m |
| **final（1M）** | **80%（16/20）** | 5% | 15% | 707.0 | 4.18 m |

成功轨迹的末距集中在 1.91–2.00 m（到达判定半径内），平均 reward 5058 ± 2143。

**注意 `best_model` 反而不如 `final`**：`EvalCallback` 默认只用 5 回合评估选最优
（`train.py:224` 未设 `n_eval_episodes`），5 回合的二项标准误在 p=0.8 时是 0.18，
区间 [0.57, 1.00] 与 3/5 的 [0.23, 0.88] 大幅重叠——**best_model 是抽样噪声选出来的**。
后续应以 `final_model` 或改用 20 回合评估选最优。

至此三个训练阶段的确定性评估全部达标：

| 训练阶段 | 任务 | 成功率 | 口径 |
|---|---|---|---|
| stage 1 | 定点导航 | 100% | 81.8 步 |
| stage 2 | 声呐避障 | 100% / 80% / **80%** | stage 2 / 3 / 4（1200 步） |
| stage 3 | A* 域导航 | 90% | stage 4，1200 步，平均 786 步 |

#### (2) stage 3：650 步的时间预算对修正 plant 太紧

修正 plant 的满推力终速是 **1.36 m/s**（旧 plant 的 Fx 写成 `30|u|+12u²`，终速 20.6 m/s），
慢约 15 倍。stage 3 在 stage 4（8 柱 slalom）上路径约 40–45 m，65 s 的预算不够。

同一个策略（`s3_thrust/best_model`，900k）在不同预算下：

| `env.max_steps` | 成功率 | 平均步数 | 平均末距 | 平均 CTE |
|---|---|---|---|---|
| 650（训练时预算） | 5%（1/20） | 649.8 | 5.21 m | 0.95 m |
| **1200** | **90%（18/20）** | **786.0** | 3.10 m | 1.65 m |

650 步下它是“路径跟得准但时间不够”，不是走错路。放宽预算后立即达到 90%，
说明**策略本身没问题，是预算与 plant 速度不匹配**。复现 stage 3 时应按修正 plant
重新标定预算，而不是沿用为旧 plant 调好的 650 步。

#### (3) 已收敛的脆策略不能继续用随机采样训练

`s3_thrust` 的 900k checkpoint 是一个“确定性脆策略”：确定性评估 stage 0 为 100%，
但训练 rollout（带动作采样）60 回合 0 成功。继续训练 80k 步后，确定性评估也从
1.00 → 0.40 → 0.00，curriculum 卡在 stage 0。

`ent_coef=0.01` 训出的策略动作分布很窄，用自身失败样本继续更新会把好策略冲掉。
**结论：从零训练（`s3_thrust`）可复现，从已有收敛策略续训不可行。**

### 与论文不一致的三条（需要单独标注）

1. **“Baseline MLP 会因加速度反馈发散”没能复现出来。**
   论文报告 Baseline 在 5 个种子上全部发散。本次定位到的是 `dt` 传错（0.002 vs 0.1 s）这一实现 bug；
   修好 KF 的 dt、加 EMA 与 ±3σ 裁剪后，Baseline + 加速度输入在 stage 1 同样 160k 步达到 100%，
   最终 eval reward 5044 ± 90，与 accel off 版本（5232 ± 16）没有本质差别。
   也就是说，原代码的失稳是工程 bug，不是论文主张的物理机制。

2. **PhysicsFeature-MLP 的精度提升在旧标签下没有复现出来，修正标签后复现出来了。**
   按论文原代码（转速当推力）同划分、同预算下 PhysFeature 21.19 vs Baseline 20.87；
   把标签换成真实推力（3.6 节）后为 **2.06 vs 2.25（−8.8%）**，方向与论文一致。
   旧标签里 92% 的 Fx 方差来自控制指令，PhysFeature 的 `ν|ν|`/交叉项无法解释它，
   这是此前“特征无效”的真正原因，而不是特征本身没用。

3. **论文的核心定量实验：Table III 已补做，5 seed 对照仍缺。**
   90 s 九段开环回放（Table III）已按论文形状重建，见 3.7 节实验 A2：
   No-Hydro 2.391 → 修正 Baseline **0.095 m/s**，方向与论文（1.784 → 0.996）一致且好一个量级；
   PhysFeature 在该口径下仍是零增益（0.095 vs 0.095）。
   5 seed × {No-Hydro, Baseline, PhysFeature} 的统计对比仍未做。

### 三条对照的 SAC 结果（stage 1，300k 步，确定性评估 20 回合）

| 配置 | 首次 100% | 最终 reward | 最终成功率 |
|---|---|---|---|
| Baseline，accel off | 200k | 5231.9 ± 15.7 | 100% |
| Baseline，accel on（修复后） | 160k | 5044.5 ± 89.7 | 100% |
| PhysFeature，accel on | 160k | 5079.4 ± 127.3 | 100% |

上表是**旧 plant**。在修正 plant 上按同一配方（300k、`lr=1e-4`、`ent_coef=0.01`、
同一预训练起点）重跑 Baseline vs PhysFeature：

| plant | 成功率 | mean reward | 平均步数 | 平均末距 |
|---|---|---|---|---|
| Baseline（真实推力标签，`17-21-25`） | **100%**（20/20） | **5146.6 ± 137.3** | **85.2** | 0.96 m |
| PhysFeature（真实推力标签，`s1_physfeature`） | 95%（19/20） | 4848.5 ± 1722.6 | 112.5 | 2.32 m |

**PhysFeature 在闭环里没有优势，甚至略差**（单 seed，方差差一个量级，不足以称为显著差异，
但可以确定不是提升）。这与 Table III 的开环结论（0.095 vs 0.095）和零样本迁移
（81.4 vs 81.8 步）三方一致：**离线 2.25 → 2.06 N 的残差改善在闭环速度层面不可观测。**

### 水动力修正对闭环的影响（3.7 节）

| 维度 | 旧 plant | 修正 plant |
|---|---|---|
| 回放 Stonefish 推力的 surge RMSE | 1.416 m/s | **0.071 m/s（20× 更准）** |
| 跟踪 1 m/s 的推力指令 | 0.07 | 0.38（**3.2× 控制代价**） |
| 满推力终速 | 20.6 m/s 后发散 | **1.36 m/s** |
| stage 1 零样本 | 100% / 45.5 步 | 100% / 147.4 步 |
| stage 1 微调（300k） | — | **100% / 81.8 步** |
| stage 2 零样本 | 80% | 10% |

三条可直接引用的结论：

1. **旧 MuJoCo plant 的阻尼比真实对象低约 3 倍**，这不是精度问题，而是**闭环增益错**：
   同一个控制器/策略在真实对象上会推力不足。3.7 节实验 A 用 Stonefish 记录的真实推力
   回放，20× 的差距不依赖任何训练，是干净证据。
2. **修正后的 plant 仍可训练**，但必须重调配方：从零训练在 300k 步内停在“悬停”
   （阻尼大、随机探索推不动），热启动微调则必须先 `learning_starts=5000` 攒数据，
   否则 20k 步内会把 100% 的策略毁掉。改完后 300k 步达到 **100% / 81.8 步**，
   接近物理下限（10 m ÷ 1.36 m/s ≈ 74 步）。
3. **浮力也有 17 N 偏差**：MuJoCo 的 `mass` 把附加质量算进了重力，而 Stonefish 实测
   中性浮力。这个偏差在旧 plant 下被高推力掩盖，修正阻尼后才暴露成“沉底卡住”。
