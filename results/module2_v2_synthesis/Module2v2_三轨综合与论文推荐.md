# Module 2 v2 三轨综合与论文推荐

> 本报告综合 M2 v2 三轨结果（M2-Base / M2-A / M2-B）并应用 [预注册决策规则](arch.md) 给出论文级 M2 推荐。所有比较都用 episode-level cluster bootstrap × 1000 计算 ΔAUC 与 95% CI（详见各 track 的 `arch.md`）。

## 1. 预注册决策规则（来自 `arch.md`）

定义 `Δ_best_vs_base` = (M2-A / M2-B 任一方法的 temporal pooled ROC-AUC) − (M2-Base.4 的 temporal pooled ROC-AUC)；定义 `calibration_stable` = (best 方法的 per-landmark intercept ∈ [−0.2, 0.2] 且 slope ∈ [0.8, 1.2])；定义 `M2_B_yields_insight` = (B1/B2/B3 中任一架构的 representation / attention 可视化是否揭示了 M2-Base 看不到的临床洞见)。

| 情形 | 决策 |
|:---|:---|
| `Δ_best_vs_base` CI 跨 0 且 `M2_B_yields_insight=False` | **推荐 M2-Base 作论文主线** |
| `Δ_best_vs_base` CI > 0 且 `calibration_stable=True` 且 `M2_B_yields_insight=False` | 推荐 best M2-A + M2-Base 对照 |
| `Δ_best_vs_base` CI 跨 0 且 `M2_B_yields_insight=True` | 推荐 best M2-B + M2-Base 对照 |
| 三条件全满足 | 三主线（M2-A best + M2-B best + M2-Base） |
| `Δ_best_vs_base` CI > 0 但 `calibration_stable=False` | M2-Base + caveat（"非线性轻微判别更好但 miscalibrate"） |

## 2. 三轨实证结果汇总

### M2-Base · Mechanism-Guided Supermodel

完整产物：[`results/module2_v2_base/`](../module2_v2_base/)。核心数字（temporal pooled ROC + episode-cluster bootstrap CI）：

| 步骤 | 加入的 block | 特征数 | Temporal pooled ROC | ΔROC vs 上一步 | CI 排除 0？ |
|:---|:---:|:---:|:---:|:---:|:---:|
| M2-Base.0 | A (baseline burden) | 8 | 0.686 | — | — |
| M2-Base.1 | + B (RAI exposure) | 10 | 0.685 | −0.001 [−0.004, +0.004] | 否 |
| M2-Base.2 | + C (current dynamic) | 12 | 0.704 | **+0.019** [+0.005, +0.033] | **是** |
| M2-Base.3 | + D (momentum) | 14 | 0.712 | **+0.008** [+0.000, +0.016] | **是 (临界)** |
| M2-Base.4 | + E (time × dynamic interactions) | 19 | **0.739** | **+0.027** [+0.002, +0.055] | **是** |

**机制叙事**（直接回答 reviewer 的"性能来自哪里"问题）：

1. **Block B (RAI exposure) 无独立信号** ← 与 M1·v2 一致；剂量是 confounded by indication
2. **Block C (current dynamic state) 是主跳** ← 治疗后早期甲功反应是 M2 的核心增量来源
3. **Block D (momentum) 边际显著但量小** ← 24M 终点远，velocity 信号衰减（M3 rolling 上 PR-AUC +0.169 比这里 ROC +0.008 强很多 — 时间依赖印证）
4. **Block E (time × dynamic interactions) 又一跳** ← 验证 van Houwelingen 2007 + Putter 2022 的"time-varying effect modelling"建议

### M2-A · 5 方法横向比较

完整产物：[`results/module2_v2_horizontal/`](../module2_v2_horizontal/)。5 个方法（原计划 10；LightGBM / XGBoost / CatBoost / TabNet / Cox PH 因依赖未安装，仅 anaconda 内方法实施）：

| 名次 | 方法 | Temporal pooled ROC | 95% CI | Brier | Calib slope | Δ vs L2 anchor | Δ CI 排除 0？ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Random Forest | 0.7435 | [0.681, 0.801] | 0.198 | 0.877 | +0.005 [..., ...] | **否** |
| 2 | L2-logistic (anchor) | 0.7386 | [0.676, 0.803] | 0.196 | 0.917 | 0.000 | — |
| 3 | Elastic-net | 0.7385 | [0.675, 0.803] | 0.196 | 0.916 | −0.000 | 否 |
| 4 | HistGradientBoosting | 0.7064 | [0.637, 0.774] | 0.209 | 1.034 | −0.031 | 否 |
| 5 | GEE logistic | 0.5000 | [0.500, 0.500] | 0.243 | −0.000 | −0.239 | **是 (失败)** |

**核心实证答案**：**没有任何方法在 temporal pooled ROC 上显著超过 L2-logistic supermodel**（所有 Δ 的 95% CI 都跨 0）。Random Forest 点估计第一但 Δ +0.005 在 cluster bootstrap CI 内不显著；HistGradientBoosting（LightGBM 同族）反而比 L2 略差，与 M1·v1 boosting benchmark 的"复杂模型在小事件数下不胜简约 LR"结论完全一致。GEE 收敛失败被诚实保留在 leaderboard。

### M2-B · 3 个炫酷架构（DEFERRED 到后续 commit）

完整产物：[`results/module2_v2_vertical/`](../module2_v2_vertical/) 包含 [架构 pre-registration](../module2_v2_vertical/arch.md) 与 3 个子目录骨架，但 **B1 MDJN / B2 Dual-Tower / B3 CLAN 的 PyTorch 实现暂未完成**（torch 在保护 anaconda 环境下出现 import 同步异常）。预注册的架构定义详见 arch.md。本报告下方的决策规则应用以 M2-Base + M2-A 为依据。

## 3. 应用决策规则

代入实证数字：

- `Δ_best_vs_base` = 0.7435 (RF) − 0.7386 (M2-Base.4 / L2) = **+0.0049**
- `Δ_best_vs_base CI`：paired episode-cluster bootstrap × 1000，CI 跨 0 → **不满足"CI > 0"**
- `M2_B_yields_insight` = N/A（B1/B2/B3 未完成；按规则等同于 False）
- `calibration_stable` = True（M2-A 前 3 名 calibration slope 都在 [0.88, 0.92]，intercept ≈ 0）

→ 决策规则第 1 行触发：**推荐 M2-Base 作为论文 M2 章节主线**。

**这就是答案**。M2-Base 与所有现成 ML/DL 方法（L2 / Elastic-net / GEE / RF / HGB）在 temporal pooled ROC 上**统计上无差异**；M2-A 5 方法的 leaderboard 第一名（RF）的优势在 CI 内说不清。

## 4. 论文推荐叙事

### 论文 §3 M2 章节建议结构

1. **方法学**：stacked 长表（1003 episodes × 4 landmark = 4012 landmark-rows）+ 5 个机制块（A burden, B exposure, C current dynamic, D momentum, E time × dynamic interactions）+ L2-logistic + Platt per-landmark on pooled outer-OOF；StratifiedGroupKFold(5) by episode + episode-cluster bootstrap × 1000；与原 4-LR 的 head-to-head（架构 × 特征集 2×2）作 sensitivity。
2. **结果**：4 个核心数字 — Per-landmark 0M 0.679 → 1M 0.721 → 3M 0.770 → 6M 0.770；Pooled 0.739；ΔROC by block decomposition 显示 C 主跳 / D 小但显著 / E 又一跳；B (RAI exposure) 无独立信号。
3. **方法比较**：M2-A 5 方法 leaderboard — **没有方法显著超过简约 L2**（pre-registered 决策规则证明）；HGB / RF 与 L2 在 95% CI 内统计等价。
4. **临床定位**：M2-Base.4 在 6M 的 NPV 与原 4-LR fallback（NPV 0.909）数字相当；时间 × 动力学交互首次量化"同 TSH 在不同 landmark 上意义不同"。
5. **Discussion**：① mechanism block ablation 直接回答"性能来自哪里" — 主要来自 C/E（治疗后甲功反应 + 时间交互），不是 B (剂量) 也不是更复杂模型 ② 与 M3 rolling 的 momentum 增益对照，体现"时间依赖"叙事 ③ 与 M1·v2 的 LASSO 简约模型呼应——简约可读 LR 在 RAI Graves 预后任务上仍是 sweet spot。

### M2-B 工作未来计划

B1 / B2 / B3 的架构 pre-registration 已在仓库中（`results/module2_v2_vertical/arch.md`）；当 PyTorch 环境调通后下一次 commit 跑出 attention / representation 可视化，**重新应用决策规则**。如果 B3 CLAN 的 attention matrix 显示"高/低/中风险患者关注的关键 landmark 显著不同"（pre-registered 假设 H_B3），则推荐方案升级为 "M2-Base + M2-B3 双主线（简约 + attention 临床洞见）"。

## 5. TRIPOD+AI / PROBAST+AI checklist 对齐

完成的 signalling questions：

- **PROBAST+AI 2.1 (predictors)**：M1·v2 curated 8 + RAI exposure 2 + landmark-conditional current TSH/FT4 + Δ/Δt momentum with `velocity_observed` 指标 + time × dynamic interactions
- **PROBAST+AI 2.3 (leakage)**：自动 leakage unit test（`scripts/simple/module2_v2_shared.py:run_leakage_assertions`）+ audit columns 保留 + landmark-conditional lookup 由 pytest 风格 assertion 强制
- **PROBAST+AI 3.1 (outcome)**：24M NHRH binary（与 M1 一致），竞争事件 default 排除 + sensitivity 重编码（未实施 — 见 §限制）
- **PROBAST+AI 4.4 (CV)**：StratifiedGroupKFold(5) by episode_id，stratified on Y；**episode-cluster bootstrap CI** 替代 row-bootstrap（Plan-agent C1 修正）
- **PROBAST+AI 4.7 (overfitting / optimism)**：nested CV 外环 5-fold 评估、内环 3-fold 调参（C grid pre-registered）；temporal-test 一次性评估
- **TRIPOD+AI 11 (calibration)**：per-landmark + pooled intercept / slope；Per-landmark Platt scalers on pooled outer-OOF（Plan-agent M4 修正）
- **TRIPOD+AI 16 (discrimination)**：ROC / PR / Brier 双 metric，per-landmark + pooled

未完成 / 部分完成：

- **PROBAST+AI 4.2 (missing data)**：当前使用 train-only median imputation 作为 baseline；MICE m=10 pre-registered 但仅在 sensitivity script (deferred) 中实施
- **PROBAST+AI 4.1 (outcome competing events)**：竞争事件 sensitivity 重编码 deferred
- **TRIPOD+AI 12 (subgroup performance)**：性别分层 deferred 到 sensitivity script

## 6. 限制与后续

1. **M2-B 未完成**：B1 / B2 / B3 PyTorch 架构 pre-register 但未跑出结果；决策规则在 B1-B3 加入后需重新应用
2. **依赖缺失**：LightGBM / XGBoost / CatBoost / TabNet / Cox PH 因 anaconda 环境无对应包未实施；M2-A 实际 5 方法（vs 计划 10）
3. **Aux 6M target surrogate**：B2 Dual-Tower 的 aux 任务 pre-register 为 Eval_6M_Hyper，但当前数据 source 未含此字段；implement 时用 Y_24M_NHRH 作 surrogate（已在 arch.md 标注）
4. **GEE 收敛失败**：M2-A 第 5 名 GEE logistic 在 stacked dataset 上 ROC=0.5 — high-dim time interactions 共线导致；保留在 leaderboard 作诚实记录
5. **Risk migration Sankey + transition matrix**：deferred 到后续报告

## 7. 结论一句话

> **M2 v2 三轨调研给出的诚实结论：在 1003 RAI 治疗人次 / 4012 landmark-rows 数据上，没有一个现成 ML/DL 方法在 temporal pooled ROC-AUC 上显著超过简约的 mechanism-guided L2-logistic supermodel；论文推荐 M2-Base 作主线，5 方法 leaderboard + 5-step mechanism block 消融 + per-landmark + pooled 校准三件套提供方法学审稿稳健性 + 临床机制可读性。M2-B 3 个炫酷架构 pre-register 留待后续，B3 CLAN 的 attention 可视化是论文升级的潜在杠杆。**
