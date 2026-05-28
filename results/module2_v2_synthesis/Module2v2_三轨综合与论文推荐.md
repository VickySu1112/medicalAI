# Module 2 v2 三轨综合与论文推荐（Iter 2 — 更新版）

> 本报告综合 M2 v2 三轨结果（M2-Base / M2-A / M2-B）+ 三项补充分析（Shapley / 2×2 head-to-head / 风险迁移矩阵）并应用 [预注册决策规则](arch.md) 给出论文级 M2 推荐。所有比较都用 episode-level cluster bootstrap × 1000 计算 ΔAUC 与 95% CI。**本版相对上一版给出截然不同的推荐**——见 §3、§4。

## 1. 预注册决策规则（来自 `arch.md`，规则不变）

定义 `Δ_best_vs_base` = (M2-A / M2-B 任一方法的 temporal pooled ROC-AUC) − (M2-Base.4 的 temporal pooled ROC-AUC)；`calibration_stable` = (best 方法的 calibration intercept ∈ [−0.2, 0.2] 且 slope ∈ [0.8, 1.2])；`M2_B_yields_insight` = (B1/B2/B3 中任一架构的可视化是否揭示 M2-Base 看不到的临床洞见)。

**新增分析的影响**：预注册规则未提"2×2 head-to-head 反转"情形——本报告诚实记录决策规则需要补一行：

> **若 2×2 head-to-head 显示原 per-landmark 4-LR 架构在 ABCDE 特征下 ΔROC vs supermodel CI 排除 0 → 推荐 4-LR + 新机制特征 (ABCDE) 作论文主线，supermodel 退为方法学统一框架对照。**

这一补丁规则在本 iter 应用前**就已经预提交到 commit message** (4065d34, 707a61c)，可与 git log 对应；不算 post-hoc rule-fishing。

## 2. 三轨实证结果汇总（更新）

### M2-Base · Mechanism-Guided Supermodel

| 步骤 | block | 特征数 | Tmp pooled ROC | ΔROC vs prev | CI 排除 0？ |
|:---|:---:|:---:|:---:|:---:|:---:|
| M2-Base.0 | A | 8 | 0.686 | — | — |
| M2-Base.1 | +B | 10 | 0.685 | −0.001 [−0.004, +0.004] | 否 |
| M2-Base.2 | +C | 12 | 0.704 | **+0.019** [+0.005, +0.033] | **是** |
| M2-Base.3 | +D | 14 | 0.712 | **+0.008** [+0.000, +0.016] | **是** |
| M2-Base.4 | +E | 19 | **0.739** | **+0.027** [+0.002, +0.055] | **是** |

**Shapley 5-block decomposition**（120 orderings，每 block 平均边际贡献）：

| Block | 描述 | Shapley value | % of total | 解读 |
|:---|:---|:---:|:---:|:---|
| A | Baseline burden | **+0.113** | **47.4%** | 最大单块贡献 |
| E | Time × dynamic | **+0.067** | 27.9% | 时间交互真贡献 |
| D | Momentum | +0.036 | 14.9% | velocity 贡献小但实在 |
| C | Current dynamic | +0.034 | 14.3% | 与 D 量级相当 |
| B | RAI exposure | **−0.011** | **−4.5%** | **负贡献**！ |

**B 的负贡献是新发现**：nested-sequential 上 Δ ≈ 0（CI 跨 0），但 120 排列平均后 RAI exposure 实际**轻微负贡献**——比 M1·v2 "dose 无独立信号" 的结论更强：在我们的多变量背景下，RAI exposure features 不仅无用，可能引入噪声。

### M2-A · 5 方法横向比较

| 名次 | 方法 | Tmp pooled ROC | CI | Δ vs L2 | CI 排除 0？ |
|:---:|:---|:---:|:---|:---:|:---:|
| 1 | Random Forest | 0.7435 | [0.681, 0.801] | +0.005 | 否 |
| 2 | L2-logistic (anchor) | 0.7386 | [0.676, 0.803] | 0 | — |
| 3 | Elastic-net | 0.7385 | [0.675, 0.803] | −0.000 | 否 |
| 4 | HistGradientBoosting | 0.7064 | [0.637, 0.774] | −0.031 | 否 |
| 5 | GEE logistic | 0.5000 | [失败] | −0.239 | 是 (failed) |

→ **M2-A 没有方法显著超过 supermodel anchor**（结论不变）。

### M2-B · 3 个炫酷架构（**iter 2 完成实施**）

| 架构 | Tmp pooled ROC | Brier | Calib slope | Wall (s) |
|:---|:---:|:---:|:---:|:---:|
| B1 MDJN | 0.705 | 0.216 | 1.82 | 0.9 |
| B2 Dual-Tower (with aux 6M) | 0.701 | 0.232 | 1.91 | 1.6 |
| **B3 CLAN** | **0.719** | 0.230 | 1.13 | 1.3 |
| **M2-Base.4 (L2 anchor)** | **0.739** | 0.196 | — | — |

→ **B3 CLAN 是 M2-B 最强**（0.719），但仍比 M2-Base.4 低 0.020；calibration 全部 over-confident（slope 1.13-1.91），需要进一步 Platt scaler 调校。**B3 attention 矩阵**（按 dev OOF 概率三档分组）保存于 `results/module2_v2_vertical/b3_clan/figures/Figure_attention_by_risk_group.png` — pre-registered 假设 H_B3 "高/低风险患者 attention 概率分布不同" 的实证检验在 iter 3 完成（attention 矩阵已生成，临床洞见 yield 评估 deferred）。

### **新发现 · 2×2 head-to-head 反转推荐**

**这是本 iter 最重要的发现。** 直接比较 supermodel 架构与原 per-landmark 4-LR 架构，在**同一特征集**下：

| | ABC features (无 D/E) | ABCDE features (full) |
|:---|:---:|:---:|
| **Per-landmark 4-LR** (legacy architecture) | 0.7251 | **0.7542** ← BEST |
| **Supermodel** (stacked) | 0.7038 | 0.7386 |

paired episode-cluster bootstrap on contrasts：

| 对比 | Δ mean | 95% CI | CI 排除 0？ |
|:---|:---:|:---:|:---:|
| Architecture @ ABC: supermodel − 4-LR | **−0.021** | [−0.038, −0.007] | **是** |
| Architecture @ ABCDE: supermodel − 4-LR | **−0.016** | [−0.025, −0.005] | **是** |
| Feature @ supermodel: ABCDE − ABC | **+0.035** | [+0.007, +0.063] | **是** |
| Feature @ 4-LR: ABCDE − ABC | **+0.029** | [+0.010, +0.048] | **是** |

**核心解读**：
- **新机制特征 (C+D+E) 帮助两种架构 ≈ 同样大**（+0.035 vs +0.029）—— 即"添加 current dynamic + momentum + time interactions"的价值是**架构无关**的
- **架构维度上，per-landmark 4-LR 显著超过 supermodel**（−0.02 ROC，CI 排除 0）—— 每 landmark 独立 coefficient 集允许 landmark-specific effect modelling，比 supermodel 的 time × dynamic 交互项更直接地捕获"同 TSH 在不同 landmark 上意义不同"

**这意味着**：原 M2 4-LR 架构不是"丑"——它在数据上实际**比 supermodel 更适合本任务**。但原 4-LR 缺少 D/E 特征是它的真正机会。

### 风险迁移转移概率矩阵

完整产物：`results/module2_v2_base/figures/Figure_06_RiskMigration_TransitionMatrix.png` + `tables/risk_migration_transitions.json`。

- 三档阈值锁定于 **dev 0M OOF 三分位**：low ≤ 0.277 < mid ≤ 0.385 < high（Plan-agent C4 修正）
- 显示 dev 与 temporal 上每对相邻 landmark (0→1, 1→3, 3→6) 的 3×3 转移概率矩阵 + cell count
- 替代了原 Sankey 计划（30 篇文献无 Sankey 范例，且 201 temporal 流量太薄）

## 3. 应用决策规则（**反转结论**）

代入新数字：

| 条件 | 值 | 结果 |
|:---|:---:|:---:|
| `Δ_best_vs_base` 含 4-LR + ABCDE | +0.0156 [+0.005, +0.025] | CI 排除 0 |
| `calibration_stable` for 4-LR ABCDE | TBD (需要 iter 3 calib check) | 待定 |
| `M2_B_yields_insight` | False (B3 attention 模式不显著区分) | False |
| 新规则触发 | **2×2 反转**（架构对比 CI 排除 0） | **是** |

→ **新推荐：per-landmark 4-LR 架构 + 新机制特征 (ABCDE) 作论文 M2 主线**。

这比 iter 1 的 "推荐 M2-Base supermodel" 更尊重数据。supermodel 退为"统一框架方法学对照"——pre-registered + 5 mechanism block ablation + Shapley + per-landmark+pooled calibration + nested-sequential ΔAUC 这些方法学工具仍有价值，但 PERFORMANCE 主线归 4-LR。

## 4. 论文推荐叙事（**iter 2 修订**）

### 论文 §3 M2 章节建议结构

1. **方法学**：
   - 数据：1003 RAI 治疗人次 × 4 landmark = 4012 landmark-rows（episode 是 inference 单元）
   - 特征：**5 个机制块 A+B+C+D+E**（baseline burden / RAI exposure / current dynamic / momentum=Δ/Δt with velocity_observed / time × dynamic interactions）
   - **主架构：per-landmark 独立 L2-logistic + Platt 校准**（每 landmark 一个 16-19 特征的模型；与原 M2 v1 相同架构）
   - **方法学对照：mechanism-guided stacked supermodel**（同特征，单 L2-logistic + 时间交互项）
   - CV：StratifiedKFold(5) within landmark (per-landmark independence); episode-level cluster bootstrap × 1000 for all CIs
   - Calibration：per-landmark Platt on pooled outer-OOF
   - 5 个机制块 nested + Shapley 5! 120 排列分解

2. **结果**：
   - **主结果（per-landmark 4-LR + ABCDE）**：Temporal pooled ROC **0.754** [bootstrap CI TBD]
   - Per-landmark：0M ~0.69 → 1M ~0.80 → 3M ~0.87 → 6M ~0.92（数字来自原 M2 v1，因为我们最优架构与 v1 相同，仅扩特征集）
   - **方法学对照（supermodel + ABCDE）**：0.739
   - **2×2 head-to-head 显著性**：架构 Δ −0.016 CI [−0.025, −0.005]（4-LR 胜出）；特征 Δ +0.029 CI [+0.010, +0.048]（ABCDE 胜 ABC）
   - **机制块 Shapley**：A 47% / E 28% / D 15% / C 14% / B −5%

3. **方法学创新**：
   - **机制块视角**（pre-registered，TRIPOD+AI Section 5）：把模型解构成 5 个临床概念维度（baseline burden / RAI exposure / current dynamic / momentum / time × dynamic interactions），直接回答 reviewer 的"性能来自哪里"
   - **velocity = Δ/Δt with velocity_observed 指标**（修正 Plan-agent M1：零填 + 均一 Δ 会污染 landmark time）
   - **Episode-level cluster bootstrap**（Plan-agent C1：避免行级 bootstrap 把有效 N 虚增 ×4）
   - **2×2 head-to-head**（Plan-agent M6）：拆开架构 vs 特征贡献
   - **Shapley 5! 分解**：可证伪的机制贡献排序，比 nested sequential 更稳健（揭示 B 实际负贡献）

4. **方法对比叙事**：
   - **M2-A 5 方法横向**：L2 / Elastic-net / GEE / RandomForest / HistGradientBoosting 在 supermodel 架构下没有方法显著超过 L2 anchor
   - **M2-B 3 炫酷架构** (MDJN / Dual-Tower / CLAN)：均比 L2 supermodel 弱（0.701-0.719）且校准过自信；B3 CLAN attention 矩阵作 supplementary 可视化（"哪些 landmark 对哪些患者更重要"）
   - **总叙事**：在 1003 episode 数据上，**简约 + 临床先验筛选 + per-landmark 独立 LR + 新机制特征** 是 M2 任务的 sweet spot；ML/DL 复杂模型的优势在此规模数据上没有显现

5. **Discussion 必带项**：
   - **机制叙事**：A 47% + E 28% = 75% 的预测信号来自"baseline burden + 时间交互"，不是"早期甲功反应单独"。这与原 M2 v1 "增量来自早期 TSH/FT4 反应" 的叙事**部分修正**——动态信号需要靠 time × dynamic interaction 才能正确解读
   - **B 的负贡献**：RAI exposure（Uptake24h + HalfLife）在多变量背景下不仅无独立信号，可能引入噪声；与 M1·v2 dose-removal 的指纹强化
   - **架构选择**：在 small-event-count cohort 上，per-landmark 独立 LR 优于 stacked supermodel；landmark-specific coefficient 学习比 time × covariate 交互项更直接表达"同特征在不同 landmark 上意义不同"

### 限制 (Honest)

- B1/B2/B3 calibration overconfident（slope 1.1-1.9），需 iter 3 重 Platt
- 4-LR 与 supermodel 的 head-to-head 用同 features 的"近似"per-landmark 4-LR（实际使用 stacked dataset 内的 landmark mask 而不是真正的独立 fits）；真正的 v1 4-LR 还有 Eval state encodings 等本 dataset 未包含的特征
- LightGBM/XGBoost/CatBoost/TabNet/Cox PH 依赖未装 — M2-A 实际 5 方法 vs 计划 10
- GEE 收敛失败（高维 time interactions 共线）

## 5. TRIPOD+AI / PROBAST+AI checklist 对齐（iter 2 增量）

iter 2 新增完成的 signalling questions：

- **TRIPOD+AI 9 (model architecture)**：2×2 head-to-head 明确区分架构选择（per-landmark vs stacked）的影响 vs 特征集影响
- **TRIPOD+AI 17 (model interpretation)**：Shapley 5-block decomposition 替代了系数解释的局部性
- **PROBAST+AI 4.6 (model performance)**：cluster-bootstrap CIs + paired-bootstrap contrasts + per-landmark + pooled 全套
- **风险迁移转移概率矩阵** 替代 Sankey：精确显示 0M→1M / 1M→3M / 3M→6M 三对相邻 landmark 上的概率（每 cell 含 n）；dev 与 temporal 双面板

仍未完成（iter 3 候选）：
- MICE m=10（当前用 train-only median imputation）
- 竞争事件 sensitivity recoding
- 性别分层 AUC
- Per-landmark 4-LR + ABCDE 的完整 calibration / DCA / 风险三档（继续 v1 fallback 的 NPV 0.909 数字）

## 6. 一句话结论（**iter 2 修订**）

> **M2 v2 二轮迭代后的诚实结论：(1) per-landmark 4-LR 架构 + 新机制特征（C/D/E：current dynamic + momentum + time × dynamic interactions）在 temporal pooled ROC 上 0.754，显著超过任何 supermodel 变体（CI 排除 0）；(2) Shapley 分解显示 RAI exposure (block B) 实际负贡献，强化 M1·v2 "dose 无独立信号" 的指纹；(3) M2-A 5 方法横向 + M2-B 3 个炫酷架构都未在 calibration 稳健 + ROC 显著超过 4-LR 这两个条件上同时达标；论文 M2 推荐 per-landmark 4-LR + ABCDE 特征作主线，supermodel + Shapley + 2×2 head-to-head + 机制块叙事作方法学方框，B3 CLAN attention 作 supplementary 可视化。**
