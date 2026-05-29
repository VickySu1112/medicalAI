# RAI · Graves 病三模块预测框架（M1 → M2 → M3 + 综合 + M4 聚类）

> 本分支承载围绕 **1003 RAI 治疗人次**（development 802 / temporal test 201）的三模块动态预测框架与对应论文级中文报告。所有产物：图、汇总表、报告均自包含可独立阅读；不含原始数据集。

---

## 文献综述包（先看这里）

**30 篇 Q2+ 文献摘要、综合表与主题综述位于：**
[`docs/literature/rai_ml_q2plus_30/`](docs/literature/rai_ml_q2plus_30/)

| 入口 | 用途 |
|:---|:---|
| [`README.md`](docs/literature/rai_ml_q2plus_30/README.md) | 文献包说明与导读 |
| [`index.md`](docs/literature/rai_ml_q2plus_30/index.md) | 30 篇文献索引 |
| [`30_paper_summary_table.md`](docs/literature/rai_ml_q2plus_30/30_paper_summary_table.md) | 30 篇综合汇总表 |
| [`thematic_synthesis.md`](docs/literature/rai_ml_q2plus_30/thematic_synthesis.md) | 主题综合（按问题/方法/证据级别） |
| [`summaries/P01..P30_*.md`](docs/literature/rai_ml_q2plus_30/summaries/) | 30 篇单文摘要（每篇一文件） |
| [`tables/`](docs/literature/rai_ml_q2plus_30/tables/) | 期刊分位证据、开放获取来源、筛选决策、检索日志 |
| [`completion_audit.md`](docs/literature/rai_ml_q2plus_30/completion_audit.md) | 综述完成度审计 |

---

## 模块速览

```
治疗前预期           早期反应更新           滚动复发监测           患者分层（探索）
─────────────       ──────────────       ──────────────       ─────────────────
    M1     ───>        M2 (0M/1M/3M/6M) ───>   M3 (rolling)         M4
                                                                    M4b
                              三模块整合 → 综合
```

| 模块 | 任务 | 中文报告 | 结果目录 |
|:---:|:---|:---|:---|
| **M1** | 治疗前结局预期。LASSO 简约到 **10 个临床可解释特征**（Sex、Thyroid weight、24h RAI uptake、Effective iodine half-life、TRAb、TgAb、TPOAb、FT4 baseline、TSH baseline、Log disease duration）+ L2-logistic + Platt 校准；配 **5 视角可解释性**（OR Forest / SHAP / Permutation Importance / PDP+ICE / Selection Stability + LOO ΔAUC）+ **4 类鲁棒性测试**（敏感性 / 归一化变体 / 5 个独立非 LR 算法 / LASSO 三联稳定性）+ **多种结局验证**（4 档分层 / decile 校准 / 治疗剂量审计 / 滑动窗口）。 | **[M1·v3 主交付（推荐入口）](results/module1_v3/M1v3_治疗前结局预期模型.md)** · [详细分析路径](results/module1_v2_lasso_clean/M1v2_LASSO清洁去剂量基线模型.md) | [`results/module1_v3/`](results/module1_v3/) · [`results/module1_v2_lasso_clean/`](results/module1_v2_lasso_clean/) |
| **M2** | 早期固定地标长期风险更新。**主线 = M2·v2 三轨**：(M2-Base) mechanism-guided serial landmark supermodel — stacked 4012 landmark-rows + 5 个机制块（baseline burden / RAI exposure / current dynamic / momentum / time interactions）+ L2-logistic + Platt + StratifiedGroupKFold + **episode-cluster bootstrap**；(M2-A) 5 方法横向 benchmark — L2 / Elastic-net / GEE / RandomForest / HistGradientBoosting；(M2-B) 3 个炫酷架构（MDJN / Dual-Tower with aux 6M / CLAN with cross-landmark attention，PyTorch 实现）。**iter 2 综合反转推荐**：per-landmark 4-LR + ABCDE + Eval state 成为新主线（Pooled ROC 0.770），supermodel + Shapley + 2×2 head-to-head 作方法学方框；原 4 个独立 LR（v1）保留作 fallback。 | **[M2·v2 综合 + 推荐](results/module2_v2_synthesis/Module2v2_三轨综合与论文推荐.md)** · [M2·v1 fallback](results/module2_early_landmark_updating/Module2_早期固定地标长期风险更新.md) | [`results/module2_v2_base/`](results/module2_v2_base/) · [`results/module2_v2_horizontal/`](results/module2_v2_horizontal/) · [`results/module2_v2_vertical/`](results/module2_v2_vertical/) · [`results/module2_v2_synthesis/`](results/module2_v2_synthesis/) · [`results/module2_early_landmark_updating/`](results/module2_early_landmark_updating/) |
| **M3** | 滚动地标复发监测。多 horizon（H1 主、H6/H12 敏感性），含 inertia + momentum 消融、treatment/patient-level KM 与决策曲线 | [M3·v1](results/module3_rolling_monitoring/Module3_滚动地标复发监测.md) | [`results/module3_rolling_monitoring/`](results/module3_rolling_monitoring/) |
| **综合** | **三模块整合论文：将 M1→M2→M3 串成连续的"治疗前预期 → 早期更新 → 滚动监测 → 治疗级分层"双时间尺度路径，含摘要、方法、关键发现汇总、临床意义、局限与下一步** | [整合论文·v1](results/整合论文_RAI三模块双时间尺度框架.md) | [`results/整合论文_RAI三模块双时间尺度框架.md`](results/整合论文_RAI三模块双时间尺度框架.md) |
| **M4** | 患者分层聚类（探索性）。两种平行做法：M4 在基线特征上聚类，M4b 在 M2 输出的 4 维风险轨迹上聚类；二者结果合并报告 | [M4·v1 (含 M4b)](results/Module4_患者分层与风险轨迹聚类.md) | [`results/module4_baseline_clustering_trajectory/`](results/module4_baseline_clustering_trajectory/) · [`results/module4b_trajectory_clustering/`](results/module4b_trajectory_clustering/) |

---

## 关键发现（按模块一句话版）

### M1 — 治疗前结局预期

- **模型**：10 个 LASSO 简约特征（Sex、Thyroid weight、24h RAI uptake、Effective iodine half-life、TRAb、TgAb、TPOAb、FT4 baseline、TSH baseline、Log disease duration）+ L2-logistic + Platt 校准。Temporal-test 性能 **ROC-AUC 0.686 (95% CI 0.606–0.763) / PR-AUC 0.667 / Brier 0.208**；校准 **slope 0.90 / intercept ≈ 0**；DCA 在 10–40% 临床阈值区间均高于 treat-all / treat-none。
- **可解释性 5 视角收敛**：**Thyroid weight 是模型的主要驱动力**（5 视角全 Top 1；OR **2.56** (CI 2.03–3.23, p < 10⁻¹⁴)；mean\|SHAP\| ≈ 0.66；LOO ΔAUC = +0.0945 (CI 0.063–0.126)）；**Log disease duration** 提供病程维度（OR **1.24**, p = 0.014）；**TPOAb** 提供稳定的边际负向贡献（OR **0.82**, p = 0.028），可能反映向桥本式甲减转归倾向；其余特征（Sex、TRAb、TgAb、HalfLife、Uptake24h、FT4、TSH）联合提供多维稳定性与子组稳健。
- **鲁棒性 4 类测试全部通过**：(1) 敏感性 — VIF max 1.31 / 性别分层 AUC CI 完全重叠 / class_weight Δ −0.0003 / multi-seed std 0.001；(2) Thyroid weight 的 7 个归一化变体（raw / Weight / BSA / Height / BMI / log / log(/BSA)）在 temporal 上统计等价；(3) 5 个独立非 LR 算法（RandomForest / GradientBoosting / KNN / SVM-RBF / MLP）temporal AUC 在 0.66–0.68 一带，与 L2-logistic 同档；(4) **LASSO 三联稳定性**（bootstrap × 500 + Meinshausen-Bühlmann subsample × 100 + subgroup × 8 splits）——Thyroid weight 在 500 次 bootstrap 中每次都被选中（频率 1.000）。
- **治疗结局验证**：**Q1–Q4 四档**是分层 sweet spot（temporal spread 0.406；顶档事件率 60.6% / 底档 31.2%）；decile 校准 dev OOF 贴对角线、temporal 高低端轻度发散；治疗剂量审计显示预测风险与实际 RAI 总剂量 Spearman ρ = +0.69 dev / +0.80 temporal（**呈现医生按临床严重度滴定总剂量的实证指纹**）；沿预测百分位的滑动窗口观察 NHRH 曲线在 dev 与 temporal 上形态高度一致。
- **临床定位**：M1 是**治疗前咨询与预期管理工具**——推荐 Q1–Q4 四档作临床呈现；Low 档 NPV 0.70–0.72 不足以做 rule-out（rule-out 决策点在 M2 的 6M 节点 NPV 0.909）。腺体重量是临床关注的首要项，但 Log disease duration、TPOAb 等特征提供有意义的辅助维度，使个体化沟通更立体。

### M2 — 早期固定地标长期风险更新

- **主线（iter 2 修订）**：架构 = **per-landmark 4-LR + 新机制特征 (ABCDE)**，Temporal pooled ROC **0.754** —— 显著超过 supermodel (0.739)，CI 排除 0（**2×2 head-to-head 反转 iter 1 推荐**）。原 M2 v1 4-LR 架构不"丑"，它在 1003 episodes 上**真的更适合**本任务；改进点不在架构而在特征集（加 current dynamic + momentum + time × dynamic interactions）。
- **5 个机制块** (A burden / B exposure / C current dynamic / D momentum=Δ/Δt / E time × dynamic)：episode-cluster bootstrap × 1000 + per-landmark Platt on pooled outer-OOF。
- **Shapley 5! 分解**（120 排列）：A **47.4%** / E 27.9% / D 14.9% / C 14.3% / **B −4.5%（负贡献！）** — 比 nested sequential 的 Δ ≈ 0 更强：RAI exposure 在多变量下实际拖累模型，强化 M1·v2 "dose 无独立信号" 的指纹。
- **2×2 head-to-head（架构 × 特征集）**：(per-landmark 4-LR + ABCDE) 0.754 > (per-landmark 4-LR + ABC) 0.725 > (supermodel + ABCDE) 0.739 > (supermodel + ABC) 0.704。架构 Δ −0.016 CI [−0.025, −0.005] (4-LR 胜)；特征 Δ +0.029 CI [+0.010, +0.048] (ABCDE 胜) — 两者都 CI 排除 0。
- **M2-A 5 方法横向**：Random Forest 0.7435 ≈ L2 0.7386 ≈ Elastic-net 0.7385 > HGB 0.706 ≫ GEE 0.500 (failed)。没方法显著超过 supermodel。
- **M2-B 3 个炫酷架构**（iter 2 完成）：B3 CLAN 0.719（最佳）< B1 MDJN 0.705 ≈ B2 Dual-Tower 0.701；calibration overconfident (slope 1.1-1.9)；B3 attention 矩阵按风险三档分组保存。所有 B 系列均不及 supermodel。
- **风险迁移转移概率矩阵**（替代原 Sankey 计划）：3×3 转移矩阵 × 3 对相邻 landmark × 2 split (dev/temporal) = 6 panel；tier 阈值锁定于 dev 0M 三分位 (low ≤ 0.277 < mid ≤ 0.385 < high)。
- **临床定位不变**：M2 = "治疗后早期长期风险更新"，回答 "3 个月 / 6 个月时还要不要担心 24 个月以后"；6M 节点 NPV 0.909 支持 rule-out。
- **详见 [iter 2 综合报告](results/module2_v2_synthesis/Module2v2_三轨综合与论文推荐.md)**（含决策规则补丁 + 论文叙事 + Shapley + 2×2 全表）。

### M3 — 滚动地标复发监测

- 单次 H1（下一窗口甲亢）为**中等判别力**（temporal ROC-AUC 0.826、PR-AUC 0.359、Brier 0.052、NPV 0.969）。
- **核心新颖性**：在"当前甲功（**惯性**）"之上加入"甲功轨迹（**动量**：一阶变化/速度）"使 H1 的 **PR-AUC 显著提升 +0.169（95% CI 0.027–0.308）**；再叠加 early risk score 几乎无新增贡献 —— 监测期决定复发的不是"现在是什么样"，而是"正在往哪走"。
- 治疗级聚合后，**低危事件率 7.5% vs 高危 51.2%**（高/低约 6.8 倍）；Harrell C-index 0.79、log-rank P < 0.001；病人级敏感性分析确认非伪重复。
- **临床定位**：M3 是"随访期滚动复发预警"，回答"下一次复诊间隔可不可以拉长 / 要不要提前干预"；动量项是它优于 persistence 与"仅看当前甲功"的关键。

### 综合 — 三模块整合（双时间尺度 landmark 框架）

- 把 RAI 治疗后的风险评估从"一次性预测"扩展为**三个真实决策时刻的连续路径**：治疗前结局预期 → 治疗后早期长期风险更新 → 随访期滚动复发监测，并以**治疗级聚合**给出可操作的差异化随访分层。
- 增量主要来自**治疗后甲功动态（动量）**而非更多静态信息；坚持时间切分、校准、决策曲线与朴素基线对照，使其更适合**风险分层与低危排查**，而非单次高精度报警。
- 与已发表 RAI/Graves 预测工作的差异：① 同时覆盖三个临床时刻的连续框架（多数前作只做单时点）；② 时间外验证（多数前作随机划分或内部 bootstrap）；③ 显式区分"惯性"与"动量"并量化动量增量。

### M4 — 患者分层聚类（探索性）

- **M4 基线表型聚类（反例）**：在 12 个基线特征上 KMeans (k=3, silhouette 0.142)。**Development 上事件率跨度 0.159，temporal 上坍缩到 0.024，KM log-rank 三簇全 ns**（p 0.58–0.94） —— **基线特征聚类不可迁移**。
- **M4b 轨迹聚类（正例）**：在 M2 输出的 4 维风险轨迹（0M→1M→3M→6M）上 KMeans (k=3, silhouette 0.484)。三种轨迹型：**下降型 60%、上升型 21%、持续高位 14%**；temporal 事件率跨度 **0.737**，三条 KM 曲线均 **p < 1e-4** —— **轨迹聚类可迁移**。
- **"晚期上升型"**（约 21% 患者）基线看似温和（风险 0.36），到 6M **翻倍至 0.75**，24M NHRH ≥ 80%。**单靠基线完全错过 — 1–3M 轨迹监测才能抓到**。
- **临床定位**：M4/M4b 是辅助沟通工具（"你属于下降型 / 上升型"比"你的概率是 0.43"更易沟通），同时为论文"动量 > 惯性"主题在分层层面提供另一个证据。

---

## 数据 / 合规

- **N = 1003 治疗人次**（episode-level），dev 802 / temporal test 201；分析单位为人次，重复治疗按独立 episode 处理，不报告 unique-patient 数。
- 切分按治疗时序（temporal split），所有特征处理、阈值选择、模型选择、校准均在 development 内完成，temporal test 仅最终一次性评估。
- 不含原始数据集 — 仓库只保留汇总表（性能 / CI / manifest / profile / coverage / 阈值）、图、报告。
- 所有比较一律包含 persistence / 朴素基线对照（M2、M3 中显式列出）。
- 引用以 Q1/Q2 期刊为主（TRIPOD+AI、PROBAST+AI、Van Calster calibration、Vickers DCA、Putter / Houwelingen landmarking、Rizopoulos joint models 等，详见上方文献综述包）。

---

## 研究设计文档

- [`docs/RAI三模块研究设计与全文基调.md`](docs/RAI三模块研究设计与全文基调.md) — 全文研究设计与基调
- [`docs/Baseline-only高级机器学习补充分析方案.md`](docs/Baseline-only高级机器学习补充分析方案.md) — M1 增强方案设计
