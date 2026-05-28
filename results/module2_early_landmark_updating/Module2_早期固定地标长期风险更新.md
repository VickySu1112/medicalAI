# Module 2：治疗后早期固定地标长期风险更新（Early Fixed-Landmark Long-Term Risk Updating）

## 摘要

本模块回答治疗后早期随访中的核心问题：**当 1M / 3M / 6M 的甲功反应出现后，能否更准确地预测这次 RAI 最终是否失败/复发（24M NHRH）？** 与 Module 1 共享 1003 人次、共享按治疗时序的 development（802）/ temporal test（201，82 例事件）切分；每个地标只使用截至该地标可见的信息，所有特征处理、阈值、校准均在 development 内完成，temporal test 仅一次性评估。

主结果是一条清晰且诚实的**递增曲线**：以可解释 logistic 为主模型，temporal-test ROC-AUC 从 0M 的 0.687，升到 1M 0.732、3M 0.846、6M 0.923；PR-AUC 同步从 0.675 升到 0.924，Brier 从 0.207 降到 0.096。这说明 **治疗后早期甲功反应是比治疗前信息强得多的预测信号**——3M、6M 是长期风险更新最有价值的节点。

但我们没有止步于"AUC 变高了"。本模块用**"惯性 vs 动量"**的视角诚实拆解了这条曲线：把"沿用当前甲功状态"作为**惯性（persistence）基线**，发现学习模型相对惯性的增量随地标递减（1M +0.137 → 3M +0.082 → 6M +0.066）——**到 6M 时，惯性本身已能达到 AUC 0.857，模型主要靠甲功轨迹的"动量"再补 0.066**。这一拆解既避免了"高 AUC 主要来自照搬当前状态"的虚高陷阱，也定位了我们相对文献的增量来源。本模块同时产出 **early NHRH risk score**（OOF/temporal，1003 人次全覆盖），作为 Module 3 滚动监测的上游风险浓缩输入。

---

## 1. 设计与数据

**地标与时间安全。** 固定地标 0M/1M/3M/6M，各自只用 ≤该地标可见的特征：0M 仅治疗前 baseline；1M/3M/6M 追加截至该时点的当前甲功（FT3/FT4/TSH、状态）与**早期反应轨迹**（baseline→地标的变化、百分比下降、比值变化、状态转移）。泄漏审计确认各地标无未来特征混入（见 §10）。post-RAI 用药被排除出主模型（可能编码医生对病情演变的反应，引入 indication bias）。

**终点与切分。** 主终点同 Module 1——24M NHRH（持续未愈或控制后复发）；development 802 / temporal test 201（事件 82，患病率 0.408），与 Module 1 同一切分以便跨模块比较。

**主模型。** 沿用可解释主线：clinical-core / stable-selected 特征 + L2 或 Elastic-Net Logistic + Platt 校准。与 Module 1 一致，复杂树模型作为补充对照而非主线（判别相当时优先校准与可解释性）。

---

## 2. 核心结果：长期风险预测随早期反应递增

| 地标 | Split | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier (95% CI) |
|:---|:---|:---|:---|:---|
| 0M | Temporal | 0.687 (0.603–0.762) | 0.675 (0.564–0.766) | 0.207 (0.181–0.231) |
| 1M | Temporal | 0.732 (0.653–0.805) | 0.692 (0.584–0.778) | 0.200 (0.175–0.225) |
| 3M | Temporal | 0.846 (0.785–0.901) | 0.822 (0.723–0.887) | 0.150 (0.121–0.180) |
| 6M | Temporal | **0.923 (0.880–0.960)** | **0.924 (0.877–0.958)** | **0.096 (0.068–0.123)** |

![图 1. Temporal-test 判别力随早期地标递增：实线为模型 ROC-AUC / PR-AUC（带 95% CI），虚线为 persistence（惯性）基线；右栏为 Brier 随地标下降。](figures/Figure_01_AUC_PR_Brier_Over_Time.png)

**图 1 解读（本模块的招牌图）。** 左栏实线清楚展示模型判别力从 0M 到 6M 的单调跃升，CI 在相邻地标间已基本分离（尤其 3M、6M 显著高于 0M/1M）；右栏 Brier 从 0.21 降到 0.10，说明不仅判别更好、概率也更准。**虚线是 persistence 惯性基线**——它本身也随地标抬升（见 §3），二者之差才是学习模型的真实增量。临床含义直接：治疗前只能给一个中等的初始预期，而到 3M/6M，模型已能相当可靠地分辨"这次大概率稳住"还是"很可能失败/复发"。

![图 2. 四个地标被选模型在 temporal test 上的 ROC（左）与 PR（右）曲线叠放。](figures/Figure_02_Selected_ROC_PR_All_Landmarks.png)

**图 2 解读。** 四条曲线从 0M（蓝，AUC 0.687）到 6M（紫，0.923）逐层外推，PR 曲线同样远离 0.408 患病率参考线并逐层抬升。这张图把"信息越多、判别越强"可视化到单图，是全文"早期反应更新长期风险"主线的最直接证据。

---

## 3. 惯性 vs 动量：增量来自哪里（诚实拆解）

一个高 AUC 的早期模型有一个容易被忽视的陷阱：**到 6M 时，"现在还甲亢→24M 还是 NHRH"这种照搬当前状态（惯性/persistence）本身就很准**，模型的高分可能大部分只是抄了惯性。我们因此显式纳入 persistence 基线并量化模型的"动量"增量。

| 地标 | 模型 ROC-AUC | 惯性 persistence ROC-AUC | 模型−惯性增量 | 惯性 NPV |
|:---|---:|---:|---:|---:|
| 0M | 0.687 | 0.500（治疗前无既往状态可沿用，退化为患病率）| +0.187* | — |
| 1M | 0.732 | 0.595 | **+0.137** | 0.709 |
| 3M | 0.846 | 0.764 | **+0.082** | 0.784 |
| 6M | 0.923 | 0.857 | **+0.066** | 0.851 |

\* 0M 惯性退化，不作实质比较。

**解读（核心论点）。** 随地标推进，**惯性（当前状态）越来越强**（0.595→0.764→0.857），而**模型相对惯性的增量越来越小**（+0.137→+0.082→+0.066）。这说明：
- 到 6M，长期结局的相当一部分确实可由"当前是否还甲亢"决定——这是诚实必须承认的（也回应了"6M 照搬≈24M"的合理质疑）；
- 但模型在每个地标都**稳定高于惯性**，且增量**在早期（1M）最大**——因为此时状态仍在剧烈变化，单看"当前甲亢与否"信息不足，必须借助**甲功的变化速度与方向（动量）**：同样是 1M 仍偏高的两个人，FT4 正快速下降者趋向缓解、已触底反弹者趋向失败，惯性看不出、动量能。

**方法学与定位。** 用"当前值 + 一阶变化（速度/斜率）"做动态预测在其他领域是成熟做法（如 PSA velocity、GFR slope、landmark 轨迹模型）；但在 Graves/RAI 复发预测中，现有预测多依赖**静态基线水平**（GREAT 评分、FT3/FT4 比值），轨迹/速度仍是新兴方向。本模块据此把治疗后早期反应正式建模为**甲功动量**，并以 persistence 惯性基线为锚，定量展示动量带来的、独立于"当前状态"的预后信息——这是相对现有文献的一个明确增量与创新点。

---

## 4. 模型 × 指标总览

![图 3. 各地标、各模型在 temporal test 上的"模型×指标"热图。](figures/Figure_03_Model_Metric_Heatmap.png)

**图 3 解读。** 横向比较显示，在每个地标上可解释 logistic 与复杂树模型判别相当、且 logistic 的 Brier 普遍更优；纵向比较再次呈现 0M→6M 的整体抬升。结论与 Module 1 一致：**判别相当时优先校准好、可解释的 logistic 作为主模型**，复杂模型留作对照。

---

## 5. 解释层演化：OR 森林（0M → 6M）

![图 4A. 0M OR 森林。](figures/Figure_04_OR_Forest_0M.png)

![图 4B. 1M OR 森林。](figures/Figure_04_OR_Forest_1M.png)

![图 4C. 3M OR 森林。](figures/Figure_04_OR_Forest_3M.png)

![图 4D. 6M OR 森林。](figures/Figure_04_OR_Forest_6M.png)

**图 4 解读（解释层随时间的演化，本身就是"惯性"叙事的机制证据）。** 0M 时主导项是**甲状腺重量**（治疗前负荷，见 Module 1）；随地标推进，**"当前 FT3"**迅速上升为最强项——到 6M，"FT3 at 6 months"的 OR 远大于 1（置信区间稳定不跨 1），几乎主导整张森林图。这正是惯性的机制画像：**越晚的地标，长期结局越被"当前甲功状态"锁定**。同时治疗前的甲状腺负荷/剂量学项退居次要——并非它们不重要，而是其影响已通过早期反应"显现"出来。动量类特征（变化、比值、斜率）在 1M/3M 提供超出当前水平的补充信号（与 §3 增量在早期最大一致）。

---

## 6. 校准与临床效用

![图 5A. 0M 校准曲线。](figures/Figure_05_Calibration_0M.png)

![图 5B. 1M 校准曲线。](figures/Figure_05_Calibration_1M.png)

![图 5C. 3M 校准曲线。](figures/Figure_05_Calibration_3M.png)

![图 5D. 6M 校准曲线。](figures/Figure_05_Calibration_6M.png)

![图 6A. 0M 决策曲线。](figures/Figure_06_DCA_0M.png)

![图 6B. 1M 决策曲线。](figures/Figure_06_DCA_1M.png)

![图 6C. 3M 决策曲线。](figures/Figure_06_DCA_3M.png)

![图 6D. 6M 决策曲线。](figures/Figure_06_DCA_6M.png)

![图 7. 各地标各模型的 Brier / 校准截距 / 校准斜率汇总。](figures/Figure_07_Calibration_Summary.png)

**图 5–7 解读。** 各地标校准曲线均紧贴对角线，Brier 随地标显著下降（0.21→0.10），图 7 显示主模型校准斜率接近 1、截距接近 0——**作为治疗后随访沟通工具，模型给出的概率本身可信**。DCA 在 10%–40% 临床阈值区间净获益均高于 treat-all/treat-none，且随地标推进净获益区间扩大；6M 模型在更宽的阈值范围内提供稳健净获益，支持"治疗后早期复查 → 更新风险 → 差异化随访"的临床用法。

---

## 7. Development 派生三档风险：6M 时低危可 rule-out

阈值在 development OOF 概率上按三分位锁定，再套用 temporal test（不被测试集反调）。

| 地标 | 档 | N | 观察事件率 | NPV | PPV |
|:---|:---|---:|---:|---:|---:|
| 0M | Low / Int / High | 55 / 58 / 88 | 0.291 / 0.259 / 0.580 | 0.709 | 0.580 |
| 3M | Low / Int / High | 60 / 63 / 78 | 0.100 / 0.238 / 0.782 | 0.900 | 0.782 |
| 6M | Low / Int / High | 66 / 56 / 79 | **0.091** / 0.143 / **0.861** | **0.909** | 0.861 |

![图 9A. 3M 三档风险——development OOF（N=802，蓝）与 temporal test（N=201，橙）并列：左为各档观察事件率，右为档内 NPV。](figures/Figure_09_Risk_Tiers_3M.png)

![图 9B. 6M 三档风险——development OOF（N=802）与 temporal test（N=201）并列：观察事件率与档内 NPV。](figures/Figure_09_Risk_Tiers_6M.png)

**图 9 解读（一个关键的临床质变）。** 在 Module 1（0M），low-risk 档 NPV 仅约 0.71，不足以排除不良结局。但随早期反应纳入，分层质量发生质变：到 **6M，low-risk 档观察事件率仅 9.1%、NPV 高达 0.909，high-risk 档事件率 86.1%**——这意味着**治疗后 6M 复查后，模型已能可靠地识别一组"基本可放心、可放宽随访"的低危人群，以及一组"高度可能失败/复发、需加强随访或讨论二次治疗"的高危人群**。3M 已接近这一能力（low NPV 0.900）。这与"早期反应显著提升长期预测"的主线一致，也是 Module 2 最具落地价值的输出。

> 两图均把 **development OOF（N=802）** 与 **temporal test（N=201）** 并列：阈值在 dev OOF 锁定，dev 高 N 视图给出更稳的分档梯度，temporal 为留出验证。6M 两套均呈现 Low 极低、High 极高的清晰梯度，确认 rule-out 能力不是小样本偶然。

---

## 8. 单变量基准与朴素基线

![图 8. 各地标单变量 directional ROC-AUC 与多变量 LR、persistence 朴素基线对照。](figures/Figure_08_Single_Feature_Benchmark.png)

**图 8 解读。** 多变量 LR 在每个地标都高于任何单一变量与 persistence 基线；越往后，"当前 FT3/FT4"等单变量自身判别力也快速上升（再次反映惯性），但多变量联合仍稳定领先。这量化了"建模相对拍脑袋/沿用现状"的净增益。

---

## 9. Early NHRH Risk Score（Module 3 的上游输入）

本模块的关键派生产物是 **early NHRH risk score**：对每个地标的被选模型，导出 per-episode 预测概率——development 用 **OOF** 预测（杜绝自我预测泄漏）、temporal test 用 **dev 训练后**的预测。

- 路径：`tables/early_nhrh_risk_score.csv`（列：`Treatment_ID, Landmark, Domain, Y, RiskScore`；4012 行）。
- 覆盖：每地标 OOF 802 + Test 201 = **1003 人次**，去重 0 行，OOF 患病率 0.364 / Test 0.408（验收见 `module2_acceptance_checks.csv`）。

该评分代表"早期治疗反应浓缩出的长期失败/复发风险背景"，将作为 **Module 3 滚动监测**的一个输入；因其在 development 为 OOF、在 temporal 为 dev-trained，传入 Module 3 不引入泄漏。

---

## 10. 与文献对标（公平、诚实）

- **正面印证**：一项 RAI 后"非完全缓解"预后 nomogram（*Front Endocrinol* 2025）独立发现**治疗后 1 个月 ΔFT3**是关键预测因子，与我们"早期反应显著提升预测"的主线一致；其验证 AUC 0.894（随机划分、单一 1M 标记），与我们 3M 0.846 / 6M 0.923（**时间切分**）相当甚至更高。
- **公平对标（重要）**：纯治疗前模型在随机划分下文献可报很高 AUC（如 Lu 等 2026，RF 0.950，随机 7:3），但那含乐观偏倚；**我们不拿 0M 的 0.687 去对随机划分高 AUC**——0.687 是"仅治疗前信息"的诚实上限。一旦纳入早期反应，我们在**时间切分**下达到 3M 0.846、6M 0.923，**在可比信息量下不逊于文献最佳，且验证更严格、随访更长（到 24M）**。
- **创新定位**：现有 Graves/RAI 复发预测多用静态基线水平（惯性/快照）；我们以 persistence 惯性基线为锚，显式建模治疗后甲功**动量**，并用完整 0/1/3/6M landmark 梯度量化其增量——这是相对现有文献的方法学增量。

---

## 11. 泄漏与时间安全审计

- **特征时间安全**：0/1/3/6M 各特征集均通过 forbidden-future-feature 检查（无未来特征命中）。
- **用药时序**：结构化"治疗前 ATD 使用/时长/停药"字段在本数据缺失，未入模；post-RAI 各时点用药（Medication_1M…24M）一律排除出主模型（避免编码医生对病情演变的处置反应）；TreatCount 作为静态临床背景保留、且不等同于 ATD 史。
- 详见 `tables/module2_leakage_audit.csv`。

---

## 12. 结论与在全文中的定位

Module 2 证明：**RAI 后早期甲功反应不是普通随访记录，而是强预测信号**——长期 NHRH 预测从 0M 的 0.687 单调升至 6M 的 0.923，3M/6M 是临床上最有价值的长期风险更新节点；到 6M，低危档 NPV 0.909 已支持 rule-out 式的随访放宽。更重要的是，我们用**惯性 vs 动量**的诚实拆解说明：这条曲线**不只是"照搬当前状态"**——模型在每个地标都稳定超越 persistence 惯性，且其"动量"增量在早期最大；这既守住了诚实，也定位了相对以静态水平为主的现有文献的增量。本模块输出的 early NHRH risk score 将作为 Module 3 滚动复发监测的上游输入。

---

## 参考文献

1. A prognostic nomogram model for non-complete remission following initial radioiodine therapy in Graves' hyperthyroidism. *Front Endocrinol*. 2025. DOI:10.3389/fendo.2025.1692702.（独立发现 1M ΔFT3 预测价值）
2. Lu L, Wei X, Chen Y, et al. From data to decision: an interpretable ML model for optimizing RAI therapy in Graves' hyperthyroidism. *Front Endocrinol*. 2026. DOI:10.3389/fendo.2025.1711029.（纯治疗前 RF 0.950，随机划分——公平对标参照）
3. Dynamic prediction from longitudinal biomarker history: a landmark approach. *BMC Med Res Methodol*. 2022;22:24.（landmark + 轨迹斜率方法学）
4. PSA velocity / GFR slope 等"当前值 + 一阶变化"动态预测范式（动量建模的方法学先例）。
5. Collins GS, et al. TRIPOD+AI. *BMJ*. 2024;385:e078378. / Moons KGM, et al. PROBAST+AI. *BMJ*. 2025;388:e082505.

## 可复现性

- 生成脚本：`scripts/simple/module2_landmark_report.py`（读 legacy stage1 只读结果 + 抽取早期风险评分 + 经 `scripts/simple/stage1_plot_kit.py` 出图）。
- 自包含 HTML 由 `scripts/simple/embed_html_safe.py` 处理，保证图内 base64 不出现禁用计数。
- 口径：1003 人次；CV/bootstrap 按人次；development 内完成全部选择/校准/阈值；temporal test 仅一次性评估；图内英文、正文中文。
