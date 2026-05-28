# Graves 甲亢碘-131治疗后复发/治疗失败预测研究：三模块研究设计与全文基调

## 一句话总基调

本研究不应写成“单纯用机器学习预测甲亢复发”，而应写成：

> **一个覆盖治疗前结局预期、治疗后早期长期风险更新、以及随访期滚动复发监测的双时间尺度、可解释、校准良好的 RAI 治疗决策支持框架。**

也就是说，全文主线是：

```text
治疗前 0M：估计患者接受 RAI 后 24M 内稳定控制 / 治疗失败风险
        ↓
治疗后 1M / 3M / 6M：利用早期甲功反应更新长期失败/复发风险
        ↓
随访期 3M / 6M / 12M / 18M：每次复查后预测下一窗口甲亢/复发风险
        ↓
治疗级风险聚合：形成低危 / 中危 / 高危分层，指导差异化随访
```

全文不应强调“算法越复杂越好”，而应强调：

1. **时间安全**：每个模型只使用该时间点之前已经可见的信息。
2. **临床可解释**：主模型优先采用可解释、校准好的 logistic / elastic-net logistic。
3. **机器学习增益验证**：复杂 ML 模型作为 benchmark 和 nonlinear challenge，用来检验非线性/交互是否带来增益，而不是为了堆算法。
4. **临床落脚点**：输出患者可理解的风险概率和风险分层，而不是只报告 AUC。
5. **双时间尺度**：长期结局预测回答“这个疗程最终稳不稳”，短期滚动预测回答“下一阶段会不会再甲亢”。

---

# 总体研究框架

建议最终将全文设计为 **三个模块**，而不是把 0M/1M/3M/6M/12M/24M 每个阶段都机械地做“下一阶段预测 + 最终结局预测”。后者虽然完整，但容易显得啰嗦、模型过多、终点发散。

最推荐的三个模块是：

| 模块 | 名称 | 核心临床问题 | 主要时间点 | 主要输出 |
|---|---|---|---|---|
| Module 1 | 治疗前 RAI 结局预期评估 | 患者治疗前能否估计两年内稳定控制/治疗失败风险？ | 0M | 24M NHRH 风险 / stable control 概率 |
| Module 2 | 治疗后早期长期风险更新 | 早期甲功反应出现后，长期失败/复发风险是否能明显更新？ | 1M / 3M / 6M | 更新后的 24M NHRH 风险 + 早期风险评分 |
| Module 3 | 随访期滚动复发监测 | 每次复查后，下一阶段会不会再次甲亢/复发？ | 3M / 6M / 12M / 18M | H1/H6/H12 下一窗口甲亢风险 + 治疗级风险分层 |

24M 建议作为最终结局，不再作为“预测阶段”。12M/24M Hyper / Normal / Hypo 状态预测可以作为 secondary / supplementary endpoint，不建议作为全文主线。

---

# Module 1：治疗前 RAI 结局预期评估模型

## 1. 这个模块是干什么的？

这个模块回答患者治疗前最关心的问题：

> **“我现在这种情况接受碘-131治疗后，两年内大概率能稳定控制吗？还是更可能持续不愈/复发？”**

它的定位是 **pre-RAI outcome expectation model**，中文可以写成：

> **治疗前 RAI 结局预期评估模型**

注意：如果没有未接受 RAI、接受 ATD、手术等对照组，不建议在论文里直接写成严格因果意义上的 **treatment benefit prediction**。更稳妥的说法是：

- expected outcome after RAI
- individualized RAI outcome prediction
- pre-RAI outcome expectation
- predicted probability of 24-month stable control after RAI

中文可以说“帮助患者理解接受 RAI 后的预期结局”，不要过度宣称“预测 RAI 的净获益”。

## 2. 需要用到哪些数据？

只允许使用 **0M 治疗前或 index RAI 时点已经可见的数据**，不能使用 1M/3M/6M 甲功反应，也不能使用 post-RAI 用药情况。

建议纳入的数据类型：

### 2.1 基线临床资料

- 年龄
- 性别
- BMI，如有
- 病程，如有
- 是否 Graves 眼病，如有
- 既往治疗次数 / TreatCount
- 既往 RAI 史，如有
- RAI 前 ATD 使用史、ATD 使用时长、停药间隔，如后续能补齐，强烈建议加入敏感性分析

### 2.2 治疗前甲功和免疫活性

- FT3 0M
- FT4 0M
- TSH 0M
- TRAb / TBII
- TPOAb
- TgAb
- FT3/FT4 ratio

### 2.3 甲状腺负荷与 RAI 剂量学

- 甲状腺重量 / 体积
- RAI 给药剂量
- 24h 摄碘率
- 3h 摄碘率，如有
- 有效半衰期，如有
- 每克甲状腺剂量
- Dose × Uptake24h
- Dose × HalfLife
- Dose × Uptake24h × HalfLife
- Dose × Uptake24h × HalfLife / Thyroid weight

这些变量有明确生物学意义：治疗前疾病负荷、免疫活性、以及单位甲状腺组织实际接受的 RAI 生物有效暴露。

## 3. 主要结局是什么？

主结局建议使用：

> **24M NHRH binary**

也就是：

- 持续甲亢未控制；或
- 控制后再次复发；

均定义为 NHRH。

对应的反向临床语言可以写成：

> **24M stable control after RAI**

也就是患者和医生更容易理解的“治疗后两年内稳定控制概率”。

## 4. 推荐 ML / 统计方法

### 4.1 主模型

建议主模型使用：

> **Elastic Net Logistic Regression 或 L2-regularized Logistic Regression + probability calibration**

原因：

- 可解释，适合临床论文；
- 能处理一定共线性；
- 可以输出 OR / nomogram / risk score；
- 便于做校准和 DCA；
- 审稿人容易接受。

### 4.2 高级 ML 补充模型

建议作为 **nonlinear ML benchmark / challenge models** 加入：

- Random Forest
- ExtraTrees
- XGBoost
- LightGBM
- CatBoost

其中最推荐突出：

> **Random Forest + SHAP**

作用不是替代 LR，而是检验治疗前变量中的非线性和交互信息是否能提高预测能力。

### 4.3 解释方法

- Logistic：OR per SD、nomogram、calibration curve、DCA
- Tree-based ML：SHAP global importance、SHAP dependence、individual waterfall plot

## 5. 如何验证？

必须保持时间安全：

1. Development set 内做 repeated stratified CV 或 nested CV。
2. 所有 imputation、scaling、feature selection、SMOTE、hyperparameter tuning、calibration 都必须在训练折内完成。
3. Temporal test 只做最终一次性评估。
4. 阈值在 development OOF 中确定，不允许用 temporal test 反调。

评价指标：

- ROC-AUC
- PR-AUC
- Brier score
- Calibration intercept / slope
- Calibration curve
- Decision Curve Analysis, DCA
- Sensitivity / specificity
- PPV / NPV
- Low / intermediate / high risk groups 的实际 NHRH 发生率

## 6. 大概会得到什么结果？

当前已有结果提示：

- 0M 纯治疗前 baseline ROC-AUC 约 **0.687**；
- 0M PR-AUC 约 **0.675**；
- 说明治疗前变量已有可用信号，但不应夸大为高精度预测。

预期新增高级 ML 后可能出现三种情况：

### 情况 A：高级 ML 明显优于 LR

如果 Random Forest / XGBoost / LightGBM / CatBoost 在 temporal test 中：

- ROC-AUC 提高 ≥0.03–0.05；
- PR-AUC 同步提高；
- Brier 不变差；
- calibration 可接受；
- DCA 净获益增加；
- SHAP 解释符合临床机制；

则可以把高级 ML 升级为 Module 1 的主模型。

### 情况 B：高级 ML 只有轻度提升

这是最可能的情况。可以写成：

> 非线性 ML 模型对治疗前预测有一定增益，但治疗前信息本身限制了预测上限；治疗后早期反应信息仍是性能跃升的关键。

### 情况 C：高级 ML 不优于 LR

这也不是失败。可以写成：

> Baseline-only 预测的瓶颈主要来自治疗前信息有限，而不是算法过于简单；因此临床上需要治疗后早期复查信息进行动态更新。

## 7. 这个模块在全文中的位置

Module 1 是全文的 **患者治疗前咨询入口**。

它不追求最高 AUC，而是回答：

- 治疗前能否给出初始风险？
- 哪些 baseline 因素提示 RAI 后失败/复发风险较高？
- 是否能帮助患者形成治疗预期？

---

# Module 2：治疗后早期长期风险更新模型

## 1. 这个模块是干什么的？

这个模块回答治疗后早期随访中的问题：

> **“患者已经做完 RAI，1M/3M/6M 的甲功反应出来以后，我们能否更准确地预测这个疗程最终是否会失败或复发？”**

它不是“另一个 baseline 模型”，而是：

> **early fixed-landmark long-term risk updating model**

中文可以写成：

> **治疗后早期固定地标长期风险更新模型**

## 2. 需要用到哪些数据？

每个地标只使用截至该地标已经可见的信息。

### 2.1 0M 数据

沿用 Module 1 中的 baseline 临床、免疫、甲功、RAI 剂量学变量。

### 2.2 1M / 3M / 6M 当前甲功

- FT3 at 1M / 3M / 6M
- FT4 at 1M / 3M / 6M
- TSH at 1M / 3M / 6M
- 当前甲功状态：Hyper / Normal / Hypo
- 是否仍甲亢
- 是否已正常
- 是否进入甲减

### 2.3 早期反应轨迹

- FT3 baseline → landmark 的绝对变化
- FT4 baseline → landmark 的绝对变化
- FT3 百分比下降
- FT4 百分比下降
- FT3/FT4 ratio 变化
- 是否从 Hyper 转为 Normal / Hypo
- 当前状态与前一状态的转移

### 2.4 禁止进入主模型的数据

- post-RAI 用药情况
- 医生基于治疗后状态做出的处理决策
- 未来时间点化验值
- 24M 状态相关派生信息

post-RAI 用药可做审计或 sensitivity，但不应直接进入主模型，否则容易引入 indication bias。

## 3. 主要结局是什么？

主结局仍然是：

> **24M NHRH binary**

也就是最终两年内是否持续甲亢或控制后复发。

12M/24M Hyper / Normal / Hypo 状态预测可以作为 secondary endpoint，但不建议作为主线，因为 Normal 与 Hypo 混淆较明显，且临床意义与 NHRH 不完全相同。

## 4. 推荐 ML / 统计方法

### 4.1 主模型

建议继续保留：

> **Clinical-core L2 Logistic Regression + Platt calibration**

理由：

- 当前结果已经表现很好；
- 可解释性强；
- 3M/6M 之间可用统一 OR 解释；
- 便于校准和 DCA；
- 与 Module 1 的解释体系一致。

### 4.2 补充模型

可以在 Supplementary 中比较：

- ExtraTrees
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking ensemble，如有必要

但建议正文不把“复杂模型比较”作为主线，除非其在 temporal test 中稳定、明显优于 LR。

## 5. 如何验证？

每个 landmark 独立验证：

- 0M → 24M NHRH
- 1M → 24M NHRH
- 3M → 24M NHRH
- 6M → 24M NHRH

验证方式：

1. Development OOF 评估。
2. Temporal test 最终评估。
3. 与 persistence naive baseline 比较，即“直接沿用当前甲功状态”的朴素判断。
4. 校准曲线与 Brier score。
5. DCA。
6. 风险分层 observed event rate。
7. 混淆矩阵、FN/FP 错误分析。

## 6. 大概会得到什么结果？

当前已有结果非常适合支撑这个模块：

- 0M ROC-AUC 约 **0.687**；
- 1M ROC-AUC 约 **0.732**；
- 3M ROC-AUC 约 **0.846**，PR-AUC 约 **0.822**；
- 6M ROC-AUC 约 **0.923**，PR-AUC 约 **0.924**；
- 6M Brier 可降至约 **0.096**。

这个结果可以形成全文最重要的长期风险更新结论：

> 治疗前 baseline 已有一定预测信号，但治疗后 3M/6M 甲功反应显著提高了对两年内治疗失败/复发轨迹的识别能力。

## 7. 这个模块输出什么？

建议输出两个层次：

### 7.1 患者层面的长期风险概率

例如：

- 3M 时预测 24M NHRH 风险 15%
- 6M 时更新为 5%

或者：

- 3M 时预测 24M NHRH 风险 45%
- 6M 时仍高，提示长期治疗失败风险较高

### 7.2 Stage 2 可继承的 early risk score

Module 2 还应输出一个：

> **early NHRH risk score**

这个 score 作为 Module 3 rolling landmark 模型的一个输入，代表“早期治疗反应浓缩出的长期失败风险信号”。

## 8. 这个模块在全文中的位置

Module 2 是全文的 **长期结局动态更新核心**。

它证明：

- RAI 后早期反应不是普通随访记录，而是强预测信号；
- 3M/6M 是长期风险更新的关键节点；
- 早期风险评分可以传递给后续 rolling monitoring。

---

# Module 3：随访期滚动复发监测模型

## 1. 这个模块是干什么的？

这个模块回答长期随访中的问题：

> **“患者现在复查到这个时间点了，下一阶段会不会再次甲亢或生化复发？”**

它才是全文真正意义上的：

> **rolling landmark dynamic relapse monitoring model**

中文可以写成：

> **随访期滚动地标动态复发监测模型**

## 2. 需要用到哪些数据？

分析单位可以是 **随访行 / landmark visit**，同时最终聚合到 **治疗级 / patient-level**。

每个随访时点，例如 3M、6M、12M、18M，使用以下信息：

### 2.1 静态 baseline 特征

- 年龄
- 性别
- baseline FT3 / FT4 / TSH
- baseline TRAb / antibodies
- thyroid weight
- RAI dose / uptake / exposure features

### 2.2 当前随访化验

- 当前 FT3
- 当前 FT4
- 当前 TSH
- 当前甲功状态：Hyper / Normal / Hypo
- 当前是否甲亢
- 当前是否正常或甲减

### 2.3 累积病程轨迹

- 历史是否曾甲亢
- 历史是否曾正常
- 历史是否曾甲减
- 上一时间点状态
- 当前与上一时间点的状态转移
- FT3 / FT4 / TSH 的累计变化
- 甲功是否反弹
- 甲功是否持续下降

### 2.4 Module 2 继承风险

- 3M 或 6M early NHRH risk score
- 或当前时点最近可用 early fixed-landmark risk score

这个变量代表：

> 早期治疗反应已经浓缩出的长期失败/复发风险背景。

## 3. 主要结局是什么？

主结局建议使用：

> **H1：下一随访窗口甲亢事件**

对于当前已经正常的患者，可以解释为：

> **下一窗口生化复发风险**

补充端点：

- H6：当前时间点后 6 个月内甲亢事件
- H12：当前时间点后 12 个月内甲亢事件

不建议把“下个阶段三分类状态 Hyper / Normal / Hypo”作为主终点，因为这会明显增加终点数量，而且 Normal / Hypo 的临床界限和统计稳定性都较弱。

## 4. 推荐 ML / 统计方法

### 4.1 主模型

建议主模型使用：

> **Clinical L2 Logistic Regression + probability calibration**

理由：

- 现有 Stage 2 结果显示 LR 与 ExtraTrees 等模型判别力相近；
- LR 概率校准更好；
- 可以生成 nomogram-style points；
- 更利于临床落地；
- 单点 H1 预测更适合作为风险分层工具，而不是黑箱高精度报警器。

### 4.2 模型选择指标

可沿用现有设计：

```text
SelectionScore = 0.6 × H1_PR_AUC + 0.4 × H12_PR_AUC
```

H6 作为 sensitivity endpoint，不参与模型选择。

### 4.3 候选模型

可以比较：

- Clinical L2 Logistic
- Elastic Net Logistic
- Random Forest
- ExtraTrees
- XGBoost
- LightGBM
- CatBoost

但正文建议保留 LR 为主模型，复杂模型放 supplement，除非复杂模型在 temporal test、校准和 DCA 上都明显更好。

### 4.4 解释方法

- LR coefficient / nomogram-style point system
- SHAP 或 permutation importance
- 局部解释：解释某个患者某次复查为什么被判为高风险

## 5. 如何验证？

### 5.1 随访行级验证

对每个 landmark row 预测：

- H1 ROC-AUC
- H1 PR-AUC
- H1 Brier
- H1 sensitivity / specificity / PPV / NPV
- H6 / H12 同步报告

### 5.2 治疗级聚合验证

将同一疗程的多次 H1 risk 按时间加权聚合为：

> **treatment-level dynamic risk score**

然后验证：

- Q1–Q4 风险分层 observed event rate
- Kaplan-Meier 曲线
- log-rank test
- Cox / Harrell C-index
- Top-k capture / lift
- patient-level sensitivity analysis，即每个患者只计一次，排除重复随访行造成的假象

### 5.3 Ablation analysis，强烈建议补充

为了证明 Module 3 不是重复 Module 2，建议做 ablation：

| 模型 | 输入 | 目的 |
|---|---|---|
| A | 只用 Module 2 early risk score | 代表早期风险评分本身 |
| B | 只用当前随访甲功 | 代表医生当前肉眼判断 |
| C | 当前甲功 + 累积轨迹 | 代表普通 rolling landmark |
| D | 当前甲功 + 累积轨迹 + early risk score | 完整 Module 3 |

如果 D 优于 A，说明 rolling monitoring 不是重复早期风险评分。

如果 D 优于 B/C，说明 early risk score 对后续监测有增量。

如果 D 仅轻度优于 B/C，也可以说明当前甲功已经吸收了大量风险信息，但 early score 仍提供历史风险背景。

## 6. 大概会得到什么结果？

当前 Stage 2 结果已经比较适合支撑 Module 3：

### 6.1 行级 H1/H6/H12 预测

Temporal test 中：

- H1 ROC-AUC 约 **0.826**；
- H1 PR-AUC 约 **0.359**；
- H1 Brier 约 **0.052**；
- H6 ROC-AUC 约 **0.796**；
- H6 PR-AUC 约 **0.358**；
- H12 ROC-AUC 约 **0.771**；
- H12 PR-AUC 约 **0.389**。

行级预测的临床定位应谨慎：

> 单点预测判别力为中等，更适合风险分层和低危 rule-out，不宜夸大为高精度复发报警器。

### 6.2 治疗级风险分层

治疗级聚合结果更有临床价值：

- temporal test Q1 事件率约 **7.1%**；
- temporal test Q4 事件率约 **52.4%**；
- Cox C-index 约 **0.82**，高于仅用早期风险的传统基线；
- patient-level sensitivity analysis 仍呈单调梯度，例如 Q1 约 **2.5%**，Q4 约 **48.7%**，Harrell C-index 约 **0.83**。

这部分应作为 Module 3 的核心临床输出。

## 7. 这个模块在全文中的位置

Module 3 是全文的 **随访管理落脚点**。

它不只是输出一次预测，而是将多次复查风险整合为：

- 低危：可常规随访，甚至适当放宽复查频率；
- 中危：维持标准随访；
- 高危：加强随访、提前预警、必要时讨论二次治疗或替代方案。

---

# 三个模块之间的关系

三个模块不是重复，而是递进关系。

```text
Module 1：治疗前结局预期
    输入：0M baseline
    输出：pre-RAI 24M NHRH risk
    临床用途：治疗前咨询和心理预期

Module 2：早期长期风险更新
    输入：0M + 1M/3M/6M 早期反应
    输出：updated 24M NHRH risk + early risk score
    临床用途：治疗后早期识别长期失败/复发高危者

Module 3：滚动复发监测
    输入：baseline + 当前甲功 + 累积轨迹 + early risk score
    输出：下一窗口甲亢风险 + 治疗级动态风险分层
    临床用途：差异化随访管理
```

最重要的是：

> **Module 2 的 early risk score 可以作为 Module 3 的输入，因此 Module 2 不是与 Module 3 重复，而是 Module 3 的上游风险浓缩器。**

---

# 主文与补充材料如何分配

## 主文建议保留

### Figure 1：总研究框架图

展示：

```text
0M baseline expectation → 1M/3M/6M long-term update → rolling next-window monitoring → treatment-level stratification
```

### Table 1：队列与数据结构

包括：

- 分析单位：RAI treatment course
- 总治疗起点数
- development / temporal test 切分
- 0M/1M/3M/6M/12M/18M/24M 随访结构
- 主要 endpoint 定义

### Table 2：Module 1 baseline-only 模型表现

至少报告：

- ROC-AUC
- PR-AUC
- Brier
- calibration
- DCA
- risk groups observed event rate

### Table 3：Module 2 早期长期风险更新表现

保留：

- 0M / 1M / 3M / 6M → 24M NHRH
- ROC-AUC / PR-AUC / Brier
- 与 persistence baseline 对照

### Table 4：Module 3 rolling monitoring 表现

保留：

- H1 / H6 / H12 ROC-AUC、PR-AUC、Brier
- H1 作为主端点

### Figure 2：长期风险预测性能随时间递增图

展示 0M → 1M → 3M → 6M 的 AUC / PR-AUC / Brier 改善。

### Figure 3：治疗级风险分层图

展示：

- Q1–Q4 observed event rate
- Kaplan-Meier curve
- Cox / C-index
- patient-level sensitivity

### Figure 4：解释图

建议包括：

- Module 1：baseline SHAP 或 OR forest
- Module 2：早期反应 OR forest / LinearSHAP
- Module 3：nomogram-style points 或 SHAP

## Supplementary 建议放

- 全部模型族比较
- Random Forest / XGBoost / LightGBM / CatBoost 详细结果
- 完整 hyperparameter table
- 完整 calibration plots
- 12M/24M Hyper / Normal / Hypo 三分类
- persistent vs recurrence subtype
- threshold sensitivity
- error analysis
- leakage audit
- medication audit
- ablation analysis 完整表
- bootstrap 95% CI

---

# 全文最终基调

## 1. 文章不是“算法竞赛”

不要把文章写成：

> 我们比较了很多机器学习模型，某个模型 AUC 最高。

应写成：

> 我们构建了一个时间安全、可解释、校准良好的双时间尺度风险预测框架，用于 RAI 治疗前结局预期、治疗后早期长期风险更新和随访期动态复发监测。

复杂机器学习模型的作用是：

- 验证 nonlinear interactions 是否存在增益；
- 作为 performance benchmark；
- 用 SHAP 增强机制解释；
- 不应掩盖主线的临床逻辑。

## 2. 文章不是“绝对预测复发”

不建议写成：

> 该模型可以准确预测复发。

更稳妥的写法是：

> 该框架能够在不同临床阶段提供校准良好的个体化风险概率，并通过治疗级聚合将患者分为明显不同的复发风险层级。

尤其 Module 3 应强调：

- H1 单点预测判别力中等；
- 但 NPV 高，适合 rule-out；
- 治疗级风险分层更稳定、更有临床价值。

## 3. 文章不是“治疗因果效应预测”

如果没有 RAI 与其他治疗方式的对照组，不要强行称为 treatment benefit prediction。

建议写成：

- pre-RAI outcome expectation
- expected 24-month treatment outcome after RAI
- individualized probability of stable control after RAI

中文写成：

> 治疗前 RAI 结局预期评估，而非严格意义上的 RAI 净获益因果预测。

## 4. 文章的最强创新点

最强创新点不是某个单一模型的 AUC，而是：

> **把 RAI 治疗过程拆成临床真实发生的三个决策时刻：治疗前咨询、早期疗效更新、长期随访监测，并为每个时刻提供对应的个体化风险概率。**

---

# 推荐标题

## 中文标题

### 版本 1，最推荐

**Graves 甲亢碘-131治疗后治疗失败与复发的治疗前结局预期、早期风险更新和滚动随访监测：一项双时间尺度 landmark 机器学习研究**

### 版本 2，更简洁

**Graves 甲亢碘-131治疗后失败/复发风险的双时间尺度动态预测模型**

### 版本 3，更临床

**面向碘-131治疗决策与随访管理的 Graves 甲亢失败/复发个体化风险预测模型**

## 英文标题

### Version 1，最推荐

**A Dual-Horizon Landmark Prediction Framework for Pre-Treatment Outcome Expectation, Early Risk Updating, and Rolling Relapse Monitoring after Radioiodine Therapy in Graves’ Hyperthyroidism**

### Version 2，更短

**Dynamic Landmark Prediction of Treatment Failure and Relapse after Radioiodine Therapy in Graves’ Hyperthyroidism**

### Version 3，更强调临床工具

**An Interpretable Machine-Learning Framework for Individualized Outcome Expectation and Follow-up Risk Stratification after Radioiodine Therapy in Graves’ Hyperthyroidism**

---

# 推荐摘要核心表述

## Background

Graves 甲亢患者接受 RAI 后仍存在持续甲亢、控制后复发或甲减等不同治疗轨迹。现有预测多依赖治疗前或单一随访时点，缺乏覆盖治疗前咨询、早期反应更新和长期随访监测的动态定量工具。

## Methods

本研究构建双时间尺度 landmark 预测框架。Module 1 使用 0M 治疗前 baseline 信息预测 24M NHRH 风险，用于 RAI 后结局预期评估。Module 2 在 1M/3M/6M 固定地标加入早期甲功反应，更新长期 NHRH 风险并生成 early risk score。Module 3 采用 rolling landmark 设计，在 3M/6M/12M/18M 每次随访时结合 baseline、当前甲功、累积轨迹和 early risk score，预测下一窗口甲亢事件，并聚合为治疗级风险分层。所有模型选择、特征处理、阈值确定和校准均在 development set 内完成，temporal test 仅用于最终验证。

## Results

治疗前 baseline-only 模型已显示可用预测信号，但性能有限。加入早期反应后，长期 NHRH 预测性能从 0M 到 6M 明显提升。rolling landmark 模型在下一窗口甲亢事件预测中具有中等判别力，但治疗级聚合风险分层能清晰区分高低危患者，并支持差异化随访管理。

## Conclusion

本框架将 RAI 治疗后的风险评估从一次性预测扩展为连续临床决策路径：治疗前用于结局预期，治疗后早期用于长期失败风险更新，随访期用于动态复发监测。该框架更适合风险分层和低危排查，而非单次高精度报警。

---

# 最终建议执行顺序

## Step 1：先固定 endpoint

主 endpoint：

```text
24M NHRH binary：持续甲亢或控制后复发
```

rolling endpoint：

```text
H1：下一随访窗口甲亢事件
H6 / H12：敏感性端点
```

secondary endpoint：

```text
12M / 24M Hyper / Normal / Hypo
Persistent vs Recurrence subtype
```

## Step 2：把 0M baseline-only 单独拿出来做 Module 1

这是治疗前患者沟通模型。

## Step 3：把 1M/3M/6M 固定地标作为 Module 2

这是长期结局风险更新模型，不要再泛称为 Stage 2 dynamic prediction。

## Step 4：把 rolling landmark 作为 Module 3

这是随访期真正的动态监测模型。

## Step 5：补 Baseline-only 高级 ML benchmark

重点做：

- Random Forest + SHAP
- XGBoost
- LightGBM
- CatBoost
- ExtraTrees

但只有在 temporal test、calibration、DCA 均优于 LR 时，才升级为主模型。

## Step 6：补 ablation analysis

特别是 Module 3：

```text
current labs only
current labs + trajectory
early score only
current labs + trajectory + early score
```

用来证明 rolling monitoring 不是重复早期风险模型。

## Step 7：所有结果按 TRIPOD-AI / PROBAST-AI 思路整理

重点强调：

- time-safety
- leakage control
- calibration
- DCA
- temporal validation
- interpretability
- clinical utility

---

# 给合作者的简短版本

我们建议把全文定为 **三模块、双时间尺度 landmark 风险预测框架**。

**Module 1 是 0M baseline-only 模型**，只用治疗前临床、甲功、抗体、甲状腺负荷和 RAI 剂量学信息，预测 24M NHRH / stable control，用于患者治疗前 RAI 结局预期评估。主模型建议用 elastic-net / L2 logistic，新增 Random Forest、XGBoost、LightGBM、CatBoost 作为高级 ML benchmark，并用 SHAP 解释。

**Module 2 是 1M/3M/6M early fixed-landmark long-term update**，在 baseline 基础上加入早期甲功反应和状态转移，继续预测 24M NHRH。当前结果显示 AUC 从 0M 约 0.69 升至 3M 约 0.85、6M 约 0.92，说明早期治疗反应显著提高长期失败/复发风险识别。该模块还输出 early NHRH risk score，作为后续 rolling monitoring 的输入。

**Module 3 是 rolling landmark relapse monitoring**，在 3M/6M/12M/18M 每次随访时，结合 baseline、当前甲功、累积轨迹和 early risk score，预测下一窗口甲亢事件 H1，并以 H6/H12 为敏感性端点。当前行级 H1 AUC 约 0.83，单点预测为中等判别力；但治疗级聚合后可形成清晰 Q1–Q4 风险梯度，高危组事件率远高于低危组，适合用于差异化随访管理。

全文最终基调应是：

> **这不是一个单纯追求 AUC 的机器学习模型，而是一个时间安全、可解释、概率校准、覆盖治疗前咨询—早期风险更新—长期随访监测的 RAI 个体化决策支持框架。**
