# Baseline-only 治疗前高级机器学习补充分析方案

**项目方向**：Graves 甲亢碘-131（RAI）治疗失败 / 复发风险预测
**新增模块**：0M Baseline-only 非线性机器学习模型比较与解释
**建议定位**：主线保持“0M 治疗前获益评估 + 1M/3M/6M 动态更新”，Baseline-only ML 作为治疗前预测增强模块
**版本**：v1.0，供合作者讨论
**日期**：2026-05-25

> 注：本文件由微信 RWTemp 临时目录的原件重建落盘（原临时路径已被清理），内容依据收到的全文。

---

## 1. 当前项目基础

目前 Stage1 已经完成 0M、1M、3M、6M 四个 fixed-landmark 风险预测模型。0M 是纯治疗前 baseline 模型；1M、3M、6M 是治疗后早期动态更新模型。每个地标只使用截至该时间点已经可见的信息，预测 24M 内 NHRH 复合治疗失败终点。

NHRH binary 定义为：随访至 24M 期间出现持续甲亢未控制或控制后复发，即视为治疗失败 / 复发复合终点。12M/24M Hyper / Normal / Hypo 状态预测目前作为补充终点。

当前主线模型为可解释 logistic regression。3M/6M 主报固定为 `Clinical_L2_Logistic + clinical_core + nhrh_mild + Platt`。0M/1M 采用 OOF 最优的可解释 LR 配置。已有结果：0M ROC-AUC ≈ 0.687 / PR-AUC ≈ 0.675；1M ≈ 0.732；3M ≈ 0.846；6M ≈ 0.923 / PR-AUC ≈ 0.924。

说明：治疗前 baseline 信息可以提供初始风险排序，但 3M/6M 治疗反应信息才是更强的预测信号。本新增分析不应试图用 Baseline-only 模型替代动态 landmark 模型，而应作为“治疗前患者获益沟通和初始风险分层”的增强模块。

## 2. 为什么要增加 Baseline-only 高级机器学习模型？

- **临床意义**：只用治疗前/ index RAI 信息，回答“做碘-131后两年内稳定控制的可能性有多大、是否属易失败/复发人群”，用于治疗前咨询、预期管理、高危者提前加密随访、与 1M/3M/6M 动态链衔接。
- **方法学意义**：治疗前变量间可能存在非线性/交互（Dose×Uptake×HalfLife×ThyroidW、TRAb×ThyroidW、FT3/FT4、每克剂量、RAIU3h/24h 等）；普通 LR 不手工加交互很难捕捉，RF/ExtraTrees/XGBoost/LightGBM/CatBoost 可自动捕捉，作为挑战模型。
- **与主线关系**：主模型 = 可解释 LR / Elastic Net；补充 = 非线性 ML challenge；解释 = RF/GBM + SHAP。ML 明显更优才升主推；否则保持 LR 主线、ML 作补充，说明瓶颈是治疗前信息有限而非算法。

## 3. 核心研究问题

- Primary：仅用 0M 信息时，非线性 ML 是否较 baseline LR 提高 24M NHRH 预测？
- Secondary：ML 是否提高 ROC-AUC/PR-AUC/Brier/calibration？DCA 净获益是否更高？高低风险组事件率分离是否更好？SHAP 关键变量是否符合机制？增益是否在 temporal test/外部验证保持？

## 4. Endpoint 设计

- 主终点：**24M NHRH binary**（persistent non-healing 或 recurrence after initial control）。
- 次要：persistent non-healing；recurrence among initial responders；Success/Persistent/Recurrence 三分类；12M/24M 状态（探索性，不作 baseline-only 主线）。

## 5. 特征范围（Baseline-only 只能用 0M / index RAI 可见信息）

- 可用 A 人口学：Age、Sex、BMI、Disease duration、Smoking(if)、Graves orbitopathy(if)、TreatCount(if)。
- 可用 B RAI 前甲功/免疫：FT3 0M、FT4 0M、TSH 0M、TRAb/TBII 0M、TPOAb(if)、TgAb(if)；偏态变量考虑 log1p / winsorize。
- 可用 C 甲状腺负荷：thyroid weight、volume(if)、goiter grade(if)、nodule(if)。
- 可用 D RAI 剂量学：dose、RAIU3h、RAIU24h、peak uptake(if)、effective half-life(if)、dose per gram。
- 可用 E 生物有效暴露派生：Dose×RAIU24h、Dose×HalfLife、Dose×RAIU24h×HalfLife、…/thyroid weight、Dose/ThyroidW、RAIU3h/24h、FT3/FT4、ThyroidW×FT3、ThyroidW×FT4、ThyroidW×TRAb。需写入 feature dictionary（公式/单位/缺失处理）。
- 强烈建议补充（目前缺）：Pre-RAI ATD use / type / duration / withdrawal interval / prior ATD failure / disease duration>2y。补齐后作增强版 + sensitivity。
- **严禁进入**：1M/3M/6M FT3/FT4/TSH、RAI 后下降/反应变量、landmark 当前状态、post-RAI 用药、任何 outcome 派生或 baseline 之后信息。post-RAI 用药仅可审计/描述（indication bias、反向因果）。

## 6. 候选模型

- 参考：A0 当前 baseline LR。
- 稳健统计学习：A1 L2-LR；A2 Elastic Net LR；A3 LR + 限制性立方样条(可选)。
- 非线性 ML：B1 RF；B2 ExtraTrees；B3 XGBoost；B4 LightGBM；B5 CatBoost。**主推 RF + SHAP**。
- 可选：C1 Stacking；C2 Super learner（仅补充，除非 temporal/外部明显更优且校准好）。
- 不主推：DNN / TabTransformer / AutoML 黑箱（N≈1003 表格数据不适合，过拟合+解释难）。

## 7. 调参范围（仅 development 内调参，禁止用 temporal test）

- Elastic Net：C[0.01,0.03,0.1,0.3,1,3,10]、l1_ratio[0.05,0.2,0.5,0.8]、class_weight[None,balanced]、saga、max_iter5000。
- RF：n_estimators[500,1000]、max_depth[3,4,5,None]、min_samples_leaf[5,10,20]、max_features[sqrt,log2,0.5]、class_weight[None,balanced,balanced_subsample]、bootstrap True。
- ExtraTrees：同 RF 风格。
- XGBoost：n_estimators[200,500,800]、lr[0.01,0.03,0.05]、max_depth[2,3,4]、min_child_weight[3,5,10]、subsample[0.7,0.85,1]、colsample_bytree[0.7,0.85,1]、reg_lambda[1,3,10]、reg_alpha[0,0.1,1]、scale_pos_weight=不平衡比、binary:logistic。
- LightGBM：n_estimators[200,500,800]、lr[0.01,0.03,0.05]、num_leaves[7,15,31]、max_depth[2,3,4,5]、min_child_samples[20,50]、subsample/colsample[0.7,0.85,1]、reg_lambda[1,3,10]、class_weight[None,balanced]。
- CatBoost：iterations[500,1000,1500]、lr[0.01,0.03,0.05]、depth[2,3,4,5]、l2_leaf_reg[1,3,10]、auto_class_weights[None,Balanced]、Logloss、eval AUC。

## 8. 预处理（必须放进 pipeline，在每个训练折内完成）

- 缺失：连续 median、类别 most_frequent/unknown、临床有意义缺失加 missing indicator（TRAb/RAIU/half-life 等保留指示）。
- 偏态/异常：FT3/FT4/TRAb/ThyroidW/RAIU 可 log1p；训练折内 winsorize 1%/99%。
- 标准化：LR/EN/SVM 必需；树模型不需。
- 编码：LR/RF/ET/XGB/LGBM one-hot；CatBoost 可原生类别。
- 不平衡：优先 class_weight='balanced' / scale_pos_weight / BalancedRF；SMOTE 不主推，若用仅训练折内。

## 9. 训练与验证

- 切分：development（特征处理/训练/调参/阈值/校准/选模）/ temporal test（仅一次性最终评估）/ external(if)。temporal test 不参与任何 selection/tuning/threshold/calibration/cutoff。
- Dev 内部：方案 A nested CV（外 5×5 repeated stratified 出 OOF，内 3-fold 调参，最规范）；方案 B repeated stratified 5-fold×30 seeds（记 mean±SD，效率高）。算力够用 A，否则 B 并写明。
- 校准：比较 none / Platt / isotonic；样本中等主报 Platt（isotonic 易过拟合）；校准器仅 dev 内拟合。
- 阈值：不用默认 0.5，在 dev OOF 预先定（Youden / 高敏感 sens≥0.85 / DCA 10%–40% 临床区间），锁定后用于 temporal test。

## 10. 评价指标

- 判别：ROC-AUC、PR-AUC、Sens、Spec、PPV、NPV、Balanced Acc、F1（PR-AUC 对不平衡敏感，重点）。
- 概率：Brier、calibration intercept/slope、ECE(可选)、calibration curve（AUC 升但 Brier/校准明显变差不建议升主模型）。
- 临床效用：DCA、净获益、treat-all/none 对照（重点 10%–40% 或 10%–50%）。
- 风险分层：dev 预设 Low<15% / Int 15%–35% / High>35%（或按 dev OOF+DCA 定，定后不改）；temporal test 报每组人数、实际 NHRH 率、High/Low ratio、High-risk sens、Low-risk NPV、KM-like(if 时间事件可得)。

## 11. 模型比较

- 与 LR 比：ΔROC-AUC、ΔPR-AUC、ΔBrier、ΔNet benefit。
- 统计检验：ROC-AUC DeLong；PR-AUC/Brier/DCA/校准 bootstrap 95% CI（1000–2000 resample；同患者多疗程用 patient-level cluster bootstrap）。
- 升主模型标准（须同时满足）：temporal test ROC-AUC 较 LR +≥0.03–0.05；PR-AUC 同步升；Brier 不变差最好降；calibration slope≈1 曲线可接受；DCA 临床区间净获益更高；风险分层事件率分离更明显；SHAP 符合机制；repeated CV/bootstrap/temporal 稳定。只在 dev OOF 提升、temporal 不提升则不升级。

## 12. 模型解释

- LR：OR per SD、95% CI、P(可选)、OR forest（继续作医学解释主层）。
- ML：SHAP global/beeswarm/dependence/individual waterfall；permutation importance(可选)、PDP/ALE(可选)。
- 重点变量是否进前列：thyroid weight/volume、TRAb/TBII、FT3/FT4/TSH 0M、RAIU3h/24h、half-life、dose、dose per gram、Dose×RAIU24h×HalfLife/ThyroidW、disease duration、pre-RAI ATD(if)。若不合临床逻辑变量极重要，查 leakage/编码/缺失伪影/分布漂移/outcome proxy。

## 13. 敏感性分析

- 特征集：FS1 clinical-core baseline；FS2 +RAI 剂量/摄取/负荷；FS3 +派生暴露；FS4 +pre-RAI ATD(if 补齐)。
- 缺失：median+indicator / 多重插补(可选) / 完整病例(可选)。
- 阈值：Youden / 高敏感 / DCA。
- 患者重复治疗：patient-level grouped split / cluster bootstrap / 仅首次 RAI 分析。
- 终点：NHRH 复合 / persistent only / recurrence among responders / 三分类 subtype。

## 14. 结果输出模板

- Table 1 候选模型（model/feature set/calibration/purpose/explainability）。
- Table 2 Dev OOF 表现（ROC-AUC/PR-AUC/Brier/Cal slope/Sens/Spec/Balanced Acc）。
- Table 3 Temporal test 表现（含 DCA better than LR? / Upgrade candidate?）。
- Table 4 风险组表现（组/N/mean pred/observed NHRH/sens 贡献/NPV/PPV）。
- Figures：F1 设计与防泄漏流程；F2 ROC/PR LR vs ML；F3 校准 LR vs best ML；F4 DCA；F5 风险分布+错分；F6 风险组事件率；F7 SHAP beeswarm；F8 SHAP dependence top4–6；F9 个体 SHAP waterfall 高/低危；S1 调参汇总；S2 bootstrap ΔAUC/ΔBrier/ΔPR-AUC forest。

## 15. 防泄漏执行清单

只用 0M baseline；删 1M/3M/6M labs、early decline/response、landmark 当前状态、post-RAI 用药、outcome 派生标签、12M/24M 反推特征；imputation/scaling/feature selection/SMOTE 仅训练折；calibration 仅 dev；threshold 仅 dev OOF；temporal test 不参与选模；记录包版本与随机种子。

## 16. 推荐代码框架

```
baseline_ml/
  config/{baseline_feature_set_v1.yaml, model_grid.yaml, split_config.yaml}
  data/baseline_dataset_frozen.csv
  scripts/{01_build_baseline_features.py, 02_train_nested_cv.py, 03_calibrate_models.py,
           04_evaluate_temporal_test.py, 05_shap_explain.py, 06_generate_tables_figures.py}
  outputs/{tables, figures, models, logs}
```
feature set yaml 显式列 baseline_allowed_features 与 forbidden_features；训练伪代码：外层 CV 出 OOF→dev OOF 拟合校准→dev OOF 选模→full dev refit→dev 校准→temporal test 仅评估一次。

## 17. 预期结果与口径

- ML 明显提高：写“治疗前非线性 ML 较可解释 LR 进一步提高 24M NHRH 预测，提示甲状腺负荷/免疫活性/RAI 生物有效暴露的非线性组合含额外预后信息”。
- ML 提升有限（最可能）：写“非线性 ML 仅有限增益，说明治疗前预测瓶颈是信息本身有限而非算法；3M/6M 动态模型因纳入早期反应仍更强”。
- 避免写“复杂模型一定更好 / baseline-only 可替代动态 / accuracy 提高即更好”。

## 18. 论文新增小节

Methods: Baseline-only nonlinear machine-learning benchmark；Results: Incremental value of nonlinear baseline-only models；Discussion: Clinical role of the pre-treatment model（定位为治疗前咨询工具，非替代治疗后动态更新）。

## 19. 报告规范

遵循 TRIPOD+AI、PROBAST+AI；补 study design/data source、eligibility、outcome、predictors+timing、missing data、development、internal+temporal validation、calibration、clinical utility、interpretation、reproducibility/versions、risk of bias。

## 20. 最终建议

作为增强模块而非替代。最佳结构：0M baseline-only 咨询模型 → baseline-only ML benchmark + SHAP → 1M/3M/6M 动态更新 → calibration+DCA+风险分层 → 错误分析与 subtype 探索。理想结论：baseline-only ML 适度提升治疗前分层，3M/6M 动态模型因纳入早期反应仍最强。

## 21. 参考文献建议

1. Collins GS, et al. TRIPOD+AI statement. BMJ. 2024;385:e078378.
2. Moons KGM, et al. PROBAST+AI. BMJ. 2025;388:e082505.
3. Lu L, Wei X, Chen Y, et al. From data to decision: an interpretable machine learning model for optimizing RAI therapy in Graves' hyperthyroidism. Front Endocrinol. 2026;16. DOI:10.3389/fendo.2025.1711029.
