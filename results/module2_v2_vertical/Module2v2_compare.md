# M2 · v2 — EBM 与各类机器学习方法对比

> 在**同一聚合轴特征集**上、**逐地标(0/1/3/6M)**拟合 **12 种方法**(含 naive 基线),时间外测试比较 ROC-AUC(含 95% bootstrap CI)/ PR-AUC / Brier / 校准斜率 / 拟合耗时;多粒度对比图(热力图 / 逐月条形 / ROC 叠加 / 校准叠加 / 决策曲线 / 排名 / ΔvsEBM / 成本-收益)。

## 排名(4 地标平均 ROC-AUC)

| 名次 | 方法 | 平均 ROC-AUC |
|:--:|:--|:--:|
| 1 | L2-Logistic | 0.761 |
| 2 | Elastic-net | 0.760 |
| 3 | **EBM** | 0.750 |
| 4 | RandomForest | 0.742 |
| 5 | ExtraTrees | 0.734 |
| 6 | SVM-RBF | 0.734 |
| 7 | XGBoost | 0.724 |
| 8 | GaussianNB | 0.718 |
| 9 | LightGBM | 0.710 |
| 10 | HistGBM | 0.702 |
| 11 | kNN(15) | 0.701 |
| 12 | Naive(prevalence) | 0.500 |

## 关键结论

- **简约线性模型(L2-Logistic / Elastic-net ≈ 0.76)居首**;**EBM 第 3(0.750)**,略低于线性但**唯一兼顾玻璃盒可解释**;**boosting(XGBoost/LightGBM/HistGBM)、树(RF/ExtraTrees)、kNN 反而更低**。

- **Naive(prevalence)= 0.500 地板**(CLAUDE.md 强制基线):所有学习方法都显著高于盲猜,但彼此差距不大——再次指向**特征信息天花板**(小样本表格上,复杂度不回本,简约/可解释模型已逼近上限)。

- **EBM 的定位 = 准确率-可解释性的最佳折中**:在与最优仅差 ~0.01 AUC 的同时,提供逐特征非线性形状与原生重要性(见 [EBM 玻璃盒论文](Module2v2_EBM_paper.md) / [EBM 图集](Module2v2_EBM_图集.md))。

- boosting 在 ~800 样本上**轻度过拟合**(HistGBM/LightGBM 偏低);线性/加性模型的强归纳偏置在此更稳。


## 热力图(方法 × 地标 × metric)

**C01. ROC-AUC: 方法 × 地标**

![C1](m2v2_compare/figures/C01_Heatmap_ROC.png)

**C02. PR-AUC: 方法 × 地标**

![C2](m2v2_compare/figures/C02_Heatmap_PR.png)

**C03. Brier(越低越好): 方法 × 地标**

![C3](m2v2_compare/figures/C03_Heatmap_Brier.png)

**C04. 校准斜率(理想 1): 方法 × 地标**

![C4](m2v2_compare/figures/C04_Heatmap_CalibSlope.png)


## 逐月 ROC-AUC 条形(含 95% CI)

**C05. 0M 各方法 ROC-AUC(含95%CI)**

![C5](m2v2_compare/figures/C05_Bar_ROC_0M.png)

**C06. 1M 各方法 ROC-AUC(含95%CI)**

![C6](m2v2_compare/figures/C06_Bar_ROC_1M.png)

**C07. 3M 各方法 ROC-AUC(含95%CI)**

![C7](m2v2_compare/figures/C07_Bar_ROC_3M.png)

**C08. 6M 各方法 ROC-AUC(含95%CI)**

![C8](m2v2_compare/figures/C08_Bar_ROC_6M.png)


## 逐月 ROC 曲线叠加

**C09. 0M 各方法 ROC 叠加**

![C9](m2v2_compare/figures/C09_ROCoverlay_0M.png)

**C10. 1M 各方法 ROC 叠加**

![C10](m2v2_compare/figures/C10_ROCoverlay_1M.png)

**C11. 3M 各方法 ROC 叠加**

![C11](m2v2_compare/figures/C11_ROCoverlay_3M.png)

**C12. 6M 各方法 ROC 叠加**

![C12](m2v2_compare/figures/C12_ROCoverlay_6M.png)


## 逐月校准对比(代表方法)

**C13. 0M 代表方法校准对比**

![C13](m2v2_compare/figures/C13_Caliboverlay_0M.png)

**C14. 1M 代表方法校准对比**

![C14](m2v2_compare/figures/C14_Caliboverlay_1M.png)

**C15. 3M 代表方法校准对比**

![C15](m2v2_compare/figures/C15_Caliboverlay_3M.png)

**C16. 6M 代表方法校准对比**

![C16](m2v2_compare/figures/C16_Caliboverlay_6M.png)


## 总排名与相对 EBM 的差距

**C17. 各方法 4 地标平均 ROC-AUC 排名**

![C17](m2v2_compare/figures/C17_Ranking_meanAUC.png)

**C18. 各方法相对 EBM 的平均 ΔROC-AUC**

![C18](m2v2_compare/figures/C18_Delta_vs_EBM.png)


## 逐月决策曲线(DCA)对比

**C19. 0M 代表方法决策曲线(DCA)对比**

![C19](m2v2_compare/figures/C19_DCA_0M.png)

**C20. 1M 代表方法决策曲线(DCA)对比**

![C20](m2v2_compare/figures/C20_DCA_1M.png)

**C21. 3M 代表方法决策曲线(DCA)对比**

![C21](m2v2_compare/figures/C21_DCA_3M.png)

**C22. 6M 代表方法决策曲线(DCA)对比**

![C22](m2v2_compare/figures/C22_DCA_6M.png)


## 性能 vs 计算成本

**C23. 性能 vs 拟合耗时(成本-收益)**

![C23](m2v2_compare/figures/C23_Cost_vs_Perf.png)


---
*环境:本机 uv(torch2.12 / scikit-learn / interpret-core / xgboost3.2 / lightgbm4.6);数据自计算机 rsync;分析单元 = 1003 人次;完整指标见 `m2v2_compare/metrics_table.csv`。*