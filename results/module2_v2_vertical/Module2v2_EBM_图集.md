# M2 · v2 — EBM 玻璃盒全套图谱(47 图)

> 用**统一的 EBM(Explainable Boosting Machine)玻璃盒方法**重制 M2 的整套分析图(≈47 张):每个地标(0/1/3/6M)给出 ROC / PR / 校准 / 原生重要性 / 形状函数 / 决策曲线(DCA)/ 风险三档 / 混淆矩阵,外加跨地标汇总(重要性漂移、EBM vs LR、多 seed 稳定性、共线性、选择性预测、工作点等)。所有图自包含。配套论文见 [EBM 玻璃盒论文](Module2v2_EBM_paper.md);完整诊断见 [probe 报告](Module2v2_probe_可解释性与诊断.md)。


## 0M — EBM 全套(9 图)

**F01. 0M EBM ROC,AUC 0.671**

![F1](m2v2_ebm_full/figures/F01_ROC_0M.png)

**F02. 0M EBM precision-recall,AP 0.640**

![F2](m2v2_ebm_full/figures/F02_PR_0M.png)

**F03. 0M EBM reliability**

![F3](m2v2_ebm_full/figures/F03_Calib_0M.png)

**F04. 0M EBM 原生重要性 top-6**

![F4](m2v2_ebm_full/figures/F04_Importance_0M.png)

**F05. 0M EBM 形状函数 #1:Thyroid weight**

![F5](m2v2_ebm_full/figures/F05_Shape_0M_r1.png)

**F06. 0M EBM 形状函数 #2:TPOAb**

![F6](m2v2_ebm_full/figures/F06_Shape_0M_r2.png)

**F07. 0M EBM 决策曲线(DCA)**

![F7](m2v2_ebm_full/figures/F07_DCA_0M.png)

**F08. 0M EBM 风险三档(dev 三分位)事件率**

![F8](m2v2_ebm_full/figures/F08_RiskTier_0M.png)

**F09. 0M EBM 混淆矩阵(Youden 阈值)**

![F9](m2v2_ebm_full/figures/F09_Confusion_0M.png)


## 1M — EBM 全套(9 图)

**F10. 1M EBM ROC,AUC 0.709**

![F10](m2v2_ebm_full/figures/F10_ROC_1M.png)

**F11. 1M EBM precision-recall,AP 0.626**

![F11](m2v2_ebm_full/figures/F11_PR_1M.png)

**F12. 1M EBM reliability**

![F12](m2v2_ebm_full/figures/F12_Calib_1M.png)

**F13. 1M EBM 原生重要性 top-6**

![F13](m2v2_ebm_full/figures/F13_Importance_1M.png)

**F14. 1M EBM 形状函数 #1:Thyroid weight**

![F14](m2v2_ebm_full/figures/F14_Shape_1M_r1.png)

**F15. 1M EBM 形状函数 #2:FT3,FT4 level**

![F15](m2v2_ebm_full/figures/F15_Shape_1M_r2.png)

**F16. 1M EBM 决策曲线(DCA)**

![F16](m2v2_ebm_full/figures/F16_DCA_1M.png)

**F17. 1M EBM 风险三档(dev 三分位)事件率**

![F17](m2v2_ebm_full/figures/F17_RiskTier_1M.png)

**F18. 1M EBM 混淆矩阵(Youden 阈值)**

![F18](m2v2_ebm_full/figures/F18_Confusion_1M.png)


## 3M — EBM 全套(9 图)

**F19. 3M EBM ROC,AUC 0.804**

![F19](m2v2_ebm_full/figures/F19_ROC_3M.png)

**F20. 3M EBM precision-recall,AP 0.721**

![F20](m2v2_ebm_full/figures/F20_PR_3M.png)

**F21. 3M EBM reliability**

![F21](m2v2_ebm_full/figures/F21_Calib_3M.png)

**F22. 3M EBM 原生重要性 top-6**

![F22](m2v2_ebm_full/figures/F22_Importance_3M.png)

**F23. 3M EBM 形状函数 #1:FT3,FT4 level**

![F23](m2v2_ebm_full/figures/F23_Shape_3M_r1.png)

**F24. 3M EBM 形状函数 #2:Thyroid weight**

![F24](m2v2_ebm_full/figures/F24_Shape_3M_r2.png)

**F25. 3M EBM 决策曲线(DCA)**

![F25](m2v2_ebm_full/figures/F25_DCA_3M.png)

**F26. 3M EBM 风险三档(dev 三分位)事件率**

![F26](m2v2_ebm_full/figures/F26_RiskTier_3M.png)

**F27. 3M EBM 混淆矩阵(Youden 阈值)**

![F27](m2v2_ebm_full/figures/F27_Confusion_3M.png)


## 6M — EBM 全套(9 图)

**F28. 6M EBM ROC,AUC 0.817**

![F28](m2v2_ebm_full/figures/F28_ROC_6M.png)

**F29. 6M EBM precision-recall,AP 0.747**

![F29](m2v2_ebm_full/figures/F29_PR_6M.png)

**F30. 6M EBM reliability**

![F30](m2v2_ebm_full/figures/F30_Calib_6M.png)

**F31. 6M EBM 原生重要性 top-6**

![F31](m2v2_ebm_full/figures/F31_Importance_6M.png)

**F32. 6M EBM 形状函数 #1:FT3,FT4 velocity**

![F32](m2v2_ebm_full/figures/F32_Shape_6M_r1.png)

**F33. 6M EBM 形状函数 #2:TSH velocity**

![F33](m2v2_ebm_full/figures/F33_Shape_6M_r2.png)

**F34. 6M EBM 决策曲线(DCA)**

![F34](m2v2_ebm_full/figures/F34_DCA_6M.png)

**F35. 6M EBM 风险三档(dev 三分位)事件率**

![F35](m2v2_ebm_full/figures/F35_RiskTier_6M.png)

**F36. 6M EBM 混淆矩阵(Youden 阈值)**

![F36](m2v2_ebm_full/figures/F36_Confusion_6M.png)


## 跨地标汇总(11 图)

**F37. EBM ROC/PR/Brier 随地标**

![F37](m2v2_ebm_full/figures/F37_Metrics_over_time.png)

**F38. EBM vs LR 逐地标 AUC**

![F38](m2v2_ebm_full/figures/F38_EBM_vs_LR.png)

**F39. EBM 全地标 ROC**

![F39](m2v2_ebm_full/figures/F39_ROC_all.png)

**F40. EBM 多 seed(5)temporal AUC 稳定性**

![F40](m2v2_ebm_full/figures/F40_MultiSeed_stability.png)

**F41. EBM 原生重要性 组×地标 漂移热力图**

![F41](m2v2_ebm_full/figures/F41_GroupImportance_heatmap.png)

**F42. EBM 校准截距/斜率汇总**

![F42](m2v2_ebm_full/figures/F42_Calib_summary.png)

**F43. FT3-FT4 同向 vs TSH 反向 相关性(轴构造依据)**

![F43](m2v2_ebm_full/figures/F43_Collinearity.png)

**F44. EBM 各地标主导特征形状函数面板**

![F44](m2v2_ebm_full/figures/F44_Lead_shapes_panel.png)

**F45. EBM 选择性弃权 风险-覆盖(pooled)**

![F45](m2v2_ebm_full/figures/F45_Selective_prediction.png)

**F46. EBM 各地标工作点 敏感度/特异度/PPV/NPV**

![F46](m2v2_ebm_full/figures/F46_OperatingPoint.png)

**F47. EBM pooled 校准可靠性(全地标合并)**

![F47](m2v2_ebm_full/figures/F47_Calib_pooled.png)
