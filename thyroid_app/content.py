"""Static display content for the Chinese Streamlit app."""

OVERVIEW_PLOTS = [
    ("四分支动态复发框架", "results/t5_dynamic_paper/figures/Figure_00_Temporal_Landmark_Framework_Overview.png"),
    ("动态复发判别能力", "results/t5_dynamic_paper/figures/Figure_04_PR_Curves.png"),
    ("患者级风险分层", "results/t5_dynamic_paper/figures/Figure_17_Readme4_Patient_Q1Q4_CI.png"),
    ("模型解释与分支贡献", "results/t5_dynamic_paper/figures/Figure_23_Gate_Branch_Contribution.png"),
]

RELAPSE_PLOTS = [
    ("PR 曲线", "results/t5_dynamic_paper/figures/Figure_04_PR_Curves.png"),
    ("校准曲线", "results/t5_dynamic_paper/figures/Figure_06_Calibration_Curves.png"),
    ("决策曲线分析", "results/t5_dynamic_paper/figures/Figure_10_Decision_Curve.png"),
    ("阈值敏感性", "results/t5_dynamic_paper/figures/Figure_08_Threshold_Sensitivity.png"),
    ("混淆矩阵", "results/t5_dynamic_paper/figures/Figure_07_Confusion_Matrices.png"),
    ("SHAP 重要性", "results/t5_dynamic_paper/figures/Figure_25_SHAP_Feature_Bar_Dev.png"),
]

FIXED_3M_PLOTS = [
    ("3M / 6M 性能热图", "results/landmark_focus/figures/Figure_Binary_Performance_Heatmaps.png"),
    ("3M ROC / PR 曲线", "results/landmark_focus/figures/Figure_3M_LR_vs_LGBM_ROC_PR.png"),
    ("3M 校准与 DCA", "results/landmark_focus/figures/Figure_3M_LR_vs_LGBM_Calibration_DCA.png"),
    ("3M 阈值敏感性", "results/landmark_focus/figures/Figure_LR_Sparse_Threshold_Sensitivity.png"),
    ("3M 混淆矩阵", "results/landmark_focus/figures/Figure_3M_LR_vs_LGBM_Confusion.png"),
    ("3M LGBM SHAP", "results/landmark_focus/figures/Figure_3M_LGBM_SHAP_Bar.png"),
]
