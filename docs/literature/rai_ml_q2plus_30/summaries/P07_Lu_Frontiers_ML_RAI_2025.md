# P07. From data to decision: an interpretable machine learning model for optimizing RAI therapy in Graves' hyperthyroidism

## Citation

- **Short key**: `Lu_Frontiers_ML_RAI_2025`
- **Year / journal**: 2025, *Frontiers in Endocrinology*
- **DOI**: 10.3389/fendo.2025.1711029
- **URL**: https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1711029/full
- **Theme**: Interpretable ML RAI benchmark
- **Q2+ evidence**: Frontiers in Endocrinology; verify current local quartile manually
- **Reading basis**: local PDF + full-text webpage
- **Local file**: ref_paper/pdfs/2025_Lu_Frontiers_interpretable_ML_RAI_1711029.pdf

## One-paper independent summary

This is the closest machine-learning comparator. It uses multiple ML algorithms and interpretability tools for RAI therapy outcome prediction. It is useful as a benchmark, but the validation design and endpoint are different, so our report should compare cautiously rather than directly claim superiority.

## Methods readout

ML benchmark including tree models and interpretation; reports discrimination and clinical evaluation.

## Main findings relevant to this project

Nonlinear models can perform well in internal/random validation settings for RAI outcome prediction.

## How it supports the RAI three-module manuscript

Justifies our inclusion of nonlinear ML benchmarks and SHAP, while motivating temporal validation.

## Caution / boundary for citation

Likely less stringent temporal validation than our Module 1/2/3 design; endpoint definitions differ.

## Module mapping

M1 advanced ML benchmark and discussion

## Abstract / metadata notes

Objective: Radioactive iodine (RAI) therapy is a cornerstone treatment for Graves' hyperthyroidism (GH), yet failure rates remain significant due to the complexity of individual patient responses. Traditional fixed-dose or simple calculated-dose methods often fail to account for non-linear interactions among clinical features. Methods: We retrospectively analyzed data from 1,292 GH patients who received initial RAI therapy between June 2018 and July 2024. Comprehensive pre-treatment clinical, laboratory, and imaging data, including age, gender, FT4, 3-hour radioactive iodine uptake (RAIU 3h), thyroid weight, and thyroid receptor antibodies (TRAb), were collected. Stepwise regression with the Akaike Information Criterion (AIC) was employed for feature selection, identifying nine optimal predictors. Six machine learning algorithms were compared, with performance evaluated using AUC, Brier score, and Decision Curve Analysis (DCA). SHapley Additive exPlanations (SHAP) analysis provided model interpretability. Results: The final cohort, comprising 1,292 patients (61.3% female, median age 37 years), achieved a 75.8% remission rate. Nine significant variables were identified as optimal predictors: gender, age, history of antithyroid drug use, disease course over 2 years, total iodine dose (TID), free thyroxine (FT4), RAIU 3h, thyroid weight, and TRAb. Among the algorithms tested, the Random Forest (RF) model demonstrated superior performance, achieving an AUC of 0.950 on the independent test set and a Brier score of 0.067, indicating excellent discrimination and calibration. SHAP analysis confirmed RAIU 3h, FT4, age, and thyroid weight as the most influential features, providing clinical transparency. Conclusion: The developed interpretable machine learning framework offers a precise, personalized tool for predicting RAI outcomes, potentially guiding optimizing dosing strategies to reduce treatment failure.
