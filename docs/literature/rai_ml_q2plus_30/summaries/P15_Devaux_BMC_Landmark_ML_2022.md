# P15. Individual dynamic predictions of clinical endpoint from large dimensional longitudinal biomarker history: a landmark approach

## Citation

- **Short key**: `Devaux_BMC_Landmark_ML_2022`
- **Year / journal**: 2022, *BMC Medical Research Methodology*
- **DOI**: 10.1186/s12874-022-01660-3
- **URL**: https://doi.org/10.1186/s12874-022-01660-3
- **Theme**: Landmark + longitudinal biomarker summaries
- **Q2+ evidence**: BMC Medical Research Methodology is a recognized methods journal; verify quartile manually
- **Reading basis**: open access methods article
- **Local file**: not downloaded yet

## One-paper independent summary

This paper is highly relevant because it explicitly combines landmark prediction with longitudinal biomarker histories and machine-learning-style summaries. It supports our use of current values, slopes, cumulative histories and trajectory summaries instead of treating each lab as a static covariate.

## Methods readout

Landmark approach for dynamic prediction using high-dimensional longitudinal biomarker summaries.

## Main findings relevant to this project

Summaries of individual biomarker trajectories can be incorporated into dynamic prediction models.

## How it supports the RAI three-module manuscript

Supports our trajectory momentum design and biomarker family ablation language.

## Caution / boundary for citation

Not specific to Graves disease or RAI.

## Module mapping

M2/M3 biomarker trajectory methods

## Abstract / metadata notes

BACKGROUND: The individual data collected throughout patient follow-up constitute crucial information for assessing the risk of a clinical event, and eventually for adapting a therapeutic strategy. Joint models and landmark models have been proposed to compute individual dynamic predictions from repeated measures to one or two markers. However, they hardly extend to the case where the patient history includes much more repeated markers. Our objective was thus to propose a solution for the dynamic prediction of a health event that may exploit repeated measures of a possibly large number of markers. METHODS: We combined a landmark approach extended to endogenous markers history with machine learning methods adapted to survival data. Each marker trajectory is modeled using the information collected up to the landmark time, and summary variables that best capture the individual trajectories are derived. These summaries and additional covariates are then included in different prediction methods adapted to survival data, namely regularized regressions and random survival forests, to predict the event from the landmark time. We also show how predictive tools can be combined into a superlearner. The performances are evaluated by cross-validation using estimators of Brier Score and the area under the Receiver Operating Characteristic curve adapted to censored data. RESULTS: We demonstrate in a simulation study the benefits of machine learning survival methods over standard survival models, especially in the case of numerous and/or nonlinear relationships between the predictors and the event. We then applied the methodology in two prediction contexts: a clinical context with the prediction of death in primary biliary cholangitis, and a public health context with age-specific prediction of death in the general elderly population. CONCLUSIONS: Our methodology, implemented in R, enables the prediction of an event using the entire longitudinal patient history, even when the number of repeated markers is large. Although introduced with mixed models for the repeated markers and methods for a single right censored time-to-event, the technique can be used with any other appropriate modeling technique for the markers and can be easily extended to competing risks setting.
