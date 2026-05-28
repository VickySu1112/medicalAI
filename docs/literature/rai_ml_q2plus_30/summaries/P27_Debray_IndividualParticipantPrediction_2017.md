# P27. Individual participant data meta-analysis for a binary outcome: one-stage or two-stage?

## Citation

- **Short key**: `Debray_IndividualParticipantPrediction_2017`
- **Year / journal**: 2017, *Statistics in Medicine*
- **DOI**: 10.1002/sim.7582
- **URL**: https://doi.org/10.1002/sim.7582
- **Theme**: Prediction validation / heterogeneity
- **Q2+ evidence**: Statistics in Medicine; Q1/Q2, verify manually
- **Reading basis**: methods metadata
- **Local file**: not downloaded yet

## One-paper independent summary

Although not directly used in our modeling, this paper is useful for discussing why external validation and between-center heterogeneity matter. Our temporal validation is stronger than random split but still not a multi-center validation.

## Methods readout

Methodological comparison of one-stage and two-stage IPD meta-analysis for binary outcomes.

## Main findings relevant to this project

Heterogeneity and transportability matter when estimating and validating prediction effects.

## How it supports the RAI three-module manuscript

Supports the limitation that our thresholds and calibration require external validation.

## Caution / boundary for citation

Not specific to landmark or thyroid disease.

## Module mapping

External-validation limitation

## Abstract / metadata notes

In many studies, it is of interest to predict the future trajectory of subjects based on their historical data, referred to as dynamic prediction. Mixed effects models have traditionally been used for dynamic prediction. However, the commonly used random intercept and slope model is often not sufficiently flexible for modeling subject-specific trajectories. In addition, there may be useful exposures/predictors of interest that are measured concurrently with the outcome, complicating dynamic prediction. To address these problems, we propose a dynamic functional concurrent regression model to handle the case where both the functional response and the functional predictors are irregularly measured. Currently, such a model cannot be fit by existing software. We apply the model to dynamically predict children's length conditional on prior length, weight, and baseline covariates. Inference on model parameters and subject-specific trajectories is conducted using the mixed effects representation of the proposed model. An extensive simulation study shows that the dynamic functional regression model provides more accurate estimation and inference than existing methods. Methods are supported by fast, flexible, open source software that uses heavily tested smoothing techniques.
