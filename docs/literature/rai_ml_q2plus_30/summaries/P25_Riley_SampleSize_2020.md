# P25. Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes

## Citation

- **Short key**: `Riley_SampleSize_2020`
- **Year / journal**: 2020, *Statistics in Medicine*
- **DOI**: 10.1002/sim.7992
- **URL**: https://doi.org/10.1002/sim.7992
- **Theme**: Prediction model sample size
- **Q2+ evidence**: Statistics in Medicine is a leading biostatistics journal; Q1/Q2, verify manually
- **Reading basis**: methods article
- **Local file**: not downloaded yet

## One-paper independent summary

This paper is relevant for judging whether complex models are appropriate for the available event counts. It supports our conservative preference for calibrated LR when treatment-pre baseline information and event counts limit complex ML gains.

## Methods readout

Sample-size framework for developing prediction models with binary/time-to-event outcomes.

## Main findings relevant to this project

Prediction model development requires attention to overfitting, shrinkage, event fraction and anticipated performance.

## How it supports the RAI three-module manuscript

Supports our argument that not all tasks justify high-capacity nonlinear models.

## Caution / boundary for citation

Generic methods paper; exact calculations would need our final event counts and predictor degrees of freedom.

## Module mapping

Model complexity and limitations

## Abstract / metadata notes

When designing a study to develop a new prediction model with binary or time‐to‐event outcomes, researchers should ensure their sample size is adequate in terms of the number of participants ( n ) and outcome events ( E ) relative to the number of predictor parameters ( p ) considered for inclusion. We propose that the minimum values of n and E (and subsequently the minimum number of events per predictor parameter, EPP) should be calculated to meet the following three criteria: (i) small optimism in predictor effect estimates as defined by a global shrinkage factor of ≥ 0.9, (ii) small absolute difference of ≤ 0.05 in the model's apparent and adjusted Nagelkerke's R 2 , and (iii) precise estimation of the overall risk in the population. Criteria (i) and (ii) aim to reduce overfitting conditional on a chosen p , and require prespecification of the model's anticipated Cox‐Snell R 2 , which we show can be obtained from previous studies. The values of n and E that meet all three criteria provides the minimum sample size required for model development. Upon application of our approach, a new diagnostic model for Chagas disease requires an EPP of at least 4.8 and a new prognostic model for recurrent venous thromboembolism requires an EPP of at least 23. This reinforces why rules of thumb (eg, 10 EPP) should be avoided. Researchers might additionally ensure the sample size gives precise estimates of key predictor effects; this is especially important when key categorical predictors have few events in some categories, as this may substantially increase the numbers required.
