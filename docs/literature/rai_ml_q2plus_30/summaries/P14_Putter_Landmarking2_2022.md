# P14. Landmarking 2.0: Bridging the gap between joint models and landmarking

## Citation

- **Short key**: `Putter_Landmarking2_2022`
- **Year / journal**: 2022, *Statistics in Medicine*
- **DOI**: 10.1002/sim.9336
- **URL**: https://pubmed.ncbi.nlm.nih.gov/35098578/
- **Theme**: Landmark dynamic prediction methodology
- **Q2+ evidence**: Statistics in Medicine is a leading biostatistics journal; Q1/Q2, verify manually
- **Reading basis**: PubMed abstract + methods article
- **Local file**: not downloaded yet

## One-paper independent summary

This paper is one of the methodological anchors for our dynamic landmark framing. It argues for landmarking as a practical bridge between full joint modeling and simpler dynamic prediction strategies, especially when longitudinal biomarkers are available.

## Methods readout

Statistical methodology for landmark dynamic prediction with longitudinal information.

## Main findings relevant to this project

Landmarking can update risk predictions using time-dependent biomarker summaries without requiring a fully specified joint model.

## How it supports the RAI three-module manuscript

Directly supports Module 2/3 as landmark updaters rather than separate unrelated classifiers.

## Caution / boundary for citation

Methodological paper; our implementation uses pragmatic ML/logistic approximations rather than the full formal framework.

## Module mapping

M2/M3 methodology

## Abstract / metadata notes

The problem of dynamic prediction with time-dependent covariates, given by biomarkers, repeatedly measured over time, has received much attention over the last decades. Two contrasting approaches have become in widespread use. The first is joint modeling, which attempts to jointly model the longitudinal markers and the event time. The second is landmarking, a more pragmatic approach that avoids modeling the marker process. Landmarking has been shown to be less efficient than correctly specified joint models in simulation studies, when data are generated from the joint model. When the mean model is misspecified, however, simulation has shown that joint models may be inferior to landmarking. The objective of this article is to develop methods that improve the predictive accuracy of landmarking, while retaining its relative simplicity and robustness. We start by fitting a working longitudinal model for the biomarker, including a temporal correlation structure. Based on that model, we derive a predictable time-dependent process representing the expected value of the biomarker after the landmark time, and we fit a time-dependent Cox model based on the predictable time-dependent covariate. Dynamic predictions based on this approach for new patients can be obtained by first deriving the expected values of the biomarker, given the measured values before the landmark time point, and then calculating the predicted probabilities based on the time-dependent Cox model. We illustrate the approach in predicting overall survival in liver cirrhosis patients based on prothrombin index.
