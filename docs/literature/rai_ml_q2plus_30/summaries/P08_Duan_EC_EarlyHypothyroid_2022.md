# P08. Machine learning identifies baseline clinical features that predict early hypothyroidism in patients with Graves' disease after radioiodine therapy

## Citation

- **Short key**: `Duan_EC_EarlyHypothyroid_2022`
- **Year / journal**: 2022, *Endocrine Connections*
- **DOI**: 10.1530/EC-22-0119
- **URL**: https://doi.org/10.1530/EC-22-0119
- **Theme**: RAI ML / opposite endpoint
- **Q2+ evidence**: Endocrine Connections; likely Q2 in endocrine contexts, verify manually
- **Reading basis**: local PDF + metadata
- **Local file**: ref_paper/pdfs/2022_Duan_EndocrineConnections_ML_early_hypothyroidism_EC-22-0119.pdf

## One-paper independent summary

Although the endpoint is early hypothyroidism rather than persistent/recurrent hyperthyroidism, the paper is useful for machine-learning reporting in the same disease-treatment setting. It demonstrates how baseline clinical variables can be mapped to post-RAI outcomes with interpretable ML.

## Methods readout

Baseline-feature ML models for early hypothyroidism after RAI.

## Main findings relevant to this project

Routine baseline features contain signal for early post-RAI thyroid state.

## How it supports the RAI three-module manuscript

Methodological comparator for feature engineering, ML benchmark and interpretability.

## Caution / boundary for citation

Endpoint direction is opposite to NHRH; cannot be used as direct efficacy comparator.

## Module mapping

M1 ML benchmark context

## Abstract / metadata notes

Background and objective: Radioiodine therapy (RAI) is one of the most common treatment solutions for Graves' disease (GD). However, many patients will develop hypothyroidism as early as 6 months after RAI. This study aimed to implement machine learning (ML) algorithms for the early prediction of post-RAI hypothyroidism. Methods: Four hundred and seventy-one GD patients who underwent RAI between January 2016 and June 2019 were retrospectively recruited and randomly split into the training set (310 patients) and the validation set (161 patients). These patients were followed for 6 months after RAI. A set of 138 clinical and lab test features from the electronic medical record (EMR) were extracted, and multiple ML algorithms were conducted to identify the features associated with the occurrence of hypothyroidism 6 months after RAI. Results: An integrated multivariate model containing patients' age, thyroid mass, 24-h radioactive iodine uptake, serum concentrations of aspartate aminotransferase, thyrotropin-receptor antibodies, thyroid microsomal antibodies, and blood neutrophil count demonstrated an area under the receiver operating curve (AUROC) of 0.72 (95% CI: 0.61-0.85), an F1 score of 0.74, and an MCC score of 0.63 in the training set. The model also performed well in the validation set with an AUROC of 0.74 (95% CI: 0.65-0.83), an F1 score of 0.74, and a MCC of 0.63. A user-friendly nomogram was then established to facilitate the clinical utility. Conclusion: The developed multivariate model based on EMR data could be a valuable tool for predicting post-RAI hypothyroidism, allowing them to be treated differently before the therapy. Further study is needed to validate the developed prognostic model at independent sites.
