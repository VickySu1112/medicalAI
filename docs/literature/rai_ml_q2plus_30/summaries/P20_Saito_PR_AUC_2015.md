# P20. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets

## Citation

- **Short key**: `Saito_PR_AUC_2015`
- **Year / journal**: 2015, *PLOS ONE*
- **DOI**: 10.1371/journal.pone.0118432
- **URL**: https://doi.org/10.1371/journal.pone.0118432
- **Theme**: Imbalanced evaluation / PR-AUC
- **Q2+ evidence**: PLOS ONE is broad peer-reviewed journal; verify current quartile manually
- **Reading basis**: open access article
- **Local file**: not downloaded yet

## One-paper independent summary

This paper justifies emphasizing PR-AUC in our low-event-rate H1 rolling monitoring task. ROC-AUC can look acceptable even when positive predictive value in the high-risk region is clinically limited.

## Methods readout

Methodological comparison of ROC and precision-recall plots under class imbalance.

## Main findings relevant to this project

Precision-recall curves can be more informative than ROC curves for imbalanced classification.

## How it supports the RAI three-module manuscript

Supports our Module 3 interpretation that PR-AUC and risk-tier enrichment are more clinically relevant than accuracy alone.

## Caution / boundary for citation

General classification methodology, not medical-specific.

## Module mapping

M3 low-event-rate evaluation

## Abstract / metadata notes

Binary classifiers are routinely evaluated with performance measures such as sensitivity and specificity, and performance is frequently illustrated with Receiver Operating Characteristics (ROC) plots. Alternative measures such as positive predictive value (PPV) and the associated Precision/Recall (PRC) plots are used less frequently. Many bioinformatics studies develop and evaluate classifiers that are to be applied to strongly imbalanced datasets in which the number of negatives outweighs the number of positives significantly. While ROC plots are visually appealing and provide an overview of a classifier's performance across a wide range of specificities, one can ask whether ROC plots could be misleading when applied in imbalanced classification scenarios. We show here that the visual interpretability of ROC plots in the context of imbalanced datasets can be deceptive with respect to conclusions about the reliability of classification performance, owing to an intuitive but wrong interpretation of specificity. PRC plots, on the other hand, can provide the viewer with an accurate prediction of future classification performance due to the fact that they evaluate the fraction of true positives among positive predictions. Our findings have potential implications for the interpretation of a large number of studies that use ROC plots on imbalanced datasets.
