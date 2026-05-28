# P28. Scalable and accurate deep learning with electronic health records

## Citation

- **Short key**: `Rajkomar_NPJ_EHR_2018`
- **Year / journal**: 2018, *npj Digital Medicine*
- **DOI**: 10.1038/s41746-018-0029-1
- **URL**: https://doi.org/10.1038/s41746-018-0029-1
- **Theme**: Clinical ML with longitudinal EHR
- **Q2+ evidence**: npj Digital Medicine is a high-impact digital medicine journal; Q1/Q2, verify manually
- **Reading basis**: open access clinical ML article
- **Local file**: not downloaded yet

## One-paper independent summary

This article is a broad clinical-ML benchmark showing that longitudinal EHR data can support multiple prediction tasks. For our paper, it is useful as contrast: we choose interpretable landmark LR rather than high-capacity deep learning because sample size, event rate and clinical transparency differ.

## Methods readout

Large-scale EHR deep learning for multiple clinical prediction endpoints.

## Main findings relevant to this project

Longitudinal EHR information can improve predictions, but model complexity requires careful validation and interpretation.

## How it supports the RAI three-module manuscript

Supports the general idea of longitudinal medical data prediction while justifying our simpler model choice.

## Caution / boundary for citation

Different scale, data type and clinical setting.

## Module mapping

Clinical ML context

## Abstract / metadata notes

Predictive modeling with electronic health record (EHR) data is anticipated to drive personalized medicine and improve healthcare quality. Constructing predictive statistical models typically requires extraction of curated predictor variables from normalized EHR data, a labor-intensive process that discards the vast majority of information in each patient's record. We propose a representation of patients' entire raw EHR records based on the Fast Healthcare Interoperability Resources (FHIR) format. We demonstrate that deep learning methods using this representation are capable of accurately predicting multiple medical events from multiple centers without site-specific data harmonization. We validated our approach using de-identified EHR data from two US academic medical centers with 216,221 adult patients hospitalized for at least 24 h. In the sequential format we propose, this volume of EHR data unrolled into a total of 46,864,534,945 data points, including clinical notes. Deep learning models achieved high accuracy for tasks such as predicting: in-hospital mortality (area under the receiver operator curve [AUROC] across sites 0.93-0.94), 30-day unplanned readmission (AUROC 0.75-0.76), prolonged length of stay (AUROC 0.85-0.86), and all of a patient's final discharge diagnoses (frequency-weighted AUROC 0.90). These models outperformed traditional, clinically-used predictive models in all cases. We believe that this approach can be used to create accurate and scalable predictions for a variety of clinical scenarios. In a case study of a particular prediction, we demonstrate that neural networks can be used to identify relevant information from the patient's chart.
