# P29. Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records

## Citation

- **Short key**: `Miotto_DeepPatient_2016`
- **Year / journal**: 2016, *Scientific Reports*
- **DOI**: 10.1038/srep26094
- **URL**: https://doi.org/10.1038/srep26094
- **Theme**: Longitudinal EHR representation learning
- **Q2+ evidence**: Scientific Reports is broad peer-reviewed journal; verify current quartile manually
- **Reading basis**: open access clinical ML article
- **Local file**: not downloaded yet

## One-paper independent summary

Deep Patient is a classic EHR representation-learning paper. It helps frame why longitudinal histories can be transformed into patient-level risk representations, while also highlighting that our current project deliberately uses simpler, auditable landmark features.

## Methods readout

Unsupervised representation learning from EHR followed by predictive modeling.

## Main findings relevant to this project

Temporal EHR histories can encode future disease risk.

## How it supports the RAI three-module manuscript

Background for representation learning and why we avoid unreviewable black-box complexity in this dataset.

## Caution / boundary for citation

Not thyroid-specific and much larger EHR scale than our cohort.

## Module mapping

Longitudinal ML background

## Abstract / metadata notes

Secondary use of electronic health records (EHRs) promises to advance clinical research and better inform clinical decision making. Challenges in summarizing and representing patient data prevent widespread practice of predictive modeling using EHRs. Here we present a novel unsupervised deep feature learning method to derive a general-purpose patient representation from EHR data that facilitates clinical predictive modeling. In particular, a three-layer stack of denoising autoencoders was used to capture hierarchical regularities and dependencies in the aggregated EHRs of about 700,000 patients from the Mount Sinai data warehouse. The result is a representation we name "deep patient". We evaluated this representation as broadly predictive of health states by assessing the probability of patients to develop various diseases. We performed evaluation using 76,214 test patients comprising 78 diseases from diverse clinical domains and temporal windows. Our results significantly outperformed those achieved using representations based on raw EHR data and alternative feature learning strategies. Prediction performance for severe diabetes, schizophrenia, and various cancers were among the top performing. These findings indicate that deep learning applied to EHRs can derive patient representations that offer improved clinical predictions, and could provide a machine learning framework for augmenting clinical decision systems.
