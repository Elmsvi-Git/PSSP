# Psychosomatic Susceptibility Profile (PSSP)

## Overview

This repository contains the implementation of the machine learning pipeline developed for the study:

**"Data-driven identification of multidimensional psychosomatic profiles and derivation of a brief classification tool (PSSP)"**

The goal of this project is to identify latent psychosomatic profiles using multidimensional data and to derive a concise, machine learning–based classification tool based on personality and coping items.

Due to ethical and privacy restrictions, the original dataset cannot be shared publicly. Therefore, a representative subsample of the data is provided to demonstrate the implementation of the supervised learning and feature selection pipeline.

---

## Pipeline Description

The analytical framework consists of two stages:

### 1. Unsupervised Learning (Profile Discovery – Not Included in Code Execution)

- Integration of multi-domain variables:
  - Personality traits (NEO-FFI)
  - Coping strategies (COPE Inventory)
  - Psychological status (GHQ-12)
  - Body Mass Index (BMI)
  - Physical activity
- Text-based transformation of questionnaire responses
- Contextual embedding using DistilBERT
- Dimensionality reduction using PCA
- Spectral clustering to identify latent psychosomatic profiles

> ⚠️ Note: This stage was performed on the full dataset and is not fully reproducible here due to data confidentiality. Cluster labels derived from this stage are provided in the subsample for downstream analysis.

---

### 2. Supervised Learning (Classification + Feature Selection – Fully Reproducible)

- XGBoost classifier trained to predict cluster membership
- Five-fold cross-validation
- SHAP-based feature importance analysis
- Selection of a minimal 8-item subset of features
- Construction of the Psychosomatic Susceptibility Profile (PSSP)

---

## Data Availability

- **Full dataset:** Not publicly available due to ethical and privacy constraints.
- **Provided dataset:** A representative anonymized subsample is included for reproducibility of the classification pipeline.

The subsample preserves variable structure and general statistical properties of the original dataset.

---

## ⚠️ Important Note on Reproducibility

Because the full clustering procedure was conducted on the complete dataset, and only a subsample is provided here:

- Minor differences in model performance metrics may occur compared to those reported in the manuscript.
- The overall methodological pipeline and feature selection process remain fully consistent with the original analysis.
- The purpose of this repository is to demonstrate the reproducibility of the classification and feature selection steps, not to exactly replicate published performance values.

---

## Installation


pip install -r requirements.txt


## How to Run
Run classification and feature selection pipeline:

python PSSP_FeatureSelction_Classification.py


## Outputs

The script generates:

Trained XGBoost classification model
Cross-validation performance metrics (F1-score, recall, etc.)
SHAP feature importance plots
Selected 8-item PSSP feature set


## Reproducibility 

All preprocessing steps, model configurations, and feature selection procedures are fully documented in the code. The workflow is designed to ensure methodological transparency and reproducibility of the supervised learning component.

## Ethical Statement

The study uses secondary, anonymized data collected under appropriate ethical approval. No identifiable personal information is included in this repository.

## Contact

For questions regarding the methodology or implementation, please contact the corresponding author. (El.msvi@gmail.com)
