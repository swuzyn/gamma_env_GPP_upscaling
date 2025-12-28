# γ_env: Environment-regulated Conversion Factor for Upscaling Instantaneous GPP

This repository provides the implementation of an environment-regulated conversion factor (γ_env) for upscaling instantaneous gross primary productivity (GPP) observations to daily GPP. The proposed framework explicitly integrates radiation conditions and key environmental drivers to improve robustness under heterogeneous environmental stress across ecosystems.

The codebase is designed to support full reproducibility of the modeling and evaluation workflow presented in the accompanying manuscript.

---

## Overview of the Workflow

The workflow consists of two sequential stages, each implemented in a separate script and serving a distinct role in the manuscript:

1. **Site-based 10-fold cross-validation and model evaluation**  
2. **Independent-site evaluation using a fully trained model**

Both stages operate on the conversion factor γ (the ratio between instantaneous and daily GPP), rather than directly predicting daily GPP during model training.

---

## Script 1: Site-based 10-fold Cross-Validation on FLUXNET 2015

**Script:** `train_gamma_env_rf.py`

### Purpose

This script trains and evaluates the γ_env model using the **FLUXNET 2015** eddy-covariance dataset. A strict **site-based 10-fold cross-validation** strategy is applied, in which entire sites are withheld in each fold to ensure spatial independence between training and validation samples.

### Role in the Manuscript

The out-of-fold (OOF) predictions produced by this script constitute the primary dataset used in the Results section of the manuscript, including:

- Overall estimation accuracy assessment  
- Generalization across ecosystem types  
- Robustness under environmental stress (e.g., RMD gradients)  
- Consistency of diurnal dynamic features (e.g., centroid time, AM/PM ratio, peak timing)  

The OOF-predicted γ_env values are written back to the original sample table. These predicted conversion factors are subsequently used to derive estimated daily GPP, which forms the basis for all statistical analyses and grouped comparisons reported in the manuscript.

### Key Characteristics

- Strict site-based (non-random) 10-fold cross-validation  
- Prediction target: γ_env (ratio of instantaneous to daily GPP)  
- Machine learning model: Random Forest  
- Hyperparameters selected based on cross-validation performance  
- All variables are constructed at the **monthly scale**  

### Main Outputs

- A **summary CSV** reporting cross-validation performance metrics  
- A **prediction CSV** containing:
  - Observed γ (e.g., `ratio_GPP_DT`)  
  - OOF-predicted γ_env  
  - Estimated daily GPP derived from γ_env  
  - Site identifiers and environmental metadata  

---

## Script 2: Independent-Site Evaluation

**Script:** `apply_gamma_env_independent.py`

### Purpose

This script evaluates the transferability of the γ_env model using an **independent site dataset** that is not involved in model training or cross-validation.

The model is retrained on the **entire FLUXNET 2015 dataset** using the optimal hyperparameters identified in Script 1, and then applied to the independent dataset.

### Role in the Manuscript

This analysis focuses on overall model performance at independent sites and is primarily reported in the **Supplementary Materials**, with brief reference in the main text.

### Comparison Methods

The script compares three upscaling approaches:

- **γ_env**: environment-regulated conversion factor (Random Forest)  
- **γ_SW**: shortwave radiation ratio method  
- **γ_COS**: cosine-based solar zenith angle ratio method  

Daily GPP is estimated as:
Daily GPP = Instantaneous GPP / γ

for each method.

### Main Outputs

- Printed performance metrics (R², RMSE, MAE, Bias) for each method  
- An output CSV containing:
  - Predicted γ_env  
  - Daily GPP estimates derived from γ_env, γ_SW, and γ_COS  
  - Observed daily GPP for validation  

---

## Input Data Requirements

### Temporal Resolution

All input variables are constructed at the **monthly scale**.  
“Instantaneous” refers to monthly means at a fixed time of day, rather than single-day or single-observation measurements.

### Required Variables (case-insensitive)

- Instantaneous GPP (e.g., `GPP_DT_mean`)  
- Daily GPP reference (e.g., `daily_GPP_DT_mean`)  
- Environmental drivers:
  - Shortwave radiation ratio (`ratio_SW_IN_F`)  
  - Air temperature (`TA_F_mean`)  
  - Vapor pressure deficit (`VPD_F_mean`)  
  - Soil water content (`SWC_F_MDS_1_mean`)  
- Cosine-based solar zenith angle ratio (`COS_SZA_ratio`, used in independent evaluation)

Column names are matched in a **case-insensitive** manner to ensure robustness across datasets.

### Data Sources

- Training data: **FLUXNET 2015**  
- Independent-site data: publicly available eddy-covariance datasets (e.g., ICOS, AmeriFlux), processed following the same methodology  

A small example dataset may be provided for testing purposes. Complete datasets can be obtained from the respective data portals and processed according to the methods described in the manuscript. For questions regarding data preparation, users may also contact the authors.

---

## Reproducibility Notes

- Hyperparameter selection relies exclusively on site-based cross-validation  
- Independent-site data are not used at any stage of model training or tuning  
- Random seeds are fixed where applicable to ensure reproducibility  

---

## Citation

If you use this code, please cite the corresponding paper:

> Robust estimation of daily photosynthesis from instantaneous observations through machine-learning integration of radiation and environmental drivers

---

## Contact

For questions regarding the code or methodology, please contact:
zhouyn36@mail2.sysu.edu.cn

