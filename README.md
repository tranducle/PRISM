# PRISM: Predictive Risk Association Profiler

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research_Code-orange.svg)

This repository contains the core analytical scripts used for the **PRISM** framework, an empirical cybersecurity predictive modeling architecture. The study integrates tree-based machine learning (Random Forest) with cooperative game theory (SHAP TreeExplainer) to evaluate the predictive association of socio-technical cybersecurity controls against disparate threat vectors (Malware vs. Phishing).

> **Note on Anonymity:** This repository has been blinded for double-blind peer review. Author names, institutional affiliations, and identifying meta-data have been removed from all scripts and documentation.

---

## 📌 Repository Structure

```text
PRISM/
│
├── src/
│   ├── csbs_5x2cv.py              # Cross-validation paired t-test evaluation
│   ├── run_prism_shap.py          # Primary predictive model (Random Forest + SHAP)
│   └── run_prism_robustness.py    # Asymmetric noise falsification test
│
├── README.md                      # This documentation
└── requirements.txt               # Dependencies for reproducibility
```

---

## 🚀 Key Capabilities

1. **Disaggregated Threat Modeling:** Simultaneously evaluates control efficacy against technically-oriented threats (Malware) versus governance-oriented threats (Phishing/Social Engineering).
2. **Cost-Weighted Interpretability (SHAP-ROI):** Translates raw machine learning SHAP values into cost-adjusted strategic prioritization rankings.
3. **Robustness Falsification:** Injects asymmetric reporting noise (Social Desirability Bias) to prove the stability of predictive rankings under strict adversarial conditions.

---

## 📊 Data Availability Statement

Due to strict data usage agreements and data privacy regulations, the raw Cyber Security Breaches Survey (CSBS) dataset used in this study cannot be redistributed directly within this repository. 

However, all underlying survey data is fully within the public domain. Researchers can freely access and download the raw datasets through the official UK Data Service portal.

- **Data Source:** Department for Science, Innovation and Technology (DSIT) and Home Office.
- **Access Link:** [UK Data Service - Cyber Security Breaches Survey Series](https://ukdataservice.ac.uk/) (Search for "Cyber Security Breaches Survey").
- **Preparation:** Our models use a harmonized, multi-wave pooled subset of this data (2018-2023) focusing on Micro, Small, and Medium Enterprises (SMEs).

---

## ⚙️ Code Availability & Usage

The code provided in this repository is complete and reproducible given the corresponding dataset. To replicate the empirical models:

1. **Environment Setup:**
   Ensure Python 3.12+ is installed. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation:**
   Place the cleaned and harmonized dataset (`csbs_pooled_clean.csv`) in the `data/` directory (or modify the paths within the `src/` scripts to point to your data location).

3. **Execution:**
   Run the scripts sequentially to reproduce the analyses:
   ```bash
   python src/csbs_5x2cv.py
   python src/run_prism_shap.py
   python src/run_prism_robustness.py
   ```

---

## 🛡️ License

This code is released under the **MIT License**. It is provided "as is" for research and reproducibility purposes.
