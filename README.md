# Predictive Modeling via Stacked Ensemble Learning: A Two-Tiered Classification Approach

**Course:** ES 678 Machine Learning (Course Project)  
**Author:** Meet Rajendrakumar Jain (Roll No. 26210087)  
**Institution:** Indian Institute of Technology Gandhinagar  

---

## Overview

This repository contains the reproducible code and analysis for predicting passenger survival on the Kaggle *Titanic: Machine Learning from Disaster* dataset ($N_{\text{train}} = 891$, $N_{\text{test}} = 418$) using a **two-level stacked ensemble**:

- **Level-1 Base Learners (5 diverse models):** Random Forest, Extra Trees, AdaBoost, Gradient Boosting, and RBF-kernel SVM (with `StandardScaler` inside a fold-isolated `Pipeline`).
- **Level-2 Meta-Learner:** Shallow, heavily regularized **XGBoost** (`max_depth=2`, `learning_rate=0.03`, `reg_alpha=1.0`, `reg_lambda=5.0`) trained on strict 5-fold stratified Out-of-Fold (OOF) predicted probabilities.
- **Leak-Free Preprocessing:** Custom `TitanicPreprocessor` transformer learns all conditional imputations (`Age` by `Title`/`Pclass`, `Fare` by `Pclass`, `Embarked` mode, `TicketGroupSize`) **strictly within each training fold**.

### Key Results (5-Fold Stratified CV, `SEED=42`)
| Model | CV Accuracy $\pm$ SD | ROC-AUC | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Default RF) | $0.8114 \pm 0.0301$ | $0.8750$ | $0.7529$ | $0.1318$ |
| Best Single Base (Tuned RF) | $0.8406 \pm 0.0124$ | $0.8716$ | $0.7829$ | $0.1229$ |
| **Stacked XGBoost (Meta)** | **$0.8316 \pm 0.0103$** | **$0.8902$** | **$0.7664$** | **$0.1204$** |

- **vs. Baseline RF:** $+2.02\%$ accuracy improvement (McNemar exact $p = 0.0356$) and major calibration improvement (Log-Loss $0.3922$ vs. $0.7953$).
- **Graduate Depth Analyses:** Residual error correlation ($r \in [0.91, 0.99]$), meta-learner regularization ablation (overfitting gap reduced from $0.1818$ to $0.0155$), pure vs. passthrough stacking, calibration curves, mistake-correction audit, and Level-1/Level-2 SHAP explainability.

---

## Repository Structure

```text
.
├── new_ml_updated.py    # End-to-end training, CV evaluation, depth analyses, and figure/LaTeX export
├── requirements.txt     # Python package dependencies
└── README.md            # Data access, reproducibility notes, and disclosures
