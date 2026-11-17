# Predictive Modeling via Stacked Ensemble Learning: A Two-Tiered Classification Approach

**Course:** ES 678 — Machine Learning (Course Project)  
**Author:** Meet Rajendrakumar Jain (Roll No. 26210087)  
**Institution:** Indian Institute of Technology Gandhinagar  

---

## Overview

This repository contains the source code, LaTeX paper source, generated figures, and reproducibility notes for predicting passenger survival on the Kaggle *Titanic: Machine Learning from Disaster* dataset (N_train = 891, N_test = 418) using a **two-level stacked ensemble**:

- **Baseline Model:** Default `RandomForestClassifier` (100 trees) evaluated on the same leak-free fold features.
- **Level-1 Base Learners (5 diverse models):** Random Forest, Extra Trees, AdaBoost, Gradient Boosting, and RBF-kernel SVM (with `StandardScaler` inside a fold-isolated `Pipeline`).
- **Level-2 Meta-Learner:** Shallow, heavily regularized **XGBoost** (`n_estimators=120`, `max_depth=2`, `learning_rate=0.03`, `reg_alpha=1.0`, `reg_lambda=5.0`) trained strictly on 5-fold stratified Out-of-Fold (OOF) predicted probabilities.
- **Leak-Free Preprocessing (`TitanicPreprocessor`):** Learns conditional imputations (`Age` by `Title`/`Pclass`, `Fare` by `Pclass`, `Embarked` mode, and `TicketGroupSize`) **strictly within each training fold**, preventing data leakage.

---

## Key Results (5-Fold Stratified CV, `SEED=42`)

| Model | Type | CV Accuracy ± SD | 95% Bootstrap CI | ROC-AUC | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Default RF) | Baseline | 0.8114 ± 0.0301 | [0.787, 0.837] | 0.8750 | 0.7529 | 0.1318 |
| Random Forest (Tuned) | Base | 0.8406 ± 0.0124 | [0.817, 0.865] | 0.8716 | 0.7829 | 0.1229 |
| Extra Trees | Base | 0.8305 ± 0.0075 | — | 0.8681 | 0.7722 | 0.1275 |
| AdaBoost | Base | 0.8058 ± 0.0139 | — | 0.8573 | 0.7375 | 0.1482 |
| Gradient Boosting | Base | 0.8395 ± 0.0112 | — | 0.8807 | 0.7843 | 0.1188 |
| RBF-SVM (Scaled) | Base | 0.8272 ± 0.0185 | — | 0.8668 | 0.7688 | 0.1311 |
| **Stacked XGBoost** | **Meta** | **0.8316 ± 0.0103** | **[0.807, 0.856]** | **0.8902** | **0.7664** | **0.1204** |

### Main Findings & Graduate Depth Analyses
- **Comparison vs. Baseline:** Stacked XGBoost improves 5-fold CV accuracy by **+2.02%** over the Default Random Forest baseline (McNemar exact *p* = 0.0356; 42 errors fixed vs. 24 introduced) and achieves the **highest ROC-AUC (0.8902)** across all seven models while cutting Log-Loss from 0.7953 to 0.3922.
- **Diversity & Error Correlation:** Residual error correlation across the five base models remains high (*r* ∈ [0.91, 0.99]), explaining why stacking boosts probability ranking (ROC-AUC) and calibration rather than beating the strongest single tree model (Tuned RF, 84.06%; McNemar *p* = 0.1338) at a fixed 0.5 threshold.
- **Overfitting Control Ablation:** Unregularized deep XGBoost memorizes OOF folds (train 0.9910 vs. CV 0.8092, gap 0.1818), whereas our regularized shallow meta-learner reduces the overfitting gap to **0.0155** (train 0.8471 vs. CV 0.8316).
- **Architecture & Explainability:** Pure OOF stacking (0.8316) slightly outperforms passthrough stacking (0.8305), corrects **15** cases where unweighted base majority voting fails, and SHAP analysis highlights `Title_Mr`, `Sex`, `Pclass`, `FarePerPerson`, and `Age` as top drivers.
- **Test Set Predictions:** Predicts 156/418 survivors (37.3%) in `titanic_stacked_submission.csv`, with 91.39% agreement against Kaggle's naive `gender_submission.csv` baseline.

---

## Repository Structure

```text
.
├── Dataset/                     # Titanic CSV files (train, test, gender_submission)
├── new_ml_updated (1).py        # Full pipeline: EDA, leak-free CV, stacking, depth analyses & exports
├── es678_project_template.tex   # 4-page ES 678 paper (NeurIPS 2026 single-column template)
├── fig_error_corr.pdf           # Residual error correlation heatmap across Level-1 models
├── fig_ablation_cal.pdf         # Probability calibration reliability curve (Brier score)
├── fig_shap_summary.pdf         # Top-10 Level-1 SHAP feature impacts
├── ML_project_2026.zip          # LaTeX source archive
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview, data access & reproducibility notes
