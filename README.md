# Predictive Modeling via Stacked Ensemble Learning: A Two-Tiered Classification Approach

**Course:** ES 678 — Machine Learning (Course Project)  
**Author:** Meet Rajendrakumar Jain (Roll No. 26210087)  
**Institution:** Indian Institute of Technology Gandhinagar  
**Colab Notebook:** (https://colab.research.google.com/drive/1u48dL64DrjkH9IoM9S7xI5p6pRicmsEZ#scrollTo=tQLMKivJ4328)

---

## Overview

This repository contains the source code, LaTeX paper source, generated figures, and reproducibility notes for predicting binary passenger survival on the Kaggle *Titanic: Machine Learning from Disaster* dataset (N_train = 891 labeled training rows, N_test = 418 unlabeled test rows) using a **two-level stacked ensemble**:

- **Baseline Model:** Default `RandomForestClassifier` (100 trees) evaluated on identical leak-free fold features.
- **Level-1 Base Learners (5 diverse models):** Random Forest, Extra Trees, AdaBoost, Gradient Boosting, and RBF-kernel SVM (with `StandardScaler` inside a fold-isolated `Pipeline`).
- **Level-2 Meta-Learner:** Shallow, heavily regularized **XGBoost** (`n_estimators=120`, `max_depth=2`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=1.0`, `reg_lambda=5.0`) trained strictly on 5-fold stratified Out-of-Fold (OOF) predicted probabilities.

### Architecture & Leak-Free Pipeline

```text
Raw Titanic CSV (N_train = 891, N_test = 418)
  └── 5-Fold Stratified Cross-Validation (SEED = 42)
        └── TitanicPreprocessor (fit() ONLY on each training fold -> zero leakage)
              ├── Level-1 Base Learners (5 diverse models -> OOF probability matrix Z in [0,1]^5)
              │     ├── Random Forest (300 trees, max_depth=6)
              │     ├── Extra Trees (300 trees, max_depth=6)
              │     ├── AdaBoost (200 estimators, lr=0.05)
              │     ├── Gradient Boosting (200 trees, max_depth=3, lr=0.03)
              │     └── Scaled RBF-SVM (StandardScaler Pipeline, C=1.5, Platt scaling)
              └── Level-2 Meta-Learner (Evaluated on the EXACT SAME 5-fold splits)
                    └── Regularized XGBoost (max_depth=2, α=1.0, λ=5.0) -> Final Survival Prob.
```

---

## Feature Engineering & Preprocessing (`TitanicPreprocessor`)

To prevent any train–test or cross-fold leakage, all statistical imputations (`Age` medians conditioned on `Title` and `Pclass`, `Fare` medians by `Pclass`, `Embarked` mode, and `TicketGroupSize` counts) are learned inside `.fit()` **exclusively on each training fold**:

- `Title`: Extracted from `Name` via regex; synonyms normalized (`Mlle/Ms` → `Miss`, `Mme` → `Mrs`) and uncommon honorifics grouped as `Rare`.
- `FamilySize` & `IsAlone`: `SibSp + Parch + 1`, plus binary indicator for solo travelers (`FamilySize == 1`).
- `HasCabin` & `Deck`: Binary cabin availability indicator and first-letter cabin deck (or `Unknown`).
- `LogFare`, `FarePerPerson`, & `TicketGroupSize`: `log(1 + Fare)` to compress skew, `Fare / max(FamilySize, 1)`, and shared-ticket group frequency.

---

## Main Results (5-Fold Stratified CV, `SEED=42`)

| Model | Type | CV Accuracy ± SD | 95% Bootstrap CI | ROC-AUC | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Default RF) | Baseline | 0.8114 ± 0.0301 | [0.787, 0.837] | 0.8750 | 0.7529 | 0.1318 |
| Random Forest (Tuned) | Base | 0.8406 ± 0.0124 | [0.817, 0.865] | 0.8716 | 0.7829 | 0.1229 |
| Extra Trees | Base | 0.8305 ± 0.0075 | — | 0.8681 | 0.7722 | 0.1275 |
| AdaBoost | Base | 0.8058 ± 0.0139 | — | 0.8573 | 0.7375 | 0.1482 |
| Gradient Boosting | Base | 0.8395 ± 0.0112 | — | 0.8807 | 0.7843 | 0.1188 |
| RBF-SVM (Scaled) | Base | 0.8272 ± 0.0185 | — | 0.8668 | 0.7688 | 0.1311 |
| **Stacked XGBoost** | **Meta** | **0.8316 ± 0.0103** | **[0.807, 0.856]** | **0.8902** | **0.7664** | **0.1204** |

- **vs. Baseline RF:** Stacked XGBoost improves 5-fold CV accuracy by **+2.02%** (McNemar exact *p* = 0.0356; 42 errors fixed vs. 24 introduced), achieves the **highest ROC-AUC (0.8902)** across all seven models, and substantially improves calibration (Log-Loss 0.3922 vs. 0.7953).
- **vs. Best Single Base Model (Tuned RF, 84.06%):** Fixes 7 errors vs. 15 regressions (McNemar *p* = 0.1338). While Tuned RF edges ahead at the fixed 0.50 threshold, Stacked XGBoost delivers superior ranking discrimination across thresholds and lower fold-to-fold variance.

---

## Graduate Depth Analyses

### 1. Meta-Learner Regularization Ablation (Overfitting Control)
A key modeling risk in stacking is the Level-2 meta-learner memorizing Level-1 OOF predictions. Strong L1/L2 regularization and shallow tree depth cut the overfitting gap from **0.1818** down to **0.0155**:

| Meta-Learner Configuration | Train Acc. | 5-Fold CV Acc. | Overfitting Gap |
| :--- | :---: | :---: | :---: |
| Unregularized Deep XGB (`depth=6, lr=0.3, reg=0`) | 0.9910 | 0.8092 | 0.1818 |
| Moderate XGB (`depth=4, lr=0.1, α=0.5, λ=1.0`) | 0.9268 | 0.8159 | 0.1108 |
| **Proposed Regularized XGB (`depth=2, lr=0.03, α=1.0, λ=5.0`)** | **0.8471** | **0.8316** | **0.0155** |

### 2. Diversity, Architecture, Calibration, & Explainability
- **Base-Model Residual Error Diversity:** Pairwise residual error correlation (*y* − *p̂*) remains high across all five base learners (*r* ∈ [0.91, 0.99]; RBF-SVM vs. trees *r* ∈ [0.92, 0.96] with 3.3%–6.8% pairwise disagreement), explaining why stacking primarily strengthens probability ranking (ROC-AUC) and calibration rather than delivering large thresholded accuracy leaps over the best single tree model.
- **Pure vs. Passthrough Stacking:** Pure OOF stacking (0.8316 CV) slightly outperforms passthrough stacking that appends 5 raw features (0.8305 CV), avoiding redundant feature memorization.
- **Mistake-Correction Audit:** Stacked XGBoost correctly classifies **15** passengers where unweighted majority voting of the five base models failed (primarily boundary subgroups such as third-class females and larger families).
- **SHAP & Permutation Importance:** Cross-algorithm permutation importance and Level-1 SHAP agree on `Title_Mr`, `Sex`, `Pclass`, `FarePerPerson`, and `Age`, while Level-2 SHAP shows XGBoost weights Gradient Boosting and Random Forest most heavily alongside SVM boundary adjustments.
- **Test Set Predictions:** Generates `titanic_stacked_submission.csv` predicting **156/418** survivors (37.3%), with **91.39%** agreement against Kaggle's naive `gender_submission.csv` baseline.

---

## Limitations

1. **Sample Size:** N = 891 labeled training rows restrict statistical power for separating small accuracy differences.
2. **Non-Random Missingness:** Missingness in `Age` (177 train / 86 test) and `Cabin` (687 train / 327 test) varies systematically by passenger class and survival outcome.
3. **Unlabeled Test Set & Non-Causal Scope:** Ground-truth labels for `test.csv` are withheld by Kaggle, and SHAP/permutation scores reflect predictive association rather than causal survival mechanisms.

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
```

---
## References & AI-Use Disclosure

- **Key Methods & Libraries:** Built with `scikit-learn`, `xgboost`, `shap`, `scipy`, `plotly`, `matplotlib`, and `seaborn`. Core methodology follows Wolpert (1992) for stacked generalization, Chen & Guestrin (2016) for XGBoost, and Lundberg & Lee (2017) for SHAP.
- **AI-Use Disclosure:** Generative AI tools (Gemini / ChatGPT) were used to assist with Python diagnostic plotting code, README documentation, and LaTeX manuscript formatting. All code execution, numerical results, statistical interpretations, and course requirements were independently verified by the author.
