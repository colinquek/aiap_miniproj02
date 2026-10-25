# Refactored Hotel Customer No-Show Prediction Pipeline

## Overview
This repository provides an end-to-end, production-grade machine learning pipeline designed to predict hotel customer no-shows (`no_show = 1`). By accurately identifying high-risk reservations, hotel management can implement proactive retention strategies, targeted deposit policies, and optimized overbooking strategies to minimize revenue loss.

This codebase has been refactored to address all structural, analytical, and software engineering feedback from the project evaluation report.

---

## Key Architecture & Features

### 1. Hypothesis-Driven Exploratory Data Analysis (EDA)
Located in `notebooks/eda_analysis.py`, the EDA follows a 6-phase analytical narrative:
* **Systematic Structural Inspection**: Full `info()`, `describe().T`, and a documented **20% stratified sampling protocol** for rapid prototyping.
* **Pre-Imputation Missingness Audit**: Visualizes null counts prior to cleaning and proves that room missingness is statistically significant ($p < 0.05$).
* **Subgroup Analysis**: Evaluates no-show rates across booking platforms, branches, and customer types against the **global average benchmark** using Chi-Square and Cramér's V tests.
* **Chronological Seasonality**: Plots monthly arrival trends in true calendar order and explicitly documents date field metadata limits.
* **Interaction Engineering & VIF Audit**: Engineers `price_per_night`, `total_guests`, and `first_time_x_price`, auditing multicollinearity via Variance Inflation Factors (VIF).
* **Feature Importance Closure**: Extracts permutation feature importances on validation data to confirm EDA hypotheses.

### 2. Configuration-Driven & Reproducible Machine Learning Pipeline
* **Centralized YAML Config**: `config.yaml` serves as the single source of truth for paths, split ratios, feature columns, and hyperparameter grids.
* **Schema Validation**: Automated runtime schema validation (`validate_schema`) ensuring missing or malformed columns fail early.
* **Handling Class Imbalance**: Incorporates `class_weight='balanced'` in estimators and uses stratified splitting.
* **Diverse Model Suite**: Compares Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, and Gradient Boosting.
* **Optimal Threshold Tuning**: Automatically optimizes the classification probability threshold on validation set F1-score rather than assuming a default 0.5 cutoff.
* **Rich Evaluation Suite**: Reports Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, and Confusion Matrix.
* **Batch Inference Entry Point**: `batch_predict.py` enables automated batch scoring on new unseen reservations.

### 3. Software Engineering Safeguards
* **Vectorized Feature Engineering**: Eliminates slow row-wise `.apply()` loops in stay duration and pricing calculations.
* **Safe Error Handling & Structured Logging**: Uses specific exception types and standard Python `logging` with timestamps and log files (`pipeline.log`).
* **Automated Unit & Integration Test Suite**: Includes `unittest` suites (`tests/`) verifying schema validation, price parsing, vectorized stay durations, and end-to-end pipeline execution.

---

## Directory Layout
```text
mini_project_02_reworked/
├── config.yaml                    # Central configuration file
├── main.py                        # Main training orchestration script
├── batch_predict.py               # Batch inference entry point
├── requirements.txt               # Pinned library dependencies
├── README.md                      # Project documentation
├── notebooks/
│   └── eda_analysis.py            # Complete 6-phase EDA script
├── src/
│   ├── __init__.py
│   ├── data_preparation.py        # Vectorized cleaning & schema validation
│   ├── model_training.py          # Model selection, tuning, & persistence
│   └── utils.py                   # Logging & reproducibility utilities
└── tests/
    ├── __init__.py
    ├── test_data_preparation.py   # Unit tests for preprocessing
    └── test_pipeline.py          # Integration test for training pipeline
```

---

## Quick Start Guide

### 1. Setup Environment & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit & Integration Tests
```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py"
```

### 3. Execute End-to-End Training Pipeline
```bash
python3 main.py config.yaml
```

### 4. Run Batch Predictions on New Data
```bash
python3 batch_predict.py ./data/mini_project_2_data.csv ./data/predictions.csv config.yaml
```

---

## Final Executive Takeaways & Business Impact
1. **Top Risk Factors**: Permutation feature importance confirms that **`price_per_night`** and **`first_time_customer`** status are the single strongest predictors of customer no-shows.
2. **High-Risk Segment**: First-time customers booking via third-party platforms for high price-per-night rooms exhibit the highest operational no-show risk.
3. **Recommended Business Action**: Implement a **20% non-refundable deposit policy** or automated SMS check-in reminders 48 hours prior to arrival for reservations matching this high-risk profile.
