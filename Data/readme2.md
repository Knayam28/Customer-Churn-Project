# Customer Churn Prediction System

## Overview
A production-style, end-to-end machine learning pipeline to predict customer churn for an e-commerce platform.  
This repo demonstrates data cleaning, feature engineering, model training & tuning, explainability (SHAP), and deployment with **Streamlit** so recruiters and stakeholders can interact with the model.

---

## Key Features
- End-to-end pipeline: preprocessing → modeling → evaluation → deployment  
- Handles class imbalance with **SMOTE**  
- Model explainability using **SHAP** (global & local explanations)  
- Deployable Streamlit app for interactive predictions  
- Clear separation of code, notebooks, model artifacts, and app

---

## Quick Results (example)
| Metric | Value |
|---|---:|
| Model (best) | Random Forest / XGBoost |
| F1 Score (test) | ~0.78 |
| ROC-AUC | High (see notebook for exact value) |

> These are example summary metrics. See the notebook and `model/` folder for exact training logs and saved artifacts.

---

## Dataset
Contains customer demographic and behavioral features such as:
- Tenure, order frequency, days since last order  
- Device type, payment method, customer satisfaction  
- Target: `Churn` (0 = stay, 1 = churn)

> The raw dataset file (if included) should be placed in `data/` or a data-loading step should be added to the notebooks/scripts.

---

## Preprocessing & Engineering
- Missing values handled with **KNNImputer**  
- Categorical variables encoded with **OneHotEncoder**  
- Numerical features scaled using **StandardScaler**  
- Feature engineering to capture recency, frequency, and tenure signals  
- Pipeline built with `sklearn.pipeline.Pipeline` for reproducibility

---

## Handling Class Imbalance
- **SMOTE** is used on the training set to synthesize minority samples and reduce bias toward the majority class, improving recall on churn cases.

---

## Modeling
- Models explored: Logistic Regression, Random Forest, Gradient Boosting, XGBoost  
- Hyperparameter tuning via **GridSearchCV** with cross-validation  
- Custom probability thresholding applied for business-oriented tradeoffs between false positives and false negatives

---

## Explainability
- **SHAP** used to:
  - Identify global feature importance
  - Explain individual predictions (local explanations)
- SHAP plots are saved in the `reports/` or `notebooks/` folder (see notebook).

---

## Deployment
Interactive demo using Streamlit:

```bash
# from repository root
pip install -r requirements.txt
streamlit run app/streamlit_app.py
