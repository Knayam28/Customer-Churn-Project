# E-Commerce Customer Churn Prediction 🛒📊

## 📌 Overview
This project predicts the likelihood of e-commerce customers churning using machine learning. By identifying at-risk customers—specifically targeting high-value segments like 'VIP' and 'Loyal' customers—businesses can proactively implement retention strategies. The final output generates a structured list of at-risk users, ranked by their churn probability score for immediate marketing intervention.

## 🚀 Live Demo
**Try the web app here:** [https://customer-churn-project-qhyextfbbmws4ffxgyj2vt.streamlit.app/]

## 🧠 Methodology & Architecture
* **Data Preprocessing:** Handled missing values using robust techniques (`SimpleImputer` and `KNNImputer`), encoded categorical variables, and applied `StandardScaler` to normalize numerical features.
* **Class Imbalance Handling:** The dataset exhibited a significant class imbalance, with ~83.16% retained customers and ~16.84% churned customers. Addressed this by implementing **SMOTE** (Synthetic Minority Over-sampling Technique) within an `imblearn` pipeline to synthesize minority class examples without data leakage.
* **Modeling:** Trained and optimized an **XGBoost Classifier** (`XGBClassifier`) to capture complex, non-linear customer behaviors and transaction patterns.
* **Actionable Business Insights:** Engineered a post-prediction pipeline that filters for users predicted to churn who also belong to the 'VIP' or 'Loyal' segments. The pipeline automatically exports a `high_value_customers_at_risk.csv` report, sorted by highest `Churn_Probability_Score`.
* **Deployment:** Built a monolithic Streamlit dashboard to provide fast, real-time predictions on the cloud.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Pandas, NumPy
* **Model Explainability & Visualization:** SHAP, Matplotlib, Seaborn
* **Deployment & UI:** Streamlit

