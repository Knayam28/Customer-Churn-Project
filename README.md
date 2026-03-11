# E-Commerce Customer Churn Prediction 🛒📊

## 📌 Overview
This project predicts the likelihood of e-commerce customers churning using machine learning. By identifying at-risk customers—specifically targeting high-value segments like 'VIP' and 'Loyal' customers—businesses can proactively implement retention strategies. The final output generates a structured list of at-risk users, ranked by their churn probability score for immediate marketing intervention.

## 🚀 Live Demo
**Try the web app here:** [https://customer-churn-project-qhyextfbbmws4ffxgyj2vt.streamlit.app/]

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c387bc1-2500-4c23-b0d5-f49493215c52" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a1fc4f3a-b6cc-4916-9c7a-be2abab9a631" />

## 🧠 Methodology & Architecture
* **Data Preprocessing:** Handled missing values using robust techniques (`SimpleImputer` and `KNNImputer`), encoded categorical variables, and applied `StandardScaler` to normalize numerical features.
* **Class Imbalance Handling:** The dataset exhibited a significant class imbalance, with ~83.16% retained customers and ~16.84% churned customers. Addressed this by implementing **SMOTE** (Synthetic Minority Over-sampling Technique) within an `imblearn` pipeline to synthesize minority class examples without data leakage.
* **Modeling:** Trained and optimized an **XGBoost Classifier** (`XGBClassifier`) to capture complex, non-linear customer behaviors and transaction patterns.
* **Actionable Business Insights:** Engineered a post-prediction pipeline that filters for users predicted to churn who also belong to the 'VIP' or 'Loyal' segments. The pipeline automatically exports a `high_value_customers_at_risk.csv` report, sorted by highest `Churn_Probability_Score`.
* **Deployment:** Built a monolithic Streamlit dashboard to provide fast, real-time predictions on the cloud.

## 📊 Model Performance & Explainability

### Confusion Matrix
The confusion matrix demonstrates the model's ability to correctly identify at-risk customers, minimizing false negatives to ensure high-value churners are not missed.

<img width="584" height="484" alt="image" src="https://github.com/user-attachments/assets/494f24c2-a3d1-4afa-8fbb-01b52c620ef3" />

### Feature Importance (SHAP)
To ensure the model's decisions are transparent and actionable for the business, SHAP values were calculated. The summary plot below highlights the top features driving customer churn, such as Tenure and recent engagement metrics.

<img width="785" height="933" alt="image" src="https://github.com/user-attachments/assets/941e47fd-80f6-483b-89a1-251438e967fc" />

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Pandas, NumPy
* **Model Explainability & Visualization:** SHAP, Matplotlib, Seaborn
* **Deployment & UI:** Streamlit

