# 🛒 E-Commerce Customer Churn: End-to-End Data Science Solution

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([INSERT_YOUR_LIVE_STREAMLIT_LINK_HERE])
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)

## 📌 Executive Summary
Customer retention is highly cost-effective compared to customer acquisition. This project delivers an end-to-end data science solution designed to identify e-commerce customers at risk of churning. 

Beyond simply predicting churn, this pipeline goes a step further by **segmenting high-value customers ('VIP' and 'Loyal')** and generating automated, prioritized intervention lists for marketing and customer success teams to act upon immediately.

**Try the live web application:** [Insert Streamlit App Link Here]

---

## 🏢 The Business Value & Actionable Insights
Machine learning is only as good as the business decisions it drives. This project culminates in an automated reporting pipeline that:
1. **Predicts** the probability of churn for every user in the database.
2. **Cross-references** predictions with customer segmentation data.
3. **Isolates** users flagged as 'VIP' or 'Loyal' who have a high churn probability.
4. **Exports** a targeted `high_value_customers_at_risk.csv` report, sorted by highest risk, allowing retention teams to allocate their budget (e.g., personalized discounts) exactly where it matters most.

---

## 🔬 The Data Science Lifecycle

### 1. Exploratory Data Analysis (EDA) & Feature Engineering
* Investigated behavioral features (e.g., Tenure, App Usage) and demographic data (e.g., CityTier, Gender).
* Engineered RFM (Recency, Frequency, Monetary) metrics to establish clear customer segments.
* Identified key drivers of churn prior to modeling to inform feature selection.

### 2. Robust Data Preprocessing
* **Missing Value Imputation:** Utilized `KNNImputer` for complex numerical distributions and `SimpleImputer` for categorical data to prevent data loss.
* **Feature Scaling & Encoding:** Applied `StandardScaler` to normalize distributions and `OneHotEncoder` for categorical variables.
* **Class Imbalance Resolution:** The dataset exhibited a severe imbalance (83.16% retained vs. 16.84% churned). Applied **SMOTE** (Synthetic Minority Over-sampling Technique) inside an `imblearn` pipeline to synthesize the minority class, preventing the model from becoming biased toward the majority class without causing data leakage.

### 3. Predictive Modeling (XGBoost)
* Architected a predictive pipeline utilizing the **XGBoost Classifier**.
* XGBoost was selected over traditional ensemble methods (like Random Forest) for its superior handling of non-linear relationships and internal regularization.
* Evaluated the model prioritizing **Recall and F1-Score** over standard accuracy, ensuring that the system minimizes False Negatives (missing a customer who is actually about to churn).

### 4. Model Explainability (SHAP)
To ensure transparency for business stakeholders, **SHAP (SHapley Additive exPlanations)** was integrated.
* *[Action: Insert your SHAP summary plot image here - e.g., `![SHAP Plot](images/shap_summary.png)`]*
* The SHAP analysis clearly identifies *why* a specific customer is flagged, highlighting features like short tenure or specific interaction patterns, allowing for highly personalized retention emails.

---

## 🛠️ Technology Stack & Architecture
* **Core:** Python, Pandas, NumPy
* **Machine Learning Pipeline:** Scikit-Learn, XGBoost, Imbalanced-Learn
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn
* **Deployment & MLOps:** Streamlit Community Cloud (Monolithic Architecture)

---

## 💻 Local Installation & Usage

To run this pipeline and the web app on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Knayam28/Customer-Churn-Project.git](https://github.com/Knayam28/Customer-Churn-Project.git)
   cd Customer-Churn-Project
