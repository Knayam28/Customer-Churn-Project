# Customer Churn Prediction System
## 🚀 Live Demo
**Try the web app here:** [https://customer-churn-project-qhyextfbbmws4ffxgyj2vt.streamlit.app/]

### Application Preview

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c387bc1-2500-4c23-b0d5-f49493215c52" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a1fc4f3a-b6cc-4916-9c7a-be2abab9a631" />

## Project Overview
Project Overview

Customer churn is one of the most critical challenges faced by e-commerce and subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones, making churn reduction a key business objective.

This project combines SQL analytics, customer segmentation, Tableau visualization, and machine learning to identify customers at risk of churn and support data-driven retention strategies.

The solution not only predicts which customers are likely to churn but also prioritizes high-value customers using RFM (Recency, Frequency, Monetary) segmentation. This enables businesses to focus retention efforts on customers whose loss would have the greatest revenue impact.

Interactive dashboard built using Tableau to analyze customer churn patterns, customer segments, and business risk factors.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8382c506-023d-4c29-9863-cf8fb062e8c5" />

---

## Project Workflow

    Dataset
       ↓
    SQL Analysis
       ↓
    RFM Customer segmentation
       ↓
    Tableau Dashboard Development
       ↓
    Data Cleaning & Preprocessing
       ↓
    Handling Class Imbalance (SMOTE)
       ↓
    XGBoost Model Training & Hyperparameter Tuning
       ↓
    Model Evaluation
       ↓
    SHAP Explainability
       ↓
    High-Value Customer Retention Prioritizationg
       ↓
    Streamlit Deployment
---

## Dataset

The dataset contains customer demographic, transactional, and behavioral information from an e-commerce platform.

Example Features
Tenure
Preferred Login Device
Preferred Payment Mode
City Tier
Satisfaction Score
Order Count
Cashback Amount
Complaint Status
Day Since Last Order
Number of Addresses
Target Variable

Churn

0 → Customer Retained
1 → Customer Churned

---
## SQL Analytics & Business Insights

SQL was used to perform exploratory customer churn analysis and identify key behavioral patterns associated with churn.

Analyses Performed
Overall Churn Rate
Churn by Gender
Churn by City Tier
Churn by Complaint Status
Churn by Tenure Group
Churn by Payment Mode
Churn by Order Category
Churn by Login Device
Churn by Marital Status
Churn by Cashback Segment

Key Findings
Customers with tenure below 3 months showed the highest churn rate (41.86%).
Customers with complaints exhibited significantly higher churn behavior.
Cash on Delivery users displayed higher churn rates compared to other payment methods.
Mobile category customers demonstrated elevated churn rates.
Customer churn was strongly associated with customer engagement and purchasing behavior.

---
## Customer Segmentation (RFM Analysis)

To identify customer value, RFM (Recency, Frequency, Monetary) analysis was performed.

RFM Metrics

Recency - Days since last order

Frequency - Number of orders placed

Monetary - Cashback amount used as a spending proxy

Customer Segments
Segment	Description
VIP	: Highly valuable and engaged customers
Loyal :	Regular and consistent customers
At Risk : Customers showing declining engagement
Lost : Low-value and inactive customers

The segmentation was used as a business layer on top of churn prediction to prioritize customer retention efforts.

--

## Tableau Dashboard
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2f5fdb87-a110-4fa7-88b1-5429c8457203" />

Interactive Tableau dashboards were created to visualize:

Overall Churn Trends
Customer Demographics
Tenure-Based Churn Analysis
Payment Mode Analysis
Product Category Analysis
RFM Customer Segmentation
Customer Retention KPIs

The dashboard enables stakeholders to quickly identify high-risk customer groups and business opportunities.

---

## Data Preprocessing

The following preprocessing techniques were applied:

Missing Value Handling
KNNImputer for numerical variables
SimpleImputer for categorical variables
Encoding
OneHotEncoder for categorical features
Scaling
StandardScaler for numerical features
Pipeline Construction

Scikit-Learn Pipelines and ColumnTransformers were used to ensure reproducible preprocessing workflows.

---

## Handling Class Imbalance

Customer churn datasets are typically imbalanced because fewer customers churn compared to customers who stay.

To address this issue:

SMOTE (Synthetic Minority Oversampling Technique) was applied within an imbalanced-learn pipeline.
Oversampling was performed only on training data to prevent data leakage.
This improved the model's ability to correctly identify churned customers.

---

## Model Training

Algorithm - XGBoost Classifier

Hyperparameter Optimization done using RandomizedSearchCV

Because missing a churning customer is costly, Recall and F1 Score were prioritized during model selection.

---

## Model Evaluation

Evaluation metrics used: Accuracy, Precision, Recall, F1 Score, ROC-AUC Score, and Confusion Matrix. 

Since churn prediction focuses on identifying customers who may leave, **Recall and F1 Score** were prioritized to minimize False Negatives.

<img width="584" height="484" alt="image" src="https://github.com/user-attachments/assets/494f24c2-a3d1-4afa-8fbb-01b52c620ef3" />

---

## Model Explainability (SHAP)

To understand how the model makes predictions, **SHAP (SHapley Additive exPlanations)** was used. This ensures the model's decisions are transparent and actionable for business stakeholders by explaining feature importance and individual prediction behavior.

<img width="785" height="933" alt="image" src="https://github.com/user-attachments/assets/941e47fd-80f6-483b-89a1-251438e967fc" />

---

## Business Insights & Actionable Outcomes

Key insights from the analysis include:

* Customers with **short tenure** are more likely to churn.
* **Low order frequency** correlates with higher churn risk.
* **Long gaps since the last order** significantly increase churn probability.
* **Targeted Retention Pipeline:** The system goes beyond basic prediction by cross-referencing churn probabilities with customer segmentation. It automatically flags at-risk **'VIP' and 'Loyal'** customers and exports a prioritized `high_value_customers_at_risk.csv` report for immediate marketing intervention.

---

## Deployment

The churn prediction model is deployed using **Streamlit**. Users can enter customer information, get churn prediction results, and view churn probability scores in real time.

---

## Tech Stack

* **Programming Language:** Python, SQL
* **Data Analytics:** MySQL, Tableau
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn, Tableau
* **Deployment:** Streamlit

