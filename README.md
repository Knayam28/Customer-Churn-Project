# Customer Churn Prediction System
## 🚀 Live Demo
**Try the web app here:** [https://customer-churn-project-qhyextfbbmws4ffxgyj2vt.streamlit.app/]

### Application Preview

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c387bc1-2500-4c23-b0d5-f49493215c52" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a1fc4f3a-b6cc-4916-9c7a-be2abab9a631" />

## Project Overview

Customer churn is one of the most critical challenges faced by e-commerce and subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones, making churn reduction a key business objective.

This project combines SQL analytics, customer segmentation, Tableau visualization, and machine learning to identify customers at risk of churn and support data-driven retention strategies.

The solution not only predicts which customers are likely to churn but also prioritizes high-value customers using RFM (Recency, Frequency, Monetary) segmentation. This enables businesses to focus retention efforts on customers whose loss would have the greatest revenue impact.


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

### Features

- Tenure
- Preferred Login Device
- Preferred Payment Mode
- City Tier
- Satisfaction Score
- Order Count
- Cashback Amount
- Complaint Status
- Day Since Last Order
- Number of Addresses

### Target Variable

**Churn**

- `0` → Customer Retained
- `1` → Customer Churned

---
## SQL Analytics & Business Insights

SQL was used to perform exploratory customer churn analysis and identify key behavioral patterns associated with customer churn.

### Analyses Performed

- Overall Churn Rate
- Churn by Gender
- Churn by City Tier
- Churn by Complaint Status
- Churn by Tenure Group
- Churn by Payment Mode
- Churn by Order Category
- Churn by Login Device
- Churn by Marital Status
- Churn by Cashback Segment

### Key Findings

- Customers with tenure below **3 months** showed the highest churn rate (**41.86%**).
- Customers who raised complaints exhibited significantly higher churn behavior.
- Customers using **Cash on Delivery (COD)** displayed higher churn rates compared to other payment methods.
- Customers purchasing from the **Mobile** category demonstrated elevated churn rates.
- Customer churn was strongly associated with engagement, satisfaction, and purchasing behavior.

---

## Customer Segmentation (RFM Analysis)

To identify customer value and support retention strategies, **RFM (Recency, Frequency, Monetary)** analysis was performed.

### RFM Metrics

| Metric | Description |
|----------|-------------|
| Recency | Days since last order |
| Frequency | Number of orders placed |
| Monetary | Cashback amount used as a spending proxy |

### Customer Segments

| Segment | Description |
|----------|-------------|
| VIP | Highly valuable and engaged customers |
| Loyal | Regular and consistent customers |
| At Risk | Customers showing declining engagement |
| Lost | Low-value and inactive customers |

### Business Impact

The segmentation was used as a business layer on top of churn prediction to:

- Identify high-value customers requiring retention efforts.
- Detect customers at risk of churning.
- Support targeted marketing and loyalty campaigns.
- Prioritize customer engagement strategies based on customer value.


## Tableau Dashboard
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2f5fdb87-a110-4fa7-88b1-5429c8457203" />

## Tableau Dashboard

Interactive Tableau dashboards were created to visualize:

- Overall Churn Trends
- Customer Demographics
- Tenure-Based Churn Analysis
- Payment Mode Analysis
- Product Category Analysis
- RFM Customer Segmentation
- Customer Retention KPIs

### Business Value

The dashboard enables stakeholders to:

- Identify high-risk customer groups.
- Understand churn drivers.
- Monitor customer retention performance.
- Discover revenue growth opportunities.
- Support data-driven decision-making.

---

## Data Preprocessing

The following preprocessing techniques were applied before model training.

### Missing Value Handling

| Data Type | Technique |
|------------|------------|
| Numerical Variables | KNNImputer |
| Categorical Variables | SimpleImputer |

### Feature Encoding

- OneHotEncoder was applied to categorical features.

### Feature Scaling

- StandardScaler was used for numerical variables.

### Pipeline Construction

Scikit-Learn **Pipelines** and **ColumnTransformers** were implemented to create a reproducible and scalable preprocessing workflow.

---

## Handling Class Imbalance

Customer churn datasets are typically imbalanced because the number of churned customers is much lower than retained customers.

### Technique Used

- SMOTE (Synthetic Minority Oversampling Technique)
- Applied only to training data
- Implemented within an imbalanced-learn pipeline
- Prevented data leakage during model training

### Benefits

- Improved detection of churned customers.
- Enhanced model robustness.
- Reduced bias toward the majority class.

---

## Model Training

### Algorithm

- XGBoost Classifier

### Hyperparameter Optimization

- RandomizedSearchCV

### Evaluation Strategy

Because missing a churning customer can lead to revenue loss, the following metrics were prioritized:

- Recall
- F1 Score

This ensured the model focused on identifying potential churners while maintaining a balance between precision and recall.
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

