# Customer Churn Prediction System

**Try the web app here:** [https://customer-churn-project-qhyextfbbmws4ffxgyj2vt.streamlit.app/]

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c387bc1-2500-4c23-b0d5-f49493215c52" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a1fc4f3a-b6cc-4916-9c7a-be2abab9a631" />

## Project Overview
Customer churn is a major challenge for subscription-based and e-commerce businesses. Predicting which customers are likely to leave allows companies to take proactive actions to retain them.

This project builds an **end-to-end machine learning pipeline** to predict customer churn using customer behavioral and demographic data. The model identifies customers at risk of churn and provides probability scores that can help businesses design targeted retention strategies.

The final model is deployed using **Streamlit**, allowing users to interactively input customer data and obtain churn predictions.

---

## Project Workflow

    Dataset
       ↓
    Data Cleaning & Preprocessing
       ↓
    Exploratory Data Analysis (EDA)
       ↓
    Feature Engineering
       ↓
    Handling Class Imbalance (SMOTE)
       ↓
    Model Training & Hyperparameter Tuning
       ↓
    Model Evaluation
       ↓
    Model Explainability (SHAP)
       ↓
    Deployment with Streamlit

---

## Dataset

The dataset contains customer demographic and behavioral information from an e-commerce platform.

### Example Features
* Customer tenure
* Order frequency
* Payment methods
* Device used
* Customer satisfaction
* Days since last order

### Target Variable

    Churn
    0 → Customer stays
    1 → Customer churns

---

## Data Preprocessing

The following preprocessing techniques were applied:

* Handling missing values using **KNNImputer** and **SimpleImputer**
* Encoding categorical variables using **OneHotEncoder**
* Feature scaling using **StandardScaler**
* Building a preprocessing pipeline using **Scikit-learn Pipeline**
* Splitting the dataset into **training and testing sets**

---

## Handling Class Imbalance

Customer churn datasets are typically imbalanced because fewer customers churn compared to those who stay. 

To address this issue:
* **SMOTE (Synthetic Minority Oversampling Technique)** was used within an `imblearn` pipeline to balance the minority class without causing data leakage.
* This improves the model’s ability to correctly identify actual churned customers.

---

## Model Training

Multiple machine learning models were evaluated, including Logistic Regression, Random Forest, and Gradient Boosting. 

The final architecture utilizes an **XGBoost Classifier**, optimized using cross-validation to capture complex, non-linear customer behaviors.

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

* **Programming Language:** Python
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn
* **Deployment:** Streamlit

---



## Future Improvements

* Improve model performance using advanced ensemble techniques
* Deploy the model using **Docker and cloud platforms (AWS)**
* Build an API using **FastAPI**
* Implement real-time prediction pipelines

---

## Author

**Mayank Singh** M.Tech Bioinformatics | Delhi Technological University (DTU)  
Aspiring Data Scientist 
