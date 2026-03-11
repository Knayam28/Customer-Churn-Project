# Customer Churn Prediction System

## Project Overview
Customer churn is a major challenge for subscription-based and e-commerce businesses. Predicting which customers are likely to leave allows companies to take proactive actions to retain them.

This project builds an **end-to-end machine learning pipeline** to predict customer churn using customer behavioral and demographic data. The model identifies customers at risk of churn and provides probability scores that can help businesses design targeted retention strategies.

The final model is deployed using **Streamlit**, allowing users to interactively input customer data and obtain churn predictions.

---

## Project Workflow

```
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
```

---

## Dataset

The dataset contains customer demographic and behavioral information from an e-commerce platform.

### Example Features
- Customer tenure  
- Order frequency  
- Payment methods  
- Device used  
- Customer satisfaction  
- Days since last order  

### Target Variable

```
Churn
0 → Customer stays
1 → Customer churns
```

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing values using **KNNImputer**
- Encoding categorical variables using **OneHotEncoder**
- Feature scaling using **StandardScaler**
- Building a preprocessing pipeline using **Scikit-learn Pipeline**
- Splitting the dataset into **training and testing sets**

---

## Handling Class Imbalance

Customer churn datasets are typically imbalanced because fewer customers churn compared to those who stay.

To address this issue:

- **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the minority class.
- This improves the model’s ability to correctly identify churned customers.

---

## Model Training

Multiple machine learning models were evaluated, including:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

Hyperparameters were optimized using **GridSearchCV** with cross-validation to identify the best-performing model.

---

## Model Evaluation

Evaluation metrics used:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  

Since churn prediction focuses on identifying customers who may leave, **Recall and F1 Score were prioritized**.

Example performance:

```
Best Model: Random Forest / XGBoost
F1 Score: ~0.78
ROC-AUC: Strong predictive performance
```

A **custom probability threshold** was also applied to better balance false positives and false negatives.

---

## Model Explainability (SHAP)

To understand how the model makes predictions, **SHAP (SHapley Additive exPlanations)** was used.

This helps explain:

- Feature importance
- Individual prediction behavior
- Which customers are most likely to churn

Explainable models improve trust and transparency in machine learning systems.

---

## Business Insights

Key insights from the analysis include:

- Customers with **short tenure** are more likely to churn.
- **Low order frequency** correlates with higher churn risk.
- **Long gaps since the last order** significantly increase churn probability.
- Certain **payment methods and device types** show different churn patterns.

These insights can help businesses design targeted **customer retention strategies**.

---

## Deployment

The churn prediction model is deployed using **Streamlit**.

Users can:

- Enter customer information
- Get churn prediction results
- View churn probability scores

---

## Tech Stack

### Programming Language
- Python

### Libraries
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn  
- SHAP  
- Matplotlib  
- Seaborn  

### Deployment
- Streamlit

---

## Project Structure

```
Customer-Churn-Project
│
├── notebooks
│   └── churn_analysis.ipynb
│
├── model
│   └── churn_model.pkl
│
├── app
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Clone the repository

```bash
git clone https://github.com/Knayam28/Customer-Churn-Project.git
cd Customer-Churn-Project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## Future Improvements

- Improve model performance using advanced ensemble techniques  
- Deploy the model using **Docker and cloud platforms (AWS)**  
- Build an API using **FastAPI**  
- Implement real-time prediction pipelines  

---

## Author

**Mayank Singh**

M.Tech Bioinformatics  
Delhi Technological University (DTU)

Aspiring Data Scientist | Machine Learning Enthusiast
