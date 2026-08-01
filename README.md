# 🏦 Bank Account Fraud Detection

An end-to-end Machine Learning project for detecting fraudulent bank account applications using the **Bank Account Fraud (BAF) Dataset Suite from NeurIPS 2022**.

The project explores data preprocessing, feature engineering, class imbalance, multiple machine learning algorithms, model evaluation, explainability, and deployment using Streamlit.

---

## 🚀 Demo
 **Try the application here:**

**https://bank-account-fraud-detection-q4jcnzi5hph4pxit5uzdee.streamlit.app**
The project includes a Streamlit web application where users can enter customer information and receive a fraud-risk prediction.

The application provides:

- Fraud probability
- Risk classification
- Risk-based recommendation
- Customer input summary

---

## 📊 Dataset

This project uses the:

**Bank Account Fraud Dataset Suite (NeurIPS 2022)**

Dataset:
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

The dataset suite was introduced at NeurIPS 2022 and contains six synthetic bank account fraud datasets designed for evaluating Machine Learning and fair Machine Learning methods.

Each dataset contains:

- 1,000,000 applications
- 30 realistic features
- A `month` feature representing temporal information
- A highly imbalanced fraud target
- Protected attributes such as age group, employment status, and income percentage

The datasets were designed to reflect realistic challenges in fraud detection, including class imbalance, bias, temporal distribution shifts, and privacy preservation.

---

## 🎯 Problem

Fraud detection is a highly imbalanced classification problem.

In this dataset, fraudulent applications represent only a small fraction of all applications.

Therefore, accuracy alone is not a reliable metric.

For example, a model could achieve very high accuracy simply by predicting almost every application as legitimate while missing most fraudulent applications.

Because of this, this project focuses particularly on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 🛠️ Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Categorical Encoding
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Class Imbalance Analysis
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Threshold Analysis
   ↓
SHAP Explainability
   ↓
Final XGBoost Model
   ↓
Streamlit Deployment
