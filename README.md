# loan-risk-prediction

This project builds a machine learning pipeline to predict loan default risk using historical loan application and repayment-related features. The workflow covers data auditing, missing value analysis, leakage removal, feature engineering, encoding, scaling, baseline modeling, and model improvement using XGBoost.

## Project Overview

The goal of this project is to identify whether a borrower is likely to default on a loan based on information available at or near the time of application.

The notebook focuses on a practical credit-risk modeling workflow:

- Loading and inspecting a large loan dataset
- Auditing missing values and data quality
- Removing non-predictive and data leakage columns
- Handling categorical and numerical features
- Applying encoding and feature scaling
- Training a baseline Logistic Regression model
- Improving performance with XGBoost
- Evaluating models using classification metrics, confusion matrix, and ROC-AUC

## Dataset Summary

The dataset used in this project contains:

- **145 original columns**
- A large number of missing and sparse fields
- Loan-related features such as:
  - `loan_amnt`
  - `funded_amnt`
  - `term`
  - `int_rate`
  - `installment`
  - `grade`
  - `sub_grade`
  - `home_ownership`
  - `annual_inc`
  - `verification_status`
  - `purpose`
  - `dti`
  - `loan_status`

During preprocessing:

- **58 columns** with more than **50% missing values** were removed
- Rows were filtered to retain valid target classes
- Final retained dataset size after filtering: **1,303,638 rows**
- Final modeling dataset shape after encoding: **(1,303,638, 122)**

## Key Preprocessing Steps

### 1. Data Audit
The notebook begins by inspecting dataset structure, column names, and missing values to identify unusable or problematic fields.

### 2. Missing Value Handling
Columns with excessive null values were removed. Additional missing values in retained columns were handled during cleaning.

### 3. Data Leakage Removal
A major part of the workflow is removing features that would not be available at prediction time. Examples include repayment-related or post-loan fields such as:

- `last_pymnt_amnt`
- `recoveries`
- `total_pymnt`
- `out_prncp`

This avoids “cheating” and makes the model more realistic for real-world deployment.

### 4. Feature Selection
The notebook keeps features that are relevant to borrower risk and excludes:

- highly sparse columns
- IDs and URLs
- text-heavy fields requiring NLP
- future information / leakage columns

### 5. Categorical Encoding
Categorical variables were converted using one-hot encoding. Encoded features include fields such as:

- `term`
- `grade`
- `sub_grade`
- `home_ownership`
- `verification_status`
- `purpose`
- `initial_list_status`
- `application_type`
- `disbursement_method`

### 6. Outlier Handling and Scaling
The notebook also discusses handling outliers and scaling numerical variables, especially for finance-related columns such as income and debt-to-income ratio.

## Models Used

### Logistic Regression (Baseline)
A baseline Logistic Regression model was trained first to establish a reference point.

**Baseline performance:**
- Accuracy: **0.57**
- ROC-AUC: **0.5835**

### XGBoost
An XGBoost classifier was then trained to improve predictive performance.

**XGBoost performance:**
- Accuracy: **0.65**
- ROC-AUC: **0.7260**

## Results

### Logistic Regression
```text
Accuracy: 0.57
ROC-AUC: 0.5835
