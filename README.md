# 🚗 Claim Fraud Detection with Imbalanced Data Using Machine Learning

Insurance claim fraud is a costly problem in the industry, driving up losses and premiums. Travelers Insurance Company is focusing on first-party physical damage auto claims (i.e. claims for the policyholder’s own vehicle) from 2015–2016 to detect potential fraud. The challenge is that fraudulent claims are relatively rare, so a sophisticated detection approach is needed to address this imbalance issue. This project explores how machine learning can help flag fraudulent claims, providing insights into key risk factors and improving the efficiency of fraud investigations.

## 📌 Objective

To identify fraudulent insurance claims with high accuracy by developing a predictive model that flags claims as fraudulent (`1`) or not (`0`).

## 📂 Data

- **Source:** Travelers Insurance, covering 2015–2016 auto claims
- **Train Set:** 80% of 17,866 rows (after cleaning)
- **Test Set:** 20% of 17,866 rows
- **Target Variable:**  
  - `0`: Non-fraudulent claim  
  - `1`: Fraudulent claim
- **Issue:** Only 15.86% of claims are fraudulent → strong class imbalance

### 🔧 Preprocessing & Feature Engineering

- **Outliers**: Capped extreme driver ages
- **Transformations**: Applied log transform to `vehicle_price` 
- **Categorical Encoding**: One-hot encoding on variables like 'gender','accident_site', 'claim_month', 'claim_quarter'
- **Feature Selection**: Focused on informative variables

## 📊 Data Visualization

- Visuals highlight class imbalance and distribution of key predictors
- Included SHAP feature importance and class distribution comparisons

## 🧠 Models Used

We evaluated 8 classifiers under two setups:
1. Without class weighting
2. With class weighting (`class_weight='balanced'` or `scale_pos_weight`)

**Models**:
- Logistic Regression
- LDA
- KNN
- Random Forest
- XGBoost

## 🧪 Evaluation

- **Metric**: F1 Score (balances precision and recall)
- **Baseline Accuracy**: High due to imbalance, but not reliable
- **Best Model**: Logistic Regression with class weights  
  - **Precision**: 0.225  
  - **Recall**: 0.7478  
  - **F1 Score**: 0.3463  
  - **TP**: 424 | **FP**: 1458 | **FN**: 143 | **TN**: 1549

## 📌 Key Insights

- Class weighting drastically improved recall for fraud cases
- Ensemble models (like RF and XGB) performed well in balanced recall/precision tradeoffs
- Fraud detection is sensitive to class imbalance handling — underscoring the need for evaluation beyond accuracy
