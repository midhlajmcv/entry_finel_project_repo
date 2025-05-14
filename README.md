# Lung Cancer Prediction using Machine Learning


## 📌 Overview

This project builds a machine learning-based classification system to predict the presence of lung cancer in patients. It includes detailed data preprocessing, feature engineering, and model evaluation to determine the most effective classifier.

## 🎯 Objectives

- Analyze patient data related to lung cancer.
- Handle missing values, outliers, and imbalanced data.
- Apply and compare multiple ML classification algorithms.
- Evaluate models using metrics like accuracy, precision, recall, and F1-score.
- Identify the best-performing model for reliable lung cancer prediction.

## 🧾 Dataset Description

The dataset (`lung cancer.csv`) includes both categorical and numerical health-related features:

### Categorical Features
- Gender  
- Smoking History  
- Family History  
- Treatment History  
- Mutation Type  
- Biopsy Result  
- Gene  

### Numerical Features
- Various health indicators (e.g., age, gene expressions, etc.)

### Target Variable
- **Lung Cancer**: Binary label (1 for cancer presence, 0 for absence)

## 🔧 Methods Used

### Data Cleaning & Preprocessing
- Removed duplicates  
- Imputed missing values using `SimpleImputer`  
- Detected outliers with boxplots  
- Balanced dataset using **SMOTE**

### Exploratory Data Analysis (EDA)
- Summary statistics  
- Distribution visualization  
- Outlier and imbalance identification

### Feature Selection & Scaling
- Used `SelectKBest` for feature selection  
- Standardized features with `StandardScaler`

### Model Training
Evaluated the following ML models:
- Logistic Regression  
- Support Vector Machine (SVC)  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Gradient Boosting  
- AdaBoost  
- Multi-Layer Perceptron (MLP)

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC-AUC Score

## 📊 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`  
- `scikit-learn`, `imblearn` (for SMOTE)

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lung-cancer-prediction.git
   cd lung-cancer-prediction
