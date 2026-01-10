# 🌲 Forest Cover Type Classification

## 📌 Project Overview
This project focuses on **classifying forest cover types** using environmental and geographical features.  
The workflow follows an **end-to-end machine learning pipeline**, starting from data exploration and statistical analysis to model training, evaluation, and artifact saving.

The objective is to **accurately predict forest cover types** using multiple classification algorithms and compare their performance.

---

## 📂 Dataset
- **Source:** Forest Cover Type dataset (CSV)
- **Target Variable:** `Cover_Type`
- **Features Include:**
  - Elevation, Aspect, Slope
  - Distance to hydrology, roadways, fire points
  - Hillshade measurements
  - Wilderness area and soil type indicators

✔ No missing values  
✔ No duplicate records  
✔ Zero values retained as valid physical measurements  

---

## 🔍 Exploratory Data Analysis (EDA)
- Dataset shape, info, and statistical summary
- Missing value and duplicate checks
- Correlation matrix
- Feature skewness and kurtosis analysis
- Box-Cox transformation testing for distribution improvement

---

## 🧪 Statistical Hypothesis Testing
Three types of hypothesis tests were performed to understand feature relationships:

- **Categorical vs Categorical:** Chi-Square Test  
- **Continuous vs Continuous:** Repeated T-Tests on sampled data  
- **Continuous vs Categorical:** One-Way ANOVA  

A final **feature interaction matrix** was created to summarize statistical relationships.

---

## 🛠️ Data Preprocessing

### Label Encoding
- Target variable (`Cover_Type`) encoded using `LabelEncoder`
- Encoder saved for inference consistency

Saved file:
models/cover_type_encoder.pkl

### Standardization
- `StandardScaler` applied to numerical features
- Scaler saved for reuse

Saved file:
models/scaler.pkl

### Train-Test Split
- 80% Training / 20% Testing
- Stratified split to preserve class distribution

---

## ⚖️ Class Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)** applied on training data
- Helps improve minority class prediction performance

---

## 🤖 Models Trained
The following classifiers were trained and evaluated:

Random Forest – Ensemble of decision trees  
Decision Tree – Tree-based classifier  
Logistic Regression – Linear classification baseline  
K-Nearest Neighbors – Distance-based classifier  
XGBoost – Gradient boosting (multi-class)  

---

## 📊 Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

Evaluation was done on **unseen test data** using accuracy score and classification report.

---

## 🏆 Key Observations
- Tree-based and boosting models performed significantly better
- SMOTE improved recall for minority classes
- Proper scaling was crucial for distance-based models (KNN, Logistic Regression)

---

## 📁 Project Structure
cover_type.csv  
models/  
 ├── cover_type_encoder.pkl  
 ├── scaler.pkl  
forest_cover_classification.ipynb  
README.md  

---

## 🚀 How to Run
Install dependencies:
pandas, numpy, scikit-learn, imbalanced-learn, xgboost, joblib

Open the notebook and run all cells sequentially.

---

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearch / Optuna)
- Feature selection using importance scores
- Model deployment using Streamlit or FastAPI
- SHAP-based model explainability

---

## 👤 Author
**Naveen Kumar**  
Machine Learning | Data Science | Analytics  
