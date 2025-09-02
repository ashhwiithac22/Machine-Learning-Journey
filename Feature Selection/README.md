# 🔍 Feature Selection in Machine Learning

## 📌 Overview
This project demonstrates **different feature selection methods** (Filter, Wrapper, and Embedded techniques) applied to a **diabetes prediction dataset** using **Logistic Regression**.  

Feature selection helps in:
- Removing irrelevant or redundant features  
- Reducing noise in the dataset  
- Improving model interpretability  
- Achieving better performance with fewer features  

---

## 🧾 Feature Selection Methods

### 1. **Filter Methods**
- Select features independently based on statistical tests.  
- **Chi-Square Test**
  - Works for categorical features.
  - High value → feature and target are dependent (important).  
  - Low value → feature and target are independent (not useful).  
- **ANOVA (F-test)**
  - Works for numerical features.
  - High F-score → good discriminative feature.  

### 2. **Wrapper Methods**
- Evaluate subsets of features by actually training a model.
- Uses **RFE (Recursive Feature Elimination)**:  
  - Iteratively removes least significant features.  
  - Keeps the most important ones.  

### 3. **Embedded Methods**
- Feature selection happens during model training.  
- **Lasso Regression (L1 Regularization)**:  
  - Shrinks some feature coefficients to 0.  
  - Only non-zero coefficient features are retained.  

---

## ⚙️ Implementation Details
- **Dataset:** `diabetes_prediction_dataset.csv`  
- **Model:** Logistic Regression  
- **Libraries Used:**
  - `pandas`
  - `scikit-learn`
  - `numpy`

---
## 📊 Results
```
Chi-Square Selected Features: ['age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']
Chi-Square Accuracy: 0.9588

ANOVA Selected Features: ['age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']
ANOVA Accuracy: 0.9588

RFE Selected Features: ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level']
Wrapper (RFE) Accuracy: 0.9419

Lasso Selected Features: ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
Embedded (Lasso regression) Accuracy: 0.9587

Best Feature Selection Method: Chi-Square with Accuracy = 0.9588
```

---

## ▶️ How to Run

1. Install dependencies:
```bash
   pip install pandas scikit-learn
  ``` 
2. Place diabetes_prediction_dataset.csv in the working directory.

3.Run the script:
```
python feature_selection.py
```
4.Compare different feature selection methods.
---
### 📚 Key Learnings

- Filter methods are fast and easy but ignore feature interactions.
- Wrapper methods consider feature interactions but are computationally expensive.
- Embedded methods strike a balance by performing selection during model training.

✅ Best performance here was achieved with Chi-Square (Accuracy = 95.88%).
