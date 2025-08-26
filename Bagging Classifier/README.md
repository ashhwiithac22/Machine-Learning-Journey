# 🌳 Decision Tree & Bagging Classifier on Diabetes Prediction Dataset

This project demonstrates how to build and evaluate a **Decision Tree Classifier** and improve its performance using a **Bagging Classifier (Bootstrap Aggregating)** on a diabetes prediction dataset.

---

## 📌 Key Concepts

### 🔹 Bagging (Bootstrap Aggregating)
- **Ensemble learning** technique → combines multiple models to improve accuracy.  
- Core idea: create multiple training sets with replacement.  
- **Reduces variance**, making models less prone to overfitting.  
- For classification → final prediction by **majority voting**.  
- For regression → final prediction by **averaging predictions**.  

---

## ⚙️ Steps in the Project

1. **Exploratory Data Analysis (EDA)**
   - Checked missing values, class balance, and numerical distributions.  
   - Selected numerical columns: `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`.  

2. **Encoding Categorical Variables**
   - Converted `gender` into numerical values.  
   - Applied **OneHotEncoding** to `smoking_history`.  

3. **Train-Test Split**
   - Training: 80%  
   - Testing: 20%  

4. **Decision Tree Classifier**
   - Hyperparameter tuning with **GridSearchCV**:  
     ```json
     {'criterion': 'gini', 'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 2}
     ```
   - Best depth found = **9**  
   - Evaluated using **Confusion Matrix, Accuracy, and F1 Score**  

5. **Bagging Classifier**
   - Base model: **Decision Tree** with depth 9  
   - Number of estimators: **49 trees**  
   - Enabled **OOB (Out-of-Bag) scoring**  
   - Compared results with single Decision Tree  

---

## 📊 Results

### 🔹 Decision Tree
- **Best Parameters:** `{'criterion': 'gini', 'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 2}`  
- **Accuracy:** `0.9722`  
- **F1 Score:** `0.8088`  

### 🔹 Bagging Classifier
- **Accuracy:** `0.9726`  
- **F1 Score:** `0.8099`  
- **Confusion Matrix:** (plotted in orange during execution)

✅ Bagging provided a **slight performance boost**, reducing variance and stabilizing predictions.  

---

## 📂 Files

- `decisiontree_bagging.py` → Python script with full workflow  
- `diabetes_prediction_dataset.csv` → Dataset file  
- Generated plots:  
- **Confusion Matrix - Decision Tree**  
- **Confusion Matrix - Bagging**  

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/ashhwiithac22/NLP---Playground.git
cd NLP---Playground

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python "decisiontree_bagging.py"
