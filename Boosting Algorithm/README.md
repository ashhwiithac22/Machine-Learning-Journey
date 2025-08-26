# 🚀 Boosting Algorithms for Diabetes Prediction

- This project demonstrates the use of **Boosting Algorithms (AdaBoost, Gradient Boosting, and XGBoost)** to predict diabetes from a medical dataset.
- The workflow includes **data preprocessing, model training, evaluation, and performance comparison**.

---

## 📖 What is Boosting?

- **Boosting**: An ensemble technique that combines multiple **weak learners** into a **strong learner**.  
- Works **sequentially**, focusing more on misclassified samples in each iteration.  
- **Bagging vs Boosting**:
  - Bagging → Runs learners **in parallel** (e.g., Random Forest).  
  - Boosting → Runs learners **sequentially**, correcting mistakes of previous learners.  

### Types of Boosting:
- **AdaBoost (Adaptive Boosting)**  
  - Assigns equal weights initially.  
  - Increases weights of misclassified samples, forcing the next weak learner to focus on them.  

- **Gradient Boosting**  
  - Builds models sequentially by minimizing error using **gradient descent**.  
  - Each new learner corrects errors made by the previous model.  

- **XGBoost (Extreme Gradient Boosting)**  
  - An optimized implementation of Gradient Boosting.  
  - Adds **regularization** to reduce overfitting.  
  - Highly efficient and widely used in ML competitions.  

---

## ⚙️ Steps Performed in the Code

1. **Import Libraries**  
   - Used `sklearn` for preprocessing and models, `xgboost` for XGBoost, and `matplotlib` for visualization.  

2. **Load Dataset**  
   - Loaded `diabetes_prediction_dataset.csv`.  
   - Removed rows with missing `diabetes` labels.  
   - Separated features (`X`) and target (`y`).  

3. **Preprocessing**  
   - **Numerical columns** → Imputed missing values with **mean**.  
   - **Categorical columns** → Imputed missing values with **most frequent** + applied **OneHotEncoding**.  
   - Used `ColumnTransformer` + `Pipeline` to combine steps.  

4. **Train-Test Split**  
   - Split dataset into **80% training** and **20% testing**.  

5. **Model Training & Evaluation**
   - **AdaBoost**: Used Decision Trees as weak learners (50 estimators).  
   - **Gradient Boosting**: Default classifier from `sklearn`.  
   - **XGBoost**: Used `XGBClassifier` with `logloss` evaluation metric.  

6. **Performance Metrics**  
   - **Accuracy**  
   - **F1 Score** (better metric for imbalanced datasets)  
   - **Confusion Matrix** (visualized with `ConfusionMatrixDisplay`)  

7. **Model Comparison**  
   - Created a **bar chart** comparing Accuracy and F1 Score across AdaBoost, Gradient Boosting, and XGBoost.  

---

## 📊 Results

### 🔹 Accuracy & F1 Score
| Model             | Accuracy | F1 Score |
|-------------------|----------|----------|
| **AdaBoost**      | 0.9565   | 0.7431   |
| **GradientBoost** | 0.9724   | 0.8088   |
| **XGBoost**       | 0.9712   | 0.8057   |

### 🔹 Confusion Matrices
- **Blue** → AdaBoost  
- **Orange** → Gradient Boosting  
- **Green** → XGBoost  

### 🔹 Bar Chart (Accuracy vs F1 Score)
- Visual comparison of model performance.  

---

## 📂 Files

- `boosting_models.py` → Python script containing preprocessing, training, and evaluation.  
- `diabetes_prediction_dataset.csv` → Dataset used.  

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/ashhwiithac22/NLP---Playground.git
cd NLP---Playground

# Install dependencies
pip install pandas scikit-learn xgboost matplotlib
```
Run the script:
```
python boosting_models.py
```
### 📌 Key Takeaways

- AdaBoost: Simple, focuses on misclassified samples, good but less powerful.
- Gradient Boosting: Corrects previous errors using gradient descent, higher accuracy.
- XGBoost: Most robust, prevents overfitting with regularization, highly efficient.

--- 
Boosting models significantly improve accuracy and reduce bias compared to single decision trees
