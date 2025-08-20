# Decision Trees 

## 📌 Overview  
Decision Trees are a **supervised machine learning algorithm** used for both **classification** and **regression** tasks.  
They work like a **flowchart of yes/no questions**, splitting the dataset step by step until a decision (leaf node) is made.  

- ✅ Easy to understand & interpret (works like human decision-making).  
- ✅ Handles both categorical & numerical data.  
- ✅ Requires little data preprocessing.  

---

## 🌲 Types of Decision Trees  

1. **Classification Tree** → Predicts categorical outcomes (e.g., Spam / Not Spam).  
2. **Regression Tree** → Predicts continuous values (e.g., House Prices).  

---

## 📑 Key Concepts  

- **Decision Node** → Point where dataset splits based on a feature.  
- **Leaf Node** → Final outcome (prediction).  
- **Pruning** → Removes unnecessary branches to reduce overfitting.  
  - Pre-pruning → Stop tree early.  
  - Post-pruning → Grow full tree, then trim.  

**Tree Depth**  
- Small depth → Underfitting (too simple).  
- Large depth → Overfitting (too complex).  

**Splitting Criteria**  
- **Entropy** & **Information Gain** → Measures purity/uncertainty in nodes.  
- **Gini Index** → Another impurity measure used by default in sklearn.  

---

## 🧪 Experiment Steps  

1. **Dataset Used**: `diabetes_prediction_dataset.csv`  
   - Target variable: `diabetes` (Yes/No).  
   - Features: age, bmi, HbA1c_level, blood_glucose_level, gender, smoking history, etc.  

2. **Data Preprocessing**  
   - Encoded categorical variables (`gender`, `smoking_history`) using OneHotEncoding.  
   - Checked for missing values & dataset balance.  

3. **Model Training**  
   - Trained **DecisionTreeClassifier** using `train_test_split`.  
   - Plotted **Tree Depth vs F1 Score** to find best depth.  

4. **Hyperparameter Tuning**  
   - Used **GridSearchCV** to optimize:  
     - `max_depth` → Tree depth  
     - `min_samples_split` → Minimum samples to split a node  
     - `min_samples_leaf` → Minimum samples in a leaf  
     - `criterion` → gini / entropy  
   - Selected best parameters for final model.  

5. **Visualization**  
   - Plotted **Decision Tree** with features.  
   - Generated **Confusion Matrix**.  

---

## 📊 Results  

- Found **best depth** using F1 score comparison.  
- Optimized parameters with **GridSearchCV**.  
- Model performed well with balanced bias-variance tradeoff.  

**Confusion Matrix Example**:  
- Predicted vs Actual diabetes outcomes.  
- Evaluated with **F1 Score** for balanced performance.  

---

## 🛠 Libraries Used  

- `pandas`, `numpy` – Data handling  
- `matplotlib`, `seaborn` – Visualization  
- `sklearn` – DecisionTreeClassifier, GridSearchCV, Metrics  

---

## 📌 Key Learnings  

- Decision Trees are easy to visualize & interpret.  
- Without pruning, they may **overfit** (high variance).  
- **Hyperparameter tuning** improves generalization.  
- Best practice: use **Ensemble methods** (Bagging, Random Forest, Boosting) for stronger performance.  

---

## 📷 Visualizations  

- **Tree Depth vs F1 Score** curve  
- **Decision Tree structure**  
- **Confusion Matrix heatmap**  

---

## ✅ Conclusion  

- Decision Trees are powerful and interpretable models but can easily overfit.
- With proper tuning (max depth, pruning), they provide strong results and form the **base algorithm for ensemble methods** like Random Forests and Gradient Boosting.  
