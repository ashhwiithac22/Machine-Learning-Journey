# Cross Validation   

## 📌 Overview  
Cross Validation is a supervised learning technique used to evaluate how well a machine learning model performs on unseen data.  
It prevents **overfitting**, allows us to **compare different ML models**, and ensures that our model generalizes well.  

- ✅ Prevents overfitting (model performs well on training but poorly on test data)  
- ✅ Evaluates model performance  
- ✅ Splits dataset into **k folds** for training & testing  
- ✅ Helps in **model comparison** and **hyperparameter tuning**  

---

## 🧪 Experiment Steps  

1. **Load Dataset**  
   - Used **Avocado dataset** for regression tasks.  
   - Cleaned data by handling missing values, encoding categorical variables, and removing outliers.  

2. **EDA (Exploratory Data Analysis)**  
   - Summary statistics and feature correlation.  
   - Converted `Date` to **Month** and **Day** features.  
   - Encoded categorical columns (`type`, `region`).  

3. **Model Training**  
   - Applied **Multiple Linear Regression** using features (`4046`, `4225`, `4770`) to predict `Total Volume`.  
   - Visualized **Actual vs Predicted** results with line plots.  

4. **Cross Validation (KFold)**  
   - Implemented **KFold Cross Validation** for `k = 3 to 10`.  
   - Calculated **R² Score** for each fold.  
   - Computed **Average R² Score** for every `k`.  
   - Plotted **Average R² Score vs k-folds**.  

---

## 📊 Results  

- **Cross Validation gave more reliable accuracy than a single train-test split.**  
- The model’s performance improved with increasing **k** values (up to a point).  
- Found the best `k` based on highest average **R² score**.  

---

## 🛠 Libraries Used  

- `pandas` – Data handling  
- `numpy` – Numerical operations  
- `matplotlib`, `seaborn` – Visualization  
- `sklearn` – Linear Regression, Cross Validation, Metrics  

---

## 📌 Key Learnings  

- Cross validation avoids overfitting and provides a **fairer evaluation**.  
- Larger `k` values → lower bias, but higher variance.  
- Best practice: use **k=5 or k=10** for stable results.  

---

## 📷 Visualizations  

- Actual vs Predicted line plot (Regression)  
- Average R² Score vs k-folds plot  

---

## ✅ Conclusion  

- Cross Validation ensures our model is not just memorizing training data but actually learning patterns that generalize well.
- In real projects, it is a **must-use step** for reliable model evaluation and selection.  
