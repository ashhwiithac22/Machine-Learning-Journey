# 📊 Regression Analysis – Avocado Dataset

This project demonstrates **Regression techniques** in Machine Learning using the **Avocado dataset**.  
It covers **Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression** along with Exploratory Data Analysis (EDA), outlier detection, and model performance evaluation.

---

## 📌 Steps Covered

### 🔍 Exploratory Data Analysis (EDA)
- Dataset inspection (`.head()`, `.info()`, `.describe()`)
- Handling missing values with mean imputation
- Feature engineering:
  - Extracting `Month` and `Day` from `Date`
  - Encoding categorical variables (`type`, `region`)
- Correlation Heatmap using **Seaborn**
- Outlier detection using **Boxplots** for `4046`, `4225`, `4770`

---

### 📈 Regression Models Implemented
1. **Simple Linear Regression**
   - One independent variable (`4046`)
   - Target: `Total Volume`
   - Line chart & scatter plot comparisons

2. **Multiple Linear Regression**
   - Independent variables: `4046`, `4225`, `4770`
   - Target: `Total Volume`
   - Scatter plot with line of best fit

3. **Polynomial Regression (Degree = 2)**
   - Captures **non-linear relationships**
   - Uses Least Squares method
   - Visualized with scatter plot & predictions

---

### 📊 Model Evaluation
Performance metrics compared across all three models:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

📌 A **bar chart comparison** of metrics is plotted to evaluate the best model.  
Finally, the **best model is selected automatically based on R² value**.

---

### 🔮 Prediction
The program allows the user to input values for:
- `4046`
- `4225`
- `4770`

➡️ It then predicts the **Total Avocado Sales Volume** using the trained Multiple Linear Regression model.

---

## ⚙️ Tech Stack
- **Python**
- **Pandas, NumPy** – Data preprocessing
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Regression models, metrics, preprocessing

---

## 🚀 How to Run
1. Clone this repository:
```bash
git clone https://github.com/your-username/Machine-Learning-Lab.git
cd Machine-Learning-Lab/Regression
```

2.Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3.Place the dataset (avocado.csv) inside the folder.

Run the script:
```bash
python regression_avocado.py
```
