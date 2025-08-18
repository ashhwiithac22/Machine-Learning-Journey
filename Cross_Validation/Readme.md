# Cross Validation 

- This project demonstrates the implementation of **K-Fold Cross Validation** using the Avocado dataset.
- Cross Validation is a supervised learning technique that helps evaluate and improve machine learning models by preventing **overfitting** and allowing fair comparison of models.

---

## 📌 Key Concepts
- **Cross Validation** helps to:
  - Prevent overfitting  
  - Evaluate model performance on unseen data  
  - Compare different machine learning models  
- **Overfitting** means the model performs well on training data but poorly on test data.  
- **K-Fold Cross Validation** splits the dataset into *k* equal parts and iteratively trains & tests the model.

---

## 🛠️ Libraries Used
- `pandas`, `numpy` → Data handling  
- `matplotlib`, `seaborn` → Data visualization  
- `sklearn.model_selection` → KFold, cross_val_score, train_test_split  
- `sklearn.linear_model` → LinearRegression  
- `sklearn.metrics` → r² score, MAE, MSE  

---

## 📊 Workflow
1. **Exploratory Data Analysis (EDA):**
   - Handled missing values  
   - Converted categorical features (type, region)  
   - Extracted features from Date (Month, Day)  
   - Removed unnecessary columns  

2. **Model Training:**
   - Built a **Multiple Linear Regression** model using features:
     - `4046`, `4225`, `4770` (avocado PLU codes)  
   - Target variable: `Total Volume`  

3. **Visualization:**
   - Line plot comparing Actual vs Predicted values  
   - Correlation Heatmap (commented in code)  
   - Outlier Analysis (commented in code)  

4. **Cross Validation:**
   - Implemented **K-Fold Cross Validation** (k = 3 to 10)  
   - Calculated and displayed **R² scores** for each fold  
   - Plotted **Average R² Score vs k-folds**  

---

## 📈 Results
- Observed the **R² score stability** across folds.  
- Visualized performance trend as `k` varied.  
- Showed how cross validation improves model evaluation.  

---

## 🚀 How to Run
1. Clone this repo:

       git clone https://github.com/ashhwiithac22/Machine-Learning-Journey.git


2.Navigate to Cross_Validation folder.

3.Install requirements (if not already):

      pip install pandas numpy matplotlib seaborn scikit-learn


4.Run the script:

     python ex2- cross validation.py


---

## 📂 Files in This Folder

- ex2- cross validation.py → Main script
- avocado.csv → Dataset
- Rsquared value vs K-folds.png → Visualization of CV results

---

## 🔑 Learnings

- Cross validation provides a better estimate of model performance.
- R² score varies across folds but averaging reduces bias.
- Helps in model selection and hyperparameter tuning.




