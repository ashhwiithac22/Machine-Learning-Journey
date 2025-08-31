# Random Forest Classifier for Diabetes Prediction

## 📌 Overview
This project demonstrates the use of **Random Forest Classifier** to predict the presence of diabetes based on medical data.  
Random Forest is an ensemble learning method that builds multiple decision trees on random subsets of the data and features, then combines their predictions for improved accuracy and robustness.

---

## ⚙️ Steps of Random Forest
1. The dataset is split into random subsets (rows and features).
2. A decision tree is built for each subset (with replacement → duplicates allowed).
3. Predictions from all trees are combined (majority voting).
4. Final prediction is made based on aggregated results.

---

## 🧾 Dataset
- **File:** `diabetes_prediction_dataset.csv`  
- **Target Column:** `diabetes`  
- The dataset contains patient health features used to predict whether the person has diabetes.

---

## 🚀 Features of the Code
- Handles **missing values** (removes rows with missing `diabetes` values).
- Converts categorical features into numeric using **One-Hot Encoding** (`get_dummies`).
- Splits data into training (80%) and testing (20%).
- Uses **GridSearchCV** to find the best hyperparameters:
  - `n_estimators` → number of trees in the forest
  - `max_depth` → maximum depth of each tree
  - `min_samples_split` → minimum samples required to split a node
- Evaluates performance using:
  - **Accuracy Score**
  - **F1 Score**
  - **Confusion Matrix**

---

## 📊 Results
- Displays the **best hyperparameters** chosen by GridSearchCV.
- Prints:
  - Accuracy
  - F1 Score
- Plots a confusion matrix for visual evaluation.

---

### 🖥️ Dependencies
1. Install the required Python libraries before running:
```bash
pip install pandas scikit-learn matplotlib
```
### ▶️ How to Run

2. Place the diabetes_prediction_dataset.csv file in the project directory.

Run the script:
```
python ex8-random_forest.py
```
Check the console output for best parameters, accuracy, and F1 score.

View the confusion matrix plot.

### 📷 Visualization

Confusion Matrix is plotted with Purples colormap for better readability.

### 📚 Concepts Learned

- Random Forest Ensemble Method
- Handling categorical variables with One-Hot Encoding
- GridSearchCV for hyperparameter tuning
- Model evaluation using Accuracy, F1 Score, and Confusion Matrix
