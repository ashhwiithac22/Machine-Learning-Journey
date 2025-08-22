# 🩺 Diabetes Prediction using K-Nearest Neighbors (KNN)

- This project applies the **K-Nearest Neighbors (KNN)** algorithm to predict diabetes based on patient health data.
- The dataset includes features like age, BMI, HbA1c level, blood glucose level, smoking history, and medical conditions.
- The model uses **feature scaling, outlier removal, and categorical encoding** to improve accuracy.
- It also allows users to **input their details** and get a prediction on whether they may have diabetes.

---

## 🚀 Features
- Preprocessing:
  - Encodes categorical variables (`gender`, `smoking_history`)
  - Removes outliers using 1% and 99% quantiles
  - Standardizes features using **StandardScaler**
- Model:
  - Trains multiple KNN models with different `k` values
  - Plots **Elbow Curve** to find best `k`
  - Uses **Euclidean distance** metric
- Evaluation:
  - Confusion matrix
  - F1 Score
- User Interaction:
  - Accepts health details from the user
  - Predicts whether the user has diabetes

---

## 📂 Dataset
The dataset used is:
diabetes_prediction_dataset.csv

## 🛠️ Requirements
Install the following Python libraries before running:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 📊 Workflow

#  Load Dataset
- Reads CSV file using pandas.
# Preprocess Data
- Convert categorical columns to numerical (gender, smoking history).
- Remove outliers (1% and 99% quantiles).
- Scale features using StandardScaler.
# Train-Test Split
- Split dataset into 80% training, 20% testing.
# Model Training & Selection
- Train KNN classifier for k = 1, 3, 5, …, 19.
- Plot error rates to find best k.
- Train final model with best k.
## Evaluation

- Print Confusion Matrix.
- Print F1 Score.

# User Input Prediction
- Collects details (gender, age, BMI, HbA1c, glucose, smoking history, hypertension, heart disease).
- Encodes and scales the data.
- Predicts diabetes outcome.

### ▶️ How to Run

1.Clone the repo:
``` bash
git clone https://github.com/ashhwiithac22/Machine-Learning-Journey.git
```

2.Navigate to the project folder:
```bash
cd "ML daily practice"
```

3.Run the script:
```bash
python knn_diabetes.py
```
### 📌 Notes

- Use odd numbers of k to avoid ties in classification.
- p=2 → Euclidean distance, p=1 → Manhattan distance.
- KNN is a lazy learning algorithm since it stores all training data.
