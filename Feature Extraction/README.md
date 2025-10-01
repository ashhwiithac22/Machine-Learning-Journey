# Feature Extraction: PCA vs Factor Analysis

This project demonstrates **feature extraction** techniques, specifically **Principal Component Analysis (PCA)** and **Factor Analysis (FA)**, applied to a diabetes prediction dataset.  
It evaluates both methods by training a **Decision Tree Classifier** and comparing their accuracy.

---

## 📌 Key Concepts

### 🔹 Feature Extraction
- Creates new features by combining existing features.
- Helps in reducing dimensionality while preserving important information.

### 🔹 Principal Component Analysis (PCA)
- A **dimensionality reduction technique** that keeps the most important features.  
- Works by calculating **eigenvalues** and **eigenvectors** from the covariance matrix.  

Steps:
1. Standardize the data (mean = 0, variance = 1)  
2. Calculate the covariance matrix  
3. Calculate eigenvalues and eigenvectors  
4. Select top components (based on variance explained)

### 🔹 Factor Analysis (FA)
- Reduces the number of features by identifying **underlying latent factors** that explain correlations among variables.  

Steps:
1. Standardize the data (mean = 0, variance = 1)  
2. Estimate factor loadings  
3. Reduce data dimensionality using these factors  

---

## 🛠️ Tech Stack
- **Python 3**
- **Pandas** – Data manipulation  
- **Scikit-learn** – PCA, Factor Analysis, Decision Tree Classifier, Evaluation metrics  

---

## 📂 Workflow
1. Load dataset (`diabetes_prediction_dataset.csv`).  
2. Encode categorical variables using **Label Encoding**.  
3. Split dataset into train (80%) and test (20%).  
4. Apply **PCA** with 5 components and train a Decision Tree Classifier.  
5. Apply **Factor Analysis** with 5 components and train a Decision Tree Classifier.  
6. Compare performance using **Accuracy Score** and **Confusion Matrix**.  

---

## 🚀 How to Run
1. Clone this repository:
   ```
   git clone https://github.com/ashhwiithac22/Machine-Learning-Journey/tree/main/Feature%20Extraction
   cd your-repo-name
   ```
2. Install dependencies:
  ```
   pip install pandas scikit-learn
  ```
3.Run the script:
  ```
  python main.py
  ```
## 📊 Results
🔹 PCA

Accuracy: 95.30%

Confusion Matrix:
```
[[17818   474]
 [  465  1243]]
```
🔹 Factor Analysis

Accuracy: 92.82%

Confusion Matrix:
```
[[17604   688]
 [  747   961]]
```
✅ Best Method: PCA
