# Naive Bayes – Machine Learning Lab  

## 📌 Overview  
Naive Bayes is a **probabilistic machine learning algorithm** based on **Bayes’ Theorem**.  
It assumes that features are **independent of each other** (naive assumption).  
It is widely used for **classification problems**, especially in **medical diagnosis** and **text classification**.  

### 🔑 Types of Naive Bayes  
1. **Gaussian Naive Bayes (GNB)** → Works with **continuous values** that follow a **normal distribution** (bell curve).  
   - Example: Medical diagnosis, diabetes prediction.  
2. **Multinomial Naive Bayes (MNB)** → Works with **discrete data** (word counts, frequencies).  
   - Example: Spam detection in emails.  
3. **Bernoulli Naive Bayes** → Works with **binary features** (Yes/No, 0/1).  
   - Example: Sentiment classification.  

---

## 🧪 Datasets Used  

1. **Diabetes Prediction Dataset**  
   - Features: age, gender, BMI, HbA1c level, blood glucose level, smoking history, etc.  
   - Target: `diabetes` (0 = No, 1 = Yes).  
   - Algorithm: **Gaussian Naive Bayes**.  

2. **Spam Detection Dataset (`spam.csv`)**  
   - Features: Word frequencies from SMS text messages.  
   - Target: `label` (0 = Ham, 1 = Spam).  
   - Algorithm: **Multinomial Naive Bayes**.  

---

## ⚙️ Methodology  

### 1️⃣ Diabetes Dataset – Gaussian Naive Bayes  
- Performed **EDA**: checked missing values, summary statistics.  
- **Removed outliers** using IQR method.  
- Applied **OneHotEncoding** for categorical features (`smoking_history`).  
- Used **StandardScaler** for normalization.  
- Trained a **GaussianNB model** with all features.  
- Also tested with **selected features** using correlation heatmap.  
- Evaluated with **F1 Score** and **Cross Validation**.  

### 2️⃣ Spam Dataset – Multinomial Naive Bayes  
- Cleaned dataset (`spam.csv`), encoded labels (`ham=0`, `spam=1`).  
- Converted text into numerical features using **CountVectorizer**.  
- Plotted **Top 20 most frequent words** in SMS messages.  
- Trained a **MultinomialNB model**.  
- Evaluated using **Classification Report, Confusion Matrix, F1 Score, Cross Validation**.  
- Implemented **real-time user input prediction** (SPAM/HAM).  

---

## 📊 Results  

### ✅ Diabetes Dataset (GaussianNB)  
- **F1 Score (with all features):** ~0.31 
- **Cross-Validation Avg F1 Score:** ~0.38  
- # Naive Bayes – Machine Learning Lab  

## 📌 Overview  
Naive Bayes is a **probabilistic machine learning algorithm** based on **Bayes’ Theorem**.  
It assumes that features are **independent of each other** (naive assumption).  
It is widely used for **classification problems**, especially in **medical diagnosis** and **text classification**.  

### 🔑 Types of Naive Bayes  
1. **Gaussian Naive Bayes (GNB)** → Works with **continuous values** that follow a **normal distribution** (bell curve).  
   - Example: Medical diagnosis, diabetes prediction.  
2. **Multinomial Naive Bayes (MNB)** → Works with **discrete data** (word counts, frequencies).  
   - Example: Spam detection in emails.  
3. **Bernoulli Naive Bayes** → Works with **binary features** (Yes/No, 0/1).  
   - Example: Sentiment classification.  

---

## 🧪 Datasets Used  

1. **Diabetes Prediction Dataset**  
   - Features: age, gender, BMI, HbA1c level, blood glucose level, smoking history, etc.  
   - Target: `diabetes` (0 = No, 1 = Yes).  
   - Algorithm: **Gaussian Naive Bayes**.  

2. **Spam Detection Dataset (`spam.csv`)**  
   - Features: Word frequencies from SMS text messages.  
   - Target: `label` (0 = Ham, 1 = Spam).  
   - Algorithm: **Multinomial Naive Bayes**.  

---

## ⚙️ Methodology  

### 1️⃣ Diabetes Dataset – Gaussian Naive Bayes  
- Performed **EDA**: checked missing values, summary statistics.  
- **Removed outliers** using IQR method.  
- Applied **OneHotEncoding** for categorical features (`smoking_history`).  
- Used **StandardScaler** for normalization.  
- Trained a **GaussianNB model** with all features.  
- Also tested with **selected features** using correlation heatmap.  
- Evaluated with **F1 Score** and **Cross Validation**.  

### 2️⃣ Spam Dataset – Multinomial Naive Bayes  
- Cleaned dataset (`spam.csv`), encoded labels (`ham=0`, `spam=1`).  
- Converted text into numerical features using **CountVectorizer**.  
- Plotted **Top 20 most frequent words** in SMS messages.  
- Trained a **MultinomialNB model**.  
- Evaluated using **Classification Report, Confusion Matrix, F1 Score, Cross Validation**.  
- Implemented **real-time user input prediction** (SPAM/HAM).  

---

## 📊 Results  

### ✅ Diabetes Dataset (GaussianNB)  
- **F1 Score (with all features):** ~0.32 
- **Cross-Validation Avg F1 Score:** ~0.38
- Performance was low (affected by class imbalance & feature correlations).  

### ✅ Spam Dataset (MultinomialNB)  
- **F1 Score:** ~0.95 – 0.97  
- **Cross-Validation Avg F1 Score:** ~0.96  
- Much higher accuracy compared to diabetes dataset.  
- Model performed very well since text classification is well-suited for MultinomialNB.  

📌 **Comparison:**  
- GaussianNB struggled with diabetes dataset (continuous medical data, not fully independent features).  
- MultinomialNB gave **much higher accuracy** for spam detection (text data matches assumptions of the model).  

---

## 📷 Visualizations  

- **Boxplots** (before & after outlier removal for diabetes).  
- **Histograms with KDE** (to verify Gaussian assumption).  
- **Feature Correlation Heatmap** (to choose important diabetes features).  
- **Top 20 Word Frequencies** (spam dataset).  
- **Confusion Matrix** for both tasks.  

---

## 🛠 Libraries Used  

- `pandas`, `numpy` – Data handling  
- `matplotlib`, `seaborn` – Visualization  
- `sklearn` – Naive Bayes, Train-Test Split, Metrics, Preprocessing  
- `CountVectorizer` – Text feature extraction (spam dataset)  

---

## ✅ Conclusion  

- **Naive Bayes is simple, fast, and effective for classification tasks.**  
- **GaussianNB** is suitable for continuous numerical data but may struggle when features are correlated (like medical data).  
- **MultinomialNB** is excellent for **text classification problems** like spam detection.  
- In this experiment:  
  - Spam dataset achieved **much higher accuracy** (~96% F1) than the diabetes dataset (~68% F1).  
  - This shows how the **choice of Naive Bayes type depends on the dataset nature**.  


### ✅ Spam Dataset (MultinomialNB)  
- **F1 Score:** ~0.9276
- **Cross-Validation Avg F1 Score:** ~0.9232
- Much higher accuracy compared to diabetes dataset.  
- Model performed very well since text classification is well-suited for MultinomialNB.  

📌 **Comparison:**  
- GaussianNB struggled with diabetes dataset (continuous medical data, not fully independent features).  
- MultinomialNB gave **much higher accuracy** for spam detection (text data matches assumptions of the model).  

---

## 📷 Visualizations  

- **Boxplots** (before & after outlier removal for diabetes).  
- **Histograms with KDE** (to verify Gaussian assumption).  
- **Feature Correlation Heatmap** (to choose important diabetes features).  
- **Top 20 Word Frequencies** (spam dataset).  
- **Confusion Matrix** for both tasks.  

---

## 🛠 Libraries Used  

- `pandas`, `numpy` – Data handling  
- `matplotlib`, `seaborn` – Visualization  
- `sklearn` – Naive Bayes, Train-Test Split, Metrics, Preprocessing  
- `CountVectorizer` – Text feature extraction (spam dataset)  

---

## ✅ Conclusion  

- **Naive Bayes is simple, fast, and effective for classification tasks.**  
- **GaussianNB** is suitable for continuous numerical data but may struggle when features are correlated (like medical data).  
- **MultinomialNB** is excellent for **text classification problems** like spam detection.  
- In this experiment:  
  - Spam dataset achieved **much higher accuracy** (~96% F1) than the diabetes dataset (~68% F1).  
  - This shows how the **choice of Naive Bayes type depends on the dataset nature**.  
