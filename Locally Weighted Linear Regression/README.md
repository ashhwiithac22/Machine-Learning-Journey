# Locally Weighted Linear Regression (LWLR)

This project demonstrates **Locally Weighted Linear Regression (LWLR)** — a non-parametric regression technique that focuses on **nearby data points** to make smooth predictions.  
It is particularly useful in **time series forecasting** and scenarios where local trends matter more than global ones.

---

## 📘 Overview
- **Focuses on nearby data points** using a **Gaussian kernel**.  
- Each data point is given a **weight** based on its distance from the query point.  
- Provides **better local predictions** than ordinary least squares regression.  
- Ideal for **non-linear relationships** and **small datasets**.

---

## 🧠 Key Concepts

### 🔹 How LWLR Works
1. Choose a bandwidth parameter `τ (tau)` to control how much nearby points influence prediction.  
2. Compute weights for each training point using the **Gaussian kernel**:  
   \[
   w_i = e^{-\frac{(x - x_i)^2}{2\tau^2}}
   \]
3. Solve for regression coefficients using weighted least squares:  
   \[
   \theta = (X^T W X)^{-1} X^T W y
   \]
4. Predict the output for the given query point.

---

## 🛠️ Tech Stack
- **Python 3**  
- [NumPy](https://numpy.org/) – Linear algebra & mathematical operations  
- [Pandas](https://pandas.pydata.org/) – Data loading and manipulation  
- [Matplotlib](https://matplotlib.org/) – Visualization and plotting  

---

## 📂 Workflow
1. Load dataset (`tips.csv`) containing **total_bill** and **tip** columns.  
2. Prepare input (`X`) and output (`y`) variables.  
3. Implement **Locally Weighted Linear Regression** with Gaussian kernel weighting.  
4. Predict tips across a range of total bills.  
5. Visualize both data points and the fitted LWLR curve.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ashhwiithac22/Machine-Learning-Journey/tree/main/Locally%20Weighted%20Linear%20Regression
   cd lwlr-tips-analysis
   ```
2.Install dependencies:
   ```
   pip install numpy pandas matplotlib

  ```
3.Place the dataset tips.csv in your project folder.

4.Run the script:
  ```
   python main.py
   ```
5. The plot will display the original data points and the LWLR fitted curve.

---
## Sample Output
- Blue dots → Original data points
- Red curve → LWLR fitted line

## Insights
- LWLR adapts to local patterns in data rather than fitting a single global line.
- Provides smooth and accurate predictions when the relationship between variables is non-linear.
- Ideal for time series forecasting and trend analysis.
