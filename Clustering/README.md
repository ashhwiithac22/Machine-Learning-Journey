# Customer Segmentation using Clustering Algorithms

This project demonstrates **unsupervised machine learning** for customer segmentation using **K-Means** and **Hierarchical (Agglomerative) Clustering**.  
It applies **PCA** for dimensionality reduction, evaluates clustering with **silhouette scores**, and visualizes clusters and dendrograms.

---

## 📌 Concepts Covered

### 🔹 K-Means Clustering
- Unsupervised algorithm using **Euclidean distance**.  
- Suitable for **larger datasets**.  
- Steps:
  1. Initialize `k` centroids randomly (`k` = number of clusters).  
  2. Assign each point to the nearest centroid.  
  3. Update centroids as the mean of points in each cluster.  
- **Elbow curve** can be used to choose the optimal `k`.  

### 🔹 Hierarchical (Agglomerative) Clustering
- Builds clusters by creating a **tree-like structure (dendrogram)**.  
- Suitable for **smaller datasets**.  
- Evaluated using **silhouette score** to measure clustering quality.  

---

## 🛠️ Tech Stack
- **Python 3**  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [NumPy](https://numpy.org/) – Numerical operations  
- [Matplotlib](https://matplotlib.org/) – Visualizations  
- [Scikit-learn](https://scikit-learn.org/) – PCA, KMeans, AgglomerativeClustering, StandardScaler, silhouette_score  
- [SciPy](https://www.scipy.org/) – Dendrogram generation  

---

## 📂 Workflow
1. Load the dataset (`marketing_campaign.csv`) and clean missing values.  
2. Encode categorical variables using **LabelEncoder**.  
3. Standardize numerical features (**mean=0, variance=1**).  
4. Reduce dimensions to 2D using **PCA** for visualization.  
5. Apply **K-Means** and **Agglomerative Clustering** with 4 clusters.  
6. Calculate **silhouette scores** to evaluate clustering quality.  
7. Visualize clusters in 2D and create a **hierarchical dendrogram**.  
8. Compare performance of K-Means vs Agglomerative Clustering.  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ashhwiithac22/Machine-Learning-Journey/tree/main/Clustering
   cd your-repo-name
2.Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn scipy
   ```
3.Place the dataset marketing_campaign.csv in the project folder.


4.Run the script:
   ```bash
  python main.py
```
---
## 📊 Sample Output
### 🔹 Silhouette Scores
KMeans Silhouette: 0.4266

Agglomerative Silhouette: 0.3662

Best Method: KMeans

### 🔹 Visualizations

KMeans Clusters (k=4):


Agglomerative Clusters (k=4):


Hierarchical Dendrogram:
---
### 🎯 Insights

K-Means performed better on this dataset based on silhouette score.

Clustering helps identify distinct customer segments for targeted marketing strategies.

---
