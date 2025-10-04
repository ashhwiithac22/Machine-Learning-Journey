'''
k-means clustering is an unsupervised machine learning algorithm uses euclidean distance,suitable for larger datasets
initialize k(represents number of clusters) centroids randomly.
each cluster has a centroid, which is the mean of the points in that cluster.
based on distance each point is assigned to the centroid.
we use elbow curve to choose the optimal number of clusters.
hierarchical clustering makes clusters by building a tree like structure 
suitable for small datasets 
silhoutte score tells us how good the clustering is
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch #for dendrogram

df = pd.read_csv("D:/ML Lab/marketing_campaign.csv", sep="\t", encoding="ISO-8859-1") #encoding avoids character errors
df = df.dropna() 
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str)) #converts categorical to numerical values

X = df.select_dtypes(include=[np.number])
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_pca)
print("KMeans Silhouette:", silhouette_score(X_pca, k_labels))

plt.scatter(X_pca[:,0], X_pca[:,1], c=k_labels, cmap='tab10', s=20)
plt.title("KMeans Clusters (k=4)")
plt.show()
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
a_labels = agg.fit_predict(X_pca)
print("Agglomerative Silhouette:", silhouette_score(X_pca, a_labels))

plt.scatter(X_pca[:,0], X_pca[:,1], c=a_labels, cmap='tab10', s=20)
plt.title("Agglomerative Clusters (k=4)")
plt.show()

sample = X_scaled[:50]   
linkage_matrix = sch.linkage(sample, method='ward')
sch.dendrogram(linkage_matrix)
plt.title("Hierarchical Dendrogram")
plt.show()
if silhouette_score(X_pca, k_labels) > silhouette_score(X_pca, a_labels):
    print("KMeans performed better")
else:
    print("Agglomerative performed better")
