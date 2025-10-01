'''
feature extraction creates new features by combining existing features.
pca is a dimensionality reduction technqiue that reduces the number of features by keeping the most important features. it uses eigenvalues and eigen vectors 
from covariance matrix
1) standardize the data mean = 0 and variance = 1
2) calculate covariance matrix
3) calculate eigenvalues and eigenvectors
factor analysis is a technique that reduces the number of features by identifying underlying factors that explain the correlations among features.
1) standardize the data mean = 0 and variance = 1
2) estimate the factor loadings
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("diabetes_prediction_dataset.csv")
label_enc = LabelEncoder() #converts categorical values to numerical values
for col in df.columns: #loops through each column in the dataset
    if df[col].dtype == "object":
        df[col] = label_enc.fit_transform(df[col]) #assigns numerical values to categorical values

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=5) #pca keeps only 5 features because they contains most of the information
X_train_pca = pca.fit_transform(X_train) #calculates eigenvalues and eigenvectors and covariance matrix
X_test_pca = pca.transform(X_test) 

dt_pca = DecisionTreeClassifier(random_state=42)
dt_pca.fit(X_train_pca, y_train)
y_pred_pca = dt_pca.predict(X_test_pca)

pca_acc = accuracy_score(y_test, y_pred_pca)
print("PCA Accuracy:", pca_acc)
print("PCA Confusion Matrix:\n", confusion_matrix(y_test, y_pred_pca))

fa = FactorAnalysis(n_components=5, random_state=42) #reduces dataset by identifying hidden factors that explain correlations between features
X_train_fa = fa.fit_transform(X_train)
X_test_fa = fa.transform(X_test)

dt_fa = DecisionTreeClassifier(random_state=42)
dt_fa.fit(X_train_fa, y_train)
y_pred_fa = dt_fa.predict(X_test_fa)

fa_acc = accuracy_score(y_test, y_pred_fa)
print("Factor Analysis Accuracy:", fa_acc)
print("Factor Analysis Confusion Matrix:\n", confusion_matrix(y_test, y_pred_fa))

if pca_acc > fa_acc:
    print("Best Method: PCA ")
else:
    print("Best Method: Factor Analysis ")
