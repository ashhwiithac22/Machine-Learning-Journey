'''
cross validation helps to prevent overfitting
allows us to compare different machine leaning models
cross validation evaluates a model
it is a supervised technique
overfitting means models performs well on training data and poor on test data
'''
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#EDA
df = pd.read_csv(r"D:\ML Lab\avocado.csv")
print(df.head())
print(df.info())
df.drop_duplicates()
print(df.isnull().sum())
print(df.describe())
df = df.fillna(df.mean(numeric_only=True))
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop(['Date', 'Unnamed: 0'], axis=1, errors='ignore')
df['type'] = df['type'].map({'conventional': 0, 'organic': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
'''
#correlation heatmap
numeric_cols = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

#outlier analysis
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['4046'])
plt.title("Before: Outliers in 4046")
plt.show()
print("Data before removing outliers:", df.shape)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['4225'])
plt.title("Before: Outliers in 4225")
plt.show()
print("Data before removing outliers:", df.shape)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['4770'])
plt.title("Before: Outliers in 4770")
plt.show
print("Data before removing outliers:", df.shape)
#removing outliers
Q1 = df['4046'].quantile(0.25)
Q3 = df['4046'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['4046'] >= lower) & (df['4046'] <= upper)]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['4046'])
plt.title("4046 :After removing Outliers")
plt.show()
Q1 = df['4225'].quantile(0.25)
Q3 = df['4225'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['4225'] >= lower) & (df['4225'] <= upper)]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['4225'])
plt.title("4225 :After removing Outliers")
plt.show()
Q1 = df['4770'].quantile(0.25)
Q3 = df['4770'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['4770'] >= lower) & (df['4770'] <= upper)]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['4770'])
plt.title("4770 :After removing Outliers")
plt.show()
'''

#Multiple Linear Regression
X_multiple = df[['4046', '4225', '4770']]
y = df['Total Volume']
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)
y_pred_multiple = model_multiple.predict(X_test)
#Line chart
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred_multiple, label="Predicted", color='red')
plt.title("Multiple Linear Regression - Line Plot")
plt.xlabel("Samples")
plt.ylabel("Total Volume")
plt.legend()
plt.show()

#Kfold cross validation 
average_scores = [] 
for k in range(3, 11):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = []
    fold = 1
    print(f"\nk = {k}")
    for train, test in kf.split(X_multiple):
        X_train = X_multiple.values[train]
        X_test = X_multiple.values[test]
        y_train = y.values[train]
        y_test = y.values[test]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        r2_scores.append(score)
        print(f"- Chunk {fold} R² Score: {score:.4f} ({score*100:.2f}%)")
        fold += 1
    avg = np.mean(r2_scores)
    average_scores.append(avg)   
    print(f"- Average R² Score for k={k}: {avg:.4f} ({avg*100:.2f}%)")

plt.plot(range(3, 11), [s*100 for s in average_scores], marker='o', color='purple')
plt.xlabel("k value (folds)")
plt.ylabel("Average R² Score (%)")
plt.title("Average R² Score vs k-folds")
plt.grid()
plt.show()



