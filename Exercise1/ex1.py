'''
simple linear regression - one independent and one dependent variable
multiple linear regresison - many independent and one dependent variable
polynomial regression - using least sqaures methdod(degree = 2 ,often used in real world scenarios,non-linear relationship between x and y)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv(r"D:\ML Lab\avocado.csv")

# EDA
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



#Testing and training the data
X = df[['4046']]               # Independent variable
y = df['Total Volume']         # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression
X_simple = df[['4046']]
y = df['Total Volume']
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)
#Line chart
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred_simple, label="Predicted", color='red')
plt.title("Simple Linear Regression - Line Plot")
plt.xlabel("Samples")
plt.ylabel("Total Volume")
plt.legend()
plt.show()
#Scatter plot 
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label="Actual")
plt.scatter(X_test, y_pred_simple, color='red', alpha=0.5, label="Predicted")
plt.xlabel("4046")
plt.ylabel("Total Volume")
plt.title("Simple Linear Regression - Scatter Plot")
plt.legend()
plt.show()


# Multiple Linear Regression
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
#Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_multiple, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Total Volume")
plt.ylabel("Predicted Total Volume")
plt.title("Multiple Linear Regression - Scatter Plot")
plt.show()

# Polynomial Regression
X_poly = PolynomialFeatures(degree=2).fit_transform(df[['4046']])
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
model_poly = LinearRegression()
model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)
#Line chart
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred_poly, label="Predicted", color='red')
plt.title("Polynomial Regression (deg 2) - Line Plot")
plt.xlabel("Samples")
plt.ylabel("Total Volume")
plt.legend()
plt.show()
#Scatter plot 
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 1], y_test, color='green', label='Actual', alpha=0.5)
plt.scatter(X_test[:, 1], y_pred_poly, color='red', label='Predicted', alpha=0.5)
plt.xlabel("4046")
plt.ylabel("Total Volume")
plt.title("Polynomial Regression (Degree 2) - Scatter Plot")
plt.legend()
plt.show()


#Model Performance Metrics
models = ['Simple Linear', 'Multiple Linear', 'Polynomial (deg 2)']
mae = [
    mean_absolute_error(y_test, y_pred_simple),
    mean_absolute_error(y_test, y_pred_multiple),
    mean_absolute_error(y_test, y_pred_poly)
]
mse = [
    mean_squared_error(y_test, y_pred_simple),
    mean_squared_error(y_test, y_pred_multiple),
    mean_squared_error(y_test, y_pred_poly)
]
rmse = [np.sqrt(i) for i in mse]
r2 = [
    r2_score(y_test, y_pred_simple),
    r2_score(y_test, y_pred_multiple),
    r2_score(y_test, y_pred_poly)
]

plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
plt.bar(models, mae, color='skyblue')
plt.title('MAE Comparison')
plt.subplot(2,2,2)
plt.bar(models, mse, color='orange')
plt.title('MSE Comparison')
plt.subplot(2,2,3)
plt.bar(models, rmse, color='green')
plt.title('RMSE Comparison')
plt.subplot(2,2,4)
plt.bar(models, r2, color='red')
plt.title('R² Comparison')
plt.tight_layout()
plt.show()
models = ['Simple Linear', 'Multiple Linear', 'Polynomial (deg 2)']
r2_values = [r2,r2,r2]

#Best Model based on R squared value 
best = r2.index(max(r2))
print(f"Best Model: {models[best]} (R² = {r2[best]:.4f})")
print("MAE:", mae[best])
print("MSE:", mse[best])
print("RMSE:", rmse[best])
val_4046 = float(input("Enter value for 4046: "))
val_4225 = float(input("Enter value for 4225: "))
val_4770 = float(input("Enter value for 4770: "))
input_data = np.array([[val_4046, val_4225, val_4770]])
predicted_volume = model_multiple.predict(input_data)
print(f"\nPredicted Total Volume: {predicted_volume[0]:.2f}")
