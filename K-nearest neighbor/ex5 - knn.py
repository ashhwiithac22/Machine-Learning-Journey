'''
knn is a supervised algorithm,its good idea to use odd numbers of k
avoid ties using odd number of k
when p = 2 it is euclidean distance,p=1 manhattan distance
knn is a lazy learning algorithm,because it uses all data for training
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score
df = pd.read_csv(r"D:\ML Lab\diabetes_prediction_dataset.csv")
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).fillna(0)
df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True) #makes categorical cloumn to numeric
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.01, 0.99]) # the value which 1% and 99% data lies
    df = df[(df[col] >= Q1) & (df[col] <= Q3)]
X = df.drop('diabetes', axis=1)
y = df['diabetes']
scaler = StandardScaler() #makes all features to a common scale
X_scaled = scaler.fit_transform(X) #makes mean 0 and variance 1
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
k_values = range(1, 20, 2)
error_rates = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error)
plt.figure(figsize=(8, 5))
plt.plot(k_values, error_rates, marker='o', linestyle='-', color='blue')
plt.title('Elbow Curve')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.xticks(k_values)
plt.xlim(0, 20) #sets a limit from 0 to 20
plt.grid(True)
plt.show()
best_k = 5
print(f"\nBest k:",best_k)
final_knn = KNeighborsClassifier(n_neighbors=best_k,metric='euclidean')
final_knn.fit(X_train, y_train)
y_pred_test = final_knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
print("\nConfusion Matrix:\n", cm)
print(f"F1 Score: {f1:.4f}")
def user_input():
    gender_input = input("Enter Gender (Male/Female): ").strip().lower() #strip is used to remove unwanted spaces
    age = float(input("Enter Age: "))   
    bmi = float(input("Enter BMI: "))
    hba1c = float(input("Enter HbA1c Level: "))
    glucose = float(input("Enter Blood Glucose Level: "))
    hypertension = int(input("Do you have Hypertension? (1 = Yes, 0 = No): "))
    heart_disease = int(input("Do you have Heart Disease? (1 = Yes, 0 = No): "))
    smoking = input("Enter Smoking History (never/no_info/current/former/not current/ever): ").strip().lower()
    gender_code = 1 if gender_input == "female" else 0
    input_data = {
        'gender': [gender_code],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    }
    smoking_col = f"smoking_history_{smoking}"
    input_data[smoking_col] = [1]
    return pd.DataFrame(input_data)

def diabetes_prediction(model, scaler, X_columns, X_scaled, y):
    user_df = user_input()

    for col in X_columns:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[X_columns]
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]
    
    if prediction == 1:
        print("\n Diabetes")
    else:
        print("\n No Diabetes")

diabetes_prediction(final_knn, scaler, X.columns, X_scaled, y)