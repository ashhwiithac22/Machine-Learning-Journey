'''
group of decision trees
each decision tree takes random values(rows) and random features
duplicates are allowed ie same row can appear in many decision trees
each tree is different a bit,picks random features
handles missing data, can handle large datasets efficiently
higher the number of trees greater the accuracy
step 1: the dataset has been broken down into different subsets
step 2: build the tree for each and every subsets
step 3: combine all the tree's output to get final decision
step 4: make prediction
'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,f1_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.dropna(subset=['diabetes'], inplace=True) #drop the missing values in the target column
X = pd.get_dummies(df.drop("diabetes", axis=1), drop_first=True) #get_dummies is used to convert categorical value into numeric columns
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
      'n_estimators': [50, 100], #number of trees in the forest
      'max_depth': [5, 10], #maximum depth for each tree
      'min_samples_split': [2, 5] #minimum samples required to split a internal node
  }
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid=params)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
f1_score = f1_score(y_test,y_pred)
print(f"Best Parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.5f}")
print(f"F1-score:{f1_score}")
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Purples')
plt.title("Random Forest Confusion Matrix")
plt.show()



