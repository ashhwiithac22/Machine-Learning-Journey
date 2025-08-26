'''
boosting - assigns equal weights at first, then it assigns higher weights for higher accuracy model and lower weigthts for lower accuracy model and in next step it concentrate on the lower accuracy weak leaner
gradient boosting - tries to rectify errors made by previous weak learner
xgboost - uses regularization add penalty(cost)
improves accuracy of models by combining multiple weak learners into strong learner
achieved by assigning higher weights
reduces bias in the model
adaboost - adaptive boost , assigns equal weights to all training samples
gradientboosting - minimizes error using gradient descent, reduces error by optimizing loss function,corrects error made by previous predecessor
XGboost - uses regularization to prevent overfitting
bagging - parallel , boosting - sequential 

bias(underfitting) - model is too simple and cannot capture the patterns in the data well,it makes lot of mistakes in both training and testing data
variance (overfitting)- model is too complex , performs well on training data and poor on test data 
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer #fills missing value in the dataset
from sklearn.compose import ColumnTransformer #apply different preprocessing steps to different columns
from sklearn.pipeline import Pipeline #combines the preprocessing steps together
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier #used as a weak learner in adaboost
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.dropna(subset=['diabetes'], inplace=True)
X = df.drop("diabetes", axis=1)
y = df["diabetes"]
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
num_transformer = SimpleImputer(strategy='mean') #uses mean for missing

#fills missing categorical values using the mode(most frequent values) 
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))
])
#combines both the transformers
preprocessor = ColumnTransformer( transformers=[('num', num_transformer, numerical_cols),('cat', cat_transformer, categorical_cols)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ada boost
ada_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)) #uses 50 decision trees as weak learners
])
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
print(f"Accuracy - adaboost:{accuracy_score(y_test, ada_preds):.4f}")
print(f"F1 Score - adaboost: {f1_score(y_test, ada_preds):.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_test, ada_preds)).plot(cmap='Blues')
plt.title("AdaBoost Confusion Matrix")
plt.show()

#gradient boosting
gb_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', GradientBoostingClassifier(random_state=42))])
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
print(f"Accuracy - gradient boosting: {accuracy_score(y_test, gb_preds):.4f}")
print(f"F1 Score - gradient boosting: {f1_score(y_test, gb_preds):.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_test, gb_preds)).plot(cmap='Oranges')
plt.title("Gradient Boosting Confusion Matrix")
plt.show()
X_encoded = pd.get_dummies(X, drop_first=True) #converts categorical data into numerical using one hot encoding and avoids dummy variable
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#xgbooost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42) #uses loglass as evaluation metric
xgb_model.fit(X_train_xgb, y_train_xgb)
xgb_preds = xgb_model.predict(X_test_xgb)
print(f"Accuracy - xgboost: {accuracy_score(y_test_xgb, xgb_preds):.4f}")
print(f"F1 Score - xgboost: {f1_score(y_test_xgb, xgb_preds):.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_test_xgb, xgb_preds)).plot(cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.show()

#comparison for performance metrics
models = ['AdaBoost', 'GradientBoosting', 'XGBoost']
accuracy_scores = [0.9565, 0.9724, 0.9712]
f1_scores = [0.7431, 0.8088, 0.8057]
bar_width = 0.35
x = range(len(models))
plt.figure(figsize=(8, 6))
plt.bar(x, accuracy_scores, width=bar_width, label='Accuracy', color='skyblue')
plt.bar([i + bar_width for i in x], f1_scores, width=bar_width, label='F1 Score', color='lightgreen')
plt.xlabel('Boosting Models')
plt.ylabel('Score')
plt.title('Accuracy vs F1 Score Comparison')
plt.xticks([i + bar_width / 2 for i in x], models)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
