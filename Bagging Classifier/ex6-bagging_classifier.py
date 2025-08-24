'''
bagging - bootstrap aggregating
ensemble - combining multiple models
bagging classifier is an ensemble learning that aims to improve model accuracy 
The core idea behind bagging is to create multiple training sets with replacement
less prone to overfit and underfit
for classification the final prediction is done by majority voting 
in regression the final prediction is done by averaging the predictions of base model
it is mainly used to reduce variance
it is done by taking random subsets of original dataset with replacement
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


#EDA
df = pd.read_csv(r"D:\ML Lab\diabetes_prediction_dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['diabetes'].value_counts())
num_cols = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']

#encoding
df['gender'] = df['gender'].map({'Male':0,'Female':1})
ohe = OneHotEncoder(drop='first',sparse_output=False) # normal dataframe instead of sparse matrix
smoking_history= ohe.fit_transform(df[['smoking_history']])
new_smoking_history = ohe.get_feature_names_out(['smoking_history'])#gets new feature name for smoking_history
new_smoking_history_df = pd.DataFrame(smoking_history,columns=new_smoking_history)#converting numpy array to dataframe
df = pd.concat([df,new_smoking_history_df],axis = 1)
df = df.drop('smoking_history',axis=1)#removes the original smoking_history column
print(df.head())

#splitting data into training and testing
X = df.drop('diabetes', axis=1) 
y = df['diabetes']# target variable
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_prediction = model.predict(X_test)

#HyperParameter tuning using gridsearch
param_grid = {'max_depth': [3, 5, 7, 9, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'criterion': ['gini', 'entropy']}
#3,5,7,9,10 --> controls how deep the tree can grow , test depth with 3 5 7 9 10
# 2,5,10 --> the minimum number of sample required to split the node,splits the node only when it has 2 5 10
#1,2,4 --> minimum number of sample leaf node must have
base_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=base_model,param_grid=param_grid,scoring='f1',cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_prediction = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_) 

#Printing Decision Tree
f1_scores = []
depths = range(1, 11)
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    print(f"Depth: {depth}, F1 Score: {f1:.4f}")    
best_depth = depths[f1_scores.index(max(f1_scores))]
model_decisiontree = DecisionTreeClassifier(max_depth=best_depth,random_state=42)
model_decisiontree.fit(X_train, y_train)

#confusion matrix
y_pred = model_decisiontree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
f1_score_dt= f1_score(y_test,y_pred)
accuracy_score_dt = accuracy_score(y_test,y_pred)
print('F1score for decision tree:',f1_score_dt)
print("Accuracy for decision tree:",accuracy_score_dt)

#Bagging classifier
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=best_depth, random_state=42), n_estimators=49,# Number of trees 
    oob_score=True,#Out-Of-Bag scoring
    random_state=42
)
bag_model.fit(X_train, y_train)
y_bagging_pred = bag_model.predict(X_test)
bag_acc = accuracy_score(y_test, y_bagging_pred)
bag_f1_score = f1_score(y_test,y_bagging_pred)
print("Bagging Classifier Accuracy:", bag_acc)
print("F1 score of bagging classifier:",bag_f1_score)
cm_bagging = confusion_matrix(y_test, y_bagging_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bagging,display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Oranges')
plt.title("Confusion Matrix - Bagging")
plt.show()
numeric_cols = df.select_dtypes(include=['number'])
X = numeric_cols.drop('diabetes', axis=1)
y = numeric_cols['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)