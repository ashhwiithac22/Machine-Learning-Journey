'''
decision trees -used for making decisions, it is a supervised algorithm, works as flowchart to make decisions step by step. it is a bunch of if-else statements
it is non parametric test - makes no assumptions about the data distribution
two types of decision trees - 1) classification tree 2) regression tree
1)classification tree - used for predicting categorical outcomes like spam or not spam
2) regression trees - used for predicting continuos outcomes like predicting house prices , instead of assigning categories it assigns numerical predictions

decision node - nodes we get after splitting the root nodes are called decision node
leaf node - nodes where further splitting is not possible , final decision and prediction is made
pruning - avoids overfitting,reduces the size of trees of decision trees by removing parts that do not have power in predicting target variables
pre pruning - stops too early 
post pruning - grows the full tree first , then remove the branches

tree height :
*depth too small - underfitting,model too simple
*depth too large - underfitting,model too complex

Decision tree Working:
from the root the tree asks a series of questions like yes or no if yes the tree follows path , if no the tree follows another path
the process ends when there are no useful questions to ask 

common splitting criteria in decsion trees:
common splitting criteria include 1)Entropy 2)Information Gain
entropy -use high information gain(sees between two trees), measures uncertainity if entropy = 0 then it is pure , if higher = more confused
purity - the group has same class
impurity - the group has different class
Algorithm - Decision tree:
step1) - first find the entropy for complete dataset and find entropy for particular column and for features then find the information gain
choose the column which has higher information gain

bias(underfitting) - model is too simple and cannot capture the patterns in the data well,it makes lot of mistakes in both training and testing data
variance (overfitting)- model is too complex , performs well on training data and poor on test data 
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

#EDA
df = pd.read_csv(r"D:\ML Lab\diabetes_prediction_dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['diabetes'].value_counts())
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']



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
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[['params', 'mean_test_score']])
 
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
plt.figure(figsize=(8, 5))
sns.lineplot(x=depths, y=f1_scores, marker='o', color='green')
plt.title("Tree Depth vs F1 Score")
plt.xlabel("Tree Depth")
plt.ylabel("F1 Score")
plt.grid(True)
plt.tight_layout()
plt.show()

    
best_depth = depths[f1_scores.index(max(f1_scores))]
model_decisiontree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
model_decisiontree.fit(X_train, y_train)
plt.figure(figsize=(20,10))
plot_tree(model_decisiontree, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
plt.savefig("tree.png", dpi=300, bbox_inches='tight')
plt.show()

#confusion matrix
y_pred = model_decisiontree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

