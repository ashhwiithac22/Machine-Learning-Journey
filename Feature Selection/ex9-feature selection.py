'''
featrure selection is the process of choosing subset
irrelevant features can introduce noise 
models with fewer features are easier to understand
filter methods - 
evaluates each feature independently based on target variable

selects features based on statistical measures like correlation ,chi squared 
wrapper - evaluates subset of features : forward selection and backward elimination
embedded - integrate feature selection as a part of model training process - lasso regression

logistic regression is a supervised algorithm used for binary classfication
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("diabetes_prediction_dataset.csv")
label_enc = LabelEncoder() #converts categorical to numerical 
for col in df.columns:
    if df[col].dtype == "object":   # if column is text
        df[col] = label_enc.fit_transform(df[col])
X = df.drop("diabetes", axis=1)   # all columns except target
y = df["diabetes"]                # target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
chi square works with categorical and non numerical data
if the value is small, then feature and target is independent (not useful)
if the value is high, the feature and target is dependent(most important feature) 
(observed - expected)^2 / expected
'''

chi_selector = SelectKBest(score_func=chi2, k=5) #k=5 selects top 5 best features
X_train_chi_sq = chi_selector.fit_transform(X_train, y_train)
X_test_chi_sq = chi_selector.transform(X_test)
chi_sq_features = X.columns[chi_selector.get_support()]#get support is used to get the names of selected features 
print("Chi-Square Selected Features:", chi_sq_features.tolist())
model_chi_sq = LogisticRegression(max_iter=2000)
model_chi_sq.fit(X_train_chi_sq, y_train)
y_pred_chi = model_chi_sq.predict(X_test_chi_sq)
print("Chi-Square Accuracy:", accuracy_score(y_test, y_pred_chi))

'''
anova works with numeric features
Big difference = good feature, small difference = bad feature
selects feature with high f-score
'''

anova_model = SelectKBest(score_func=f_classif, k=5)
X_train_anova = anova_model.fit_transform(X_train, y_train)
X_test_anova = anova_model.transform(X_test)
anova_features = X.columns[anova_model.get_support()]
print("ANOVA Selected Features:", anova_features.tolist())
model_anova = LogisticRegression(max_iter=2000)
model_anova.fit(X_train_anova, y_train)
y_pred_anova = model_anova.predict(X_test_anova)
print("ANOVA Accuracy:", accuracy_score(y_test, y_pred_anova))

'''
Lasso automatically selects features while training the model
Features with zero coefficients are removed, and the rest are selected as important
If a coefficient = 0 , feature is removed (not useful).
If a coefficient not 0, that feature is kept (important).
'''

base_model = LogisticRegression(max_iter=2000)
rfe = RFE(base_model, n_features_to_select=5) #iteratively removes least significant features 
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
base_model.fit(X_train_rfe, y_train)
y_pred_rfe = base_model.predict(X_test_rfe)
print("RFE Selected Features:", X.columns[rfe.support_].tolist())
print("Wrapper (RFE) Accuracy:", accuracy_score(y_test, y_pred_rfe))

'''
uses l1 regularization to shrink some feature coefficients to zero 
'''

lasso = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000) #liblinear is used to calculate weights
lasso.fit(X_train, y_train)
coef = lasso.coef_[0] #extracts coefficients of features
lasso_features = X.columns[coef != 0] #selects the names of features whose coefficients are non zero
print("Lasso Selected Features:", lasso_features.tolist())
X_train_lasso = X_train[lasso_features]
X_test_lasso = X_test[lasso_features]
lasso.fit(X_train_lasso, y_train)
y_prediction_lasso = lasso.predict(X_test_lasso)
print("Embedded (Lasso regression) Accuracy:", accuracy_score(y_test, y_prediction_lasso))

#comparison of models
accuracy_dict = {
    "Chi-Square": accuracy_score(y_test, y_pred_chi),
    "ANOVA": accuracy_score(y_test, y_pred_anova),
    "RFE (Wrapper)": accuracy_score(y_test, y_pred_rfe),
    "Lasso (Embedded)": accuracy_score(y_test, y_prediction_lasso)
}
best_method = max(accuracy_dict, key=accuracy_dict.get)
best_accuracy = accuracy_dict[best_method]
print(f"Best Feature Selection Method: {best_method} with Accuracy = {best_accuracy:.4f}")
