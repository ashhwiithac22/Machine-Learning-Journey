'''naive bayes uses bayes theorem , it is based on probability
types of naive bayes:
    1)gaussian naive bayes - continous values , normal distribution with bell shaped curve,used in medical diagnosis
    2)multinomial naive bayes - discrete data - mainly used in text classification,commonly used in nlp 
    3)bernoulli is used when the dataset is binary ie it may contains yes/no or 0's and 1's,true or false 
'''

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

#eda
df = pd.read_csv(r"D:\ML Lab\diabetes_prediction_dataset.csv")
print("Initial data:")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['diabetes'].value_counts())

#outlier removal
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
for col in num_cols:
    # Boxplot before removing outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} - Before Removing Outliers")
    plt.show()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} - After Removing Outliers")
    plt.show()
df = df.reset_index(drop=True)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

#plot for identiying the type of naive bayes model
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# encoding
ohe = OneHotEncoder(drop='first', sparse_output=False)
smoking_transformed = ohe.fit_transform(df[['smoking_history']])
smoking_cols = ohe.get_feature_names_out(['smoking_history'])
smoking_df = pd.DataFrame(smoking_transformed, columns=smoking_cols)
df = pd.concat([df.drop('smoking_history', axis=1), smoking_df], axis=1)
df = df.dropna()
df = df.fillna(df.mean(numeric_only=True))
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#building model with all features
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score using all features with Gaussian Naive Bayes:", round(f1, 4))

#building model with selected features using correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
selected_features = ['gender', 'age', 'HbA1c_level', 'blood_glucose_level', 'smoking_history_never']
X = df[selected_features]
y = df['diabetes']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gnb = GaussianNB()
f1 = cross_val_score(gnb, X_scaled, y, scoring=make_scorer(f1_score), cv=5)
print("F1 Scores for each fold:", f1)
print("Average F1 Score with performance tuning:", round(f1.mean(), 4))



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r"spam.csv", encoding='latin-1') #latin-1 avoids reading special characters and errors
df = df[['Category', 'Message']]
df.columns = ['label', 'message']
print(df.head())

#encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['label']) #coverts ham to 0 and spam to 1
print("\nMissing values:\n", df.isnull().sum())
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(df['message'])
y = df['label']
word_freq = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())#converts sparse matrix into 2D array
word_freq['label'] = y
words = word_freq.drop('label', axis=1)  # Only words, no label
total_word_freq = words.sum()  # Total count of each word
sorted_words = total_word_freq.sort_values(ascending=False)
top_20 = sorted_words.head(20)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_20.values, y=top_20.index, color='blue')
plt.title("Top 20 Word Frequencies")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.grid(True)
plt.tight_layout()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
f1_scores = cross_val_score(model, X, y, scoring='f1', cv=5)
print("\nCross-Validated F1 Scores:", f1_scores)
print("Average F1 Score:", round(f1_scores.mean(), 4))
print("\nSpam/Ham Prediction")
user_input = input("Enter a message: ")
vectorized_input = cv.transform([user_input])
prediction = model.predict(vectorized_input)[0]
result = 'SPAM' if prediction == 1 else 'HAM'
print("Prediction:", result)
