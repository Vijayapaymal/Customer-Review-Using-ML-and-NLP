import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\Notes\FSDS-PS\19-06-2023 Customer Review prj , ChatBot,Imp sites\Lab\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model using TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training various classification algorithms
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_ac = accuracy_score(y_test, logreg_pred)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_ac = accuracy_score(y_test, knn_pred)

# Random Forest
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
randomforest_pred = randomforest.predict(X_test)
randomforest_ac = accuracy_score(y_test, randomforest_pred)

# Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
decisiontree_pred = decisiontree.predict(X_test)
decisiontree_ac = accuracy_score(y_test, decisiontree_pred)

# Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_ac = accuracy_score(y_test, svm_pred)

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_ac = accuracy_score(y_test, xgb_pred)

# Print accuracy scores
print("Logistic Regression Accuracy:", logreg_ac)
print("KNN Accuracy:", knn_ac)
print("Random Forest Accuracy:", randomforest_ac)
print("Decision Tree Accuracy:", decisiontree_ac)
print("SVM Accuracy:", svm_ac)
print("XGBoost Accuracy:", xgb_ac)
