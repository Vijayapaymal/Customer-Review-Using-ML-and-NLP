# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\Notes\FSDS-PS\19-06-2023 Customer Review prj , ChatBot,Imp sites\Lab\Restaurant_Reviews.tsv",delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
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


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("RandomForestClassifier")
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("AC=",ac)
  
bias = classifier.score(X_train,y_train)
print("bias=",bias)

variance = classifier.score(X_test,y_test)
print("variance=",variance,"\n----------------")
#===================KNN==================

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train, y_train)

y_pred = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier")
print("AC=",ac)
  
bias = classifier_knn.score(X_train,y_train)
print("bias=",bias)

variance = classifier_knn.score(X_test,y_test)
print("variance=",variance,"\n----------------")

#===================Logistic Regression===========

from sklearn.linear_model import LogisticRegression
classifier_logit = LogisticRegression()
classifier_logit.fit(X_train, y_train)

y_pred = classifier_logit.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("LogisticRegression")
print("AC=",ac)
  
bias = classifier_logit.score(X_train,y_train)
print("bias=",bias)

variance = classifier_logit.score(X_test,y_test)
print("variance=",variance,"\n----------------")

#==============SVM============

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("SVC")
print("AC=",ac)
  
bias = classifier.score(X_train,y_train)
print("bias=",bias)
variance = classifier.score(X_test,y_test)
print("variance=",variance,"\n----------------")

#==============Decision Tree===========

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)

y_pred = classifier_dt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("DecisionTreeClassifier")
print("AC=",ac)
  
bias = classifier_dt.score(X_train,y_train)
print("bias=",bias)

variance = classifier_dt.score(X_test,y_test)
print("variance=",variance,"\n----------------")

#==============NB================

from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB() 
classifier_nb.fit(X_train, y_train)

y_pred = classifier_nb.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("GaussianNB")
print("AC=",ac)
  
bias = classifier_nb.score(X_train,y_train)
print("bias=",bias)

variance = classifier_nb.score(X_test,y_test)
print("variance=",variance,"\n----------------")

#=============== XgBoost==============

from xgboost import XGBClassifier
classifier = XGBClassifier() 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("XGBClassifier")
print("AC=",ac)
  
bias = classifier.score(X_train,y_train)
print("bias=",bias)
variance = classifier.score(X_test,y_test)
print("variance=",variance,"\n----------------")
