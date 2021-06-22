# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:08:49 2021

@author: Hp
"""

#standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree

url = url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/diabetes.csv'

data = pd.read_csv(url)
data.head()
data.columns
data.groupby('Outcome').aggregate({'Glucose':np.mean , 'BMI':np.mean, 'Age':np.mean})
data.Outcome.value_counts()
data.shape
X = data.drop('Outcome', axis=1) # Features 
y = data['Outcome'] # Target variable : has diabetes =1
X
y
#%%%split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #doubt
X_train.shape, X_test.shape
y_train.shape, y_test.shape
# 70% training and 30% test : each for train and test (X & y)
X_train.head()
clf = DecisionTreeClassifier()
help(clf)
# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)
y_train
#predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred
y_test
# accuracy
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)
len(y_test)
(118+43)/231


#prune tree
from sklearn.metrics import classification_report

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3)# doubt
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
text_representation2 = tree.export_text(dtree)
print(text_representation2)
data.columns
text_representation3 = tree.export_text(dtree, feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],  decimals=0, show_weights=True, max_depth=3)  #keep changing depth values
print(text_representation3)
