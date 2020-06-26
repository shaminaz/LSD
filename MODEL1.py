# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:58:59 2019

@author: ranju
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
from sklearn.preprocessing import LabelEncoder

#Read the training & test data
liver_df = pd.read_csv('D:\ML project\ilp.csv')
liver_df.head()
liver_df.info()
liver_df.describe(include='all')
liver_df.columns

liver_df.isnull().sum()

sns.countplot(data=liver_df, x = 'Dataset', label='Count')
LD, NLD = liver_df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)

sns.countplot(data=liver_df, x = 'Gender', label='Count')
M, F = liver_df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

liver_df.head(3)

pd.get_dummies(liver_df['Gender'], prefix = 'Gender').head()
liver_df = pd.concat([liver_df,pd.get_dummies(liver_df['Gender'], prefix = 'Gender')], axis=1)
liver_df.head()

liver_df.describe()

liver_df[liver_df['Albumin_and_Globulin_Ratio'].isnull()]

liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())
#liver_df[liver_df['Albumin_and_Globulin_Ratio'] == 0.9470639032815201]
# The input variables/features are all the inputs except Dataset. The prediction or label is 'Dataset' that determines whether the patient has liver disease or not. 
X = liver_df.drop(['Gender','Dataset'], axis=1)
X.head(3)


y = liver_df['Dataset'] # 1 for liver disease; 2 for no liver disease
# Correlation
liver_corr = X.corr()
liver_corr

# Importing modules
 
from sklearn.ensemble import RandomForestClassifier 
 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
X = liver_df.iloc[:, :10]
y = liver_df.iloc[:, -1]
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)
#Predict Output
#rf_predicted = random_forest.predict(X_test)

#random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
#random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
#print('Random Forest Score: \n', random_forest_score)
#print('Random Forest Test Score: \n', random_forest_score_test)
#print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
#print(confusion_matrix(y_test,rf_predicted))
#print(classification_report(y_test,rf_predicted))

# Saving model to disk
pickle.dump(random_forest, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))

#print(model)