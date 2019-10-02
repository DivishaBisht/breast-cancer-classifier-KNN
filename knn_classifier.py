# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:02:14 2019

@author: Divisha
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')

#replacing missing values
df.replace('?',-99999, inplace=True)   
#remove useless id column  
df.drop(['id'], 1, inplace=True)     
   
#storing features and label
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])               

#splitting train set and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#creating sample validation set
example_measures = np.array([4,2,1,1,1,3,2,2,1]).reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)





