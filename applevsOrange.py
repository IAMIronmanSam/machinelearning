# -*- coding: utf-8 -*-
"""
Predict fruit types,
    - Apple - Smooth surface
    - orange - Bumpy surface

Feature to compare:
    - Surface - Good feature
    - Weight - Not good feature
    
Created on Sat Dec 03 14:30:51 2016

@author: Arivu
"""
import sklearn
from sklearn import tree
#1 - Smooth 0 - Bump
features = [[140,1],[170,0],[130,1],[150,0]]
# 1 - Apple 0 - Orange
labels = [1,0,1,0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print (clf.predict([[120,1],[160,0]]))