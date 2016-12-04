# -*- coding: utf-8 -*-
"""
Predict dog types,
    - GreyHound - Max 28' tall
    - Labrador - Max 24' tall

Feature to compare:
    - Height - Good feature
    - Eye Color - Not good feature
    
Created on Sat Dec 03 14:30:51 2016

@author: Arivu
"""

import numpy as np
import matplotlib.pyplot as plt

greyhound = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhound)
labs_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, labs_height],stacked=True,color=['r','g'])
plt.show()