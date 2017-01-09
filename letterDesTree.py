# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 02:23:53 2016

@author: Sachi Angle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

letter_data = pd.read_csv('letter-recognition.csv', na_values = ['NA'])
all_inputs = letter_data[['xbox','ybox','width','high','onpx','xbar','ybar','x2bar','y2bar','xybar','x2ybr', 'xy2br','xede', 'xegvy', 'yege', 'yegvx']].values
all_classes = letter_data['lttr'].values

from sklearn.cross_validation import train_test_split

(train_input, test_input, train_class, test_class) = train_test_split(all_inputs, all_classes, train_size = 0.75, random_state = 1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(train_input, train_class)
sc1 = dtc.score(test_input, test_class)

model_accuracies = []
for i in range(100):
    (train_input, test_input, train_class, test_class) = train_test_split(all_inputs, all_classes, train_size = 0.75)
    dtc.fit(train_input, train_class)
    sc = dtc.score(test_input, test_class)
    model_accuracies.append(sc)

sb.distplot(model_accuracies)

from sklearn.cross_validation import cross_val_score

cvs = cross_val_score(dtc, all_inputs, all_classes, cv = 10)
sb.distplot(cvs)

"""from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

p = {'max_features':[10,11,12,13,14,15,16]}
c_v = StratifiedKFold(all_classes, n_folds = 10)

grid_search = GridSearchCV(dtc, param_grid = p, cv = c_v)
grid_search.fit(all_inputs,all_classes)
dtc = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_"""

    