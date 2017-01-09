# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 03:41:53 2016

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

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(train_input, train_class)
sc1 = rfc.score(test_input, test_class)

model_accuracies = []
for i in range(100):
    (train_input, test_input, train_class, test_class) = train_test_split(all_inputs, all_classes, train_size = 0.75)
    rfc.fit(train_input, train_class)
    sc = rfc.score(test_input, test_class)
    model_accuracies.append(sc)
    
sb.distplot(model_accuracies)

from sklearn.cross_validation import cross_val_score

cvs = cross_val_score(rfc, all_inputs, all_classes, cv = 20)
sb.distplot(cvs)
acc = np.mean(cvs)


from sklearn.cross_validation import StratifiedKFold
c_v = StratifiedKFold(all_classes, n_folds = 20)
print("HEY")
from sklearn.grid_search import GridSearchCV
p = {'n_estimators' : [ 9, 10, 11, 12] }
gs = GridSearchCV(rfc, param_grid = p, cv = c_v)
gs.fit(all_inputs, all_classes)
rfc = gs.best_estimator_
acc2 = gs.best_score_
pa = gs.best_params_