# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 04:57:53 2016

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

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_input, train_class)
acc1 = gnb.score(test_input, test_class)

nb_acc = []
for i in range(100):
    (t_in, ts_in, t_c, ts_c) = train_test_split(all_inputs, all_classes, train_size = 0.75)
    gnb.fit(t_in, t_c)
    acc = gnb.score(ts_in, ts_c)
    nb_acc.append(acc)
    
sb.distplot(nb_acc)
plt.show()

from sklearn.cross_validation import cross_val_score

cs = cross_val_score(gnb, all_inputs, all_classes, cv = 10)
sb.distplot(cs)
plt.show()