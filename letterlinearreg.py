# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:45:53 2016

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

(train_input, test_input, train_class_temp, test_class_temp) = train_test_split(all_inputs, all_classes, train_size = 0.75, random_state = 1)

dic = {"A" : 1, "B" : 2, "C" : 3,  "D" : 4, "E" : 5, "F" : 6, "G" : 7, "H" : 8, "I" : 9, "J" : 10, "K" : 11, "L" : 12, "M" : 13, "N" : 14, "O" : 15, "P" : 16, "Q" : 17, "R" : 18, "S" : 19, "T" : 20, "U" : 21, "V" : 22, "W" : 23, "X" : 24, "Y" : 25, "Z" : 26}

train_class = []
test_class = []
for i in train_class_temp:
    train_class.append(dic[i])
for i in test_class_temp:
    test_class.append(dic[i])

from sklearn.linear_model import RidgeCV
alphas = np.arange(0.1, 100000, 2)

lrc = RidgeCV(alphas)
lrc.fit(train_input, train_class)
coeff = lrc.coef_
inter = lrc.intercept_
sc1 = lrc.score(test_input, test_class)
